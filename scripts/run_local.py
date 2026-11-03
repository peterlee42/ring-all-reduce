#!/usr/bin/env python3
"""Launch separate ring workers over local IPC (default) or TCP."""
import argparse
import errno
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def available_port(workers):
    """Probe a contiguous range. The workers still handle bind races as errors."""
    for _ in range(100):
        sockets = []
        try:
            first = socket.socket()
            sockets.append(first)
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port + workers > 65536:
                continue
            for rank in range(1, workers):
                peer = socket.socket()
                sockets.append(peer)
                peer.bind(("127.0.0.1", port + rank))
            return port
        except OSError as error:
            if error.errno not in (errno.EADDRINUSE, errno.EADDRNOTAVAIL):
                raise
            continue
        finally:
            for peer in sockets:
                peer.close()
    raise RuntimeError("Could not find a free consecutive port range")


def local_sockets(workers):
    if workers == 1:
        return [()]
    links = []
    try:
        for _ in range(workers):
            links.append(socket.socketpair())
        return [(links[(rank - 1) % workers][1], links[rank][0]) for rank in range(workers)]
    except BaseException:
        for pair in links:
            for endpoint in pair:
                endpoint.close()
        raise


def run_workers(commands, timeout=60, inherited=None):
    """Return [(exit code, combined output)] or raise on timeout; always reap peers."""
    processes = []
    logs = []
    try:
        for rank, command in enumerate(commands):
            log = tempfile.TemporaryFile(mode="w+t")
            logs.append(log)
            sockets = inherited[rank] if inherited is not None else ()
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               pass_fds=tuple(peer.fileno() for peer in sockets)))
            for peer in sockets:
                peer.close()
        deadline = time.monotonic() + timeout
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Worker group exceeded {timeout}s")
            time.sleep(0.02)
        return_codes = [process.poll() for process in processes]
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        results = []
        for code, process, log in zip(return_codes, processes, logs):
            log.seek(0)
            results.append((process.returncode if code is None else code, log.read()))
        return results
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        for log in logs:
            log.close()
        if inherited is not None:
            for pair in inherited:
                for peer in pair:
                    peer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/train")
    parser.add_argument("--transport", choices=("ipc", "tcp"), default="ipc")
    parser.add_argument("--base-port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER, help="Arguments after -- go to train")
    args = parser.parse_args()
    if not 1 <= args.workers <= 128:
        parser.error("--workers must be between 1 and 128")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    extra = args.worker_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if not extra:
        extra = ["--demo", "--rounds", "3"]
    if any(value in extra for value in ("--rank", "--workers", "--base-port", "--hosts", "--incoming-fd", "--outgoing-fd")):
        parser.error("Ring membership is managed by the launcher")
    if not args.binary.is_file():
        parser.error(f"Binary does not exist: {args.binary}. Build with cmake first.")
    try:
        inherited = local_sockets(args.workers) if args.transport == "ipc" else None
        port = args.base_port if args.base_port is not None else (available_port(args.workers) if args.transport == "tcp" else 5000)
        commands = []
        for rank in range(args.workers):
            command = [str(args.binary.resolve()), "--rank", str(rank), "--workers", str(args.workers),
                       "--base-port", str(port), *extra]
            if inherited is not None and inherited[rank]:
                incoming, outgoing = inherited[rank]
                command.extend(["--incoming-fd", str(incoming.fileno()), "--outgoing-fd", str(outgoing.fileno())])
            commands.append(command)
        results = run_workers(commands, args.timeout, inherited)
    except (OSError, RuntimeError, TimeoutError, KeyboardInterrupt) as error:
        print(f"launcher error: {error}", file=sys.stderr)
        return 1
    for rank, (code, output) in enumerate(results):
        print(f"--- worker {rank} (exit {code}) ---\n{output}", end="", flush=True)
    return 0 if all(code == 0 for code, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
