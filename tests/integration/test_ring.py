#!/usr/bin/env python3
"""Exercise separate ring processes and compare training with a serial baseline."""
import argparse
import json
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_local import available_port, local_sockets, run_workers  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True)
parser.add_argument("--worker", required=True)
parser.add_argument("--transport", choices=("ipc", "tcp"), default="ipc")
ARGS, remaining = parser.parse_known_args()


class RingIntegration(unittest.TestCase):
    def group(self, workers, make_command, timeout=30):
        port = available_port(workers) if ARGS.transport == "tcp" else 5000
        inherited = local_sockets(workers) if ARGS.transport == "ipc" else None
        commands = []
        for rank in range(workers):
            command = make_command(rank, port)
            if inherited is not None and inherited[rank]:
                incoming, outgoing = inherited[rank]
                if command[0] == ARGS.worker:
                    command.extend([str(incoming.fileno()), str(outgoing.fileno())])
                else:
                    command.extend(["--incoming-fd", str(incoming.fileno()), "--outgoing-fd", str(outgoing.fileno())])
            commands.append(command)
        return run_workers(commands, timeout=timeout, inherited=inherited)

    def worker_group(self, workers, mode="success"):
        return self.group(workers, lambda rank, port: [ARGS.worker, str(rank), str(workers), str(port), mode])

    def training(self, workers, data, output, extra=()):
        return self.group(workers, lambda rank, port:
                          [ARGS.train, str(data), "--rank", str(rank), "--workers", str(workers),
                           "--base-port", str(port), "--input", "2", "--hidden", "8", "--classes", "2",
                           "--rounds", "100", "--lr", "0.1", "--threshold", "0", "--output", str(output), *extra])

    def assert_success(self, results):
        for code, output in results:
            self.assertEqual(code, 0, output)

    def test_known_sums_many_sizes_and_repeated_collectives(self):
        for workers in (1, 2, 3, 4, 7):
            with self.subTest(workers=workers):
                results = self.worker_group(workers)
                self.assert_success(results)
                self.assertTrue(all("verified rank=" in log for _, log in results))

    def test_training_matches_serial_baseline(self):
        with tempfile.TemporaryDirectory(dir=Path(ARGS.train).resolve().parent, ignore_cleanup_errors=True) as directory:
            directory = Path(directory)
            serial_path = directory / "serial.json"
            baseline = self.training(1, ROOT / "data/toy.csv", serial_path)
            self.assert_success(baseline)
            serial = json.loads(serial_path.read_text())["parameters"]
            for workers in (2, 4):
                with self.subTest(workers=workers):
                    model_path = directory / f"distributed-{workers}.json"
                    results = self.training(workers, ROOT / "data/toy.csv", model_path)
                    self.assert_success(results)
                    parameters = json.loads(model_path.read_text())["parameters"]
                    self.assertEqual(len(serial), len(parameters))
                    self.assertLess(max(abs(a - b) for a, b in zip(serial, parameters)), 2e-6)
                    summary = re.search(r"training complete updates=(\d+) loss=([\d.e+-]+) accuracy=([\d.e+-]+)", results[0][1])
                    self.assertIsNotNone(summary, results[0][1])
                    self.assertEqual(int(summary[1]), 100)
                    self.assertLess(float(summary[2]), 0.05)
                    self.assertEqual(float(summary[3]), 1.0)

    def test_workers_reject_different_datasets(self):
        with tempfile.TemporaryDirectory(dir=Path(ARGS.train).resolve().parent, ignore_cleanup_errors=True) as directory:
            directory = Path(directory)
            paths = [directory / "a.csv", directory / "b.csv"]
            paths[0].write_text("x1,x2,label\n-1,-1,0\n1,1,1\n")
            paths[1].write_text("x1,x2,label\n-1,-1,0\n0.5,1,1\n")
            result = self.group(2, lambda rank, port:
                                [ARGS.train, str(paths[rank]), "--rank", str(rank), "--workers", "2", "--base-port", str(port),
                                 "--input", "2", "--hidden", "8", "--classes", "2"])
            self.assertTrue(any(code != 0 for code, _ in result), result)
            self.assertTrue(any("different run configurations" in log for _, log in result), result)

    def test_training_with_empty_shards_and_collective_early_stop(self):
        with tempfile.TemporaryDirectory(dir=Path(ARGS.train).resolve().parent, ignore_cleanup_errors=True) as directory:
            directory = Path(directory)
            data = directory / "tiny.csv"
            data.write_text("x1,x2,label\n-1,-1,0\n1,1,1\n")
            result = self.training(4, data, directory / "tiny.json")
            self.assert_success(result)
            self.assertIn("samples=0", result[3][1])
            result = self.training(4, data, directory / "early.json", ("--threshold", "100"))
            self.assert_success(result)
            self.assertIn("updates=0", result[0][1])
            self.assertIn("reason=loss_threshold", result[0][1])

    def test_configuration_length_disconnect_and_stall_fail_promptly(self):
        for mode in ("configuration", "length", "disconnect", "stall"):
            with self.subTest(mode=mode):
                start = time.monotonic()
                results = self.worker_group(2, mode)
                self.assertTrue(any(code != 0 for code, _ in results), results)
                self.assertTrue(any("error:" in output for _, output in results), results)
                self.assertLess(time.monotonic() - start, 8)
                if mode == "stall":
                    self.assertTrue(any("timed out" in output for _, output in results), results)

    def test_missing_peer_times_out(self):
        if ARGS.transport == "ipc":
            # Keep the absent peer's endpoints open without starting that peer.
            inherited = local_sockets(2)
            incoming, outgoing = inherited[0]
            command = [ARGS.worker, "0", "2", "5000", "missing", str(incoming.fileno()), str(outgoing.fileno())]
            try:
                result = subprocess.run(command, pass_fds=(incoming.fileno(), outgoing.fileno()),
                                        capture_output=True, text=True, timeout=5)
            finally:
                for pair in inherited:
                    for endpoint in pair:
                        endpoint.close()
        else:
            port = available_port(2)
            result = subprocess.run([ARGS.worker, "0", "2", str(port), "missing"], capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("timed out", result.stderr)

    def test_malformed_peer_header_rejected_before_payload_read(self):
        header = struct.pack("!IHHIIHIII", 0x52414C52, 1, 1, 0xFFFFFFFF, 1, 0, 0, 0, 2)
        if ARGS.transport == "ipc":
            inherited = local_sockets(2)
            incoming, outgoing = inherited[0]
            process = subprocess.Popen([ARGS.worker, "0", "2", "5000", "success", str(incoming.fileno()), str(outgoing.fileno())],
                                       pass_fds=(incoming.fileno(), outgoing.fileno()), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                incoming.close()
                outgoing.close()
                inherited[1][1].sendall(header)
                _, error = process.communicate(timeout=5)
            finally:
                for pair in inherited:
                    for endpoint in pair:
                        endpoint.close()
                if process.poll() is None:
                    process.kill()
                process.communicate()
        else:
            port = available_port(2)
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", port + 1))
                listener.listen(1)
                listener.settimeout(5)
                process = subprocess.Popen([ARGS.worker, "0", "2", str(port), "success"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    outgoing, _ = listener.accept()
                    with outgoing, socket.create_connection(("127.0.0.1", port), timeout=5) as incoming:
                        incoming.sendall(header)
                        _, error = process.communicate(timeout=5)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Invalid protocol header", error)

    def test_launcher_and_invalid_csv(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_local.py"), "--binary", ARGS.train,
                                 "--workers", "3", "--transport", ARGS.transport, "--", "--demo", "--rounds", "2", "--elements", "11"],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("all_reduce=ok"), 3)
        with tempfile.TemporaryDirectory(dir=Path(ARGS.train).resolve().parent, ignore_cleanup_errors=True) as directory:
            directory = Path(directory)
            for content in ("x,y,label\n1,2,-1\n", "x,y,label\n1,2,0,extra\n", "x,y,label\nnan,2,0\n", "x,y,label\n", "x,y,label\n1x,2,0\n"):
                data = directory / "invalid.csv"
                data.write_text(content)
                result = self.training(2, data, directory / "unused.json")
                self.assertTrue(any(code != 0 for code, _ in result), result)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
