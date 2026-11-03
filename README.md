# Ring all-reduce

A C++20 implementation of synchronous ring all-reduce, connected to full-batch
training of a shallow ReLU/softmax classifier. Workers are **separate processes**;
gradients travel only between ring neighbors. There is no central gradient server.

Requires Linux or macOS, a C++20 compiler, CMake 3.16+, and Python 3.10+ for the
launcher/tests. Builds require no downloaded dependencies or environment files.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure

# Four processes, three verified all-reduces of a 17-element vector.
python3 scripts/run_local.py --workers 4

# Train on the bundled 15-example, two-class dataset (uneven worker shards).
python3 scripts/run_local.py --workers 4 -- \
  data/toy.csv --input 2 --hidden 8 --classes 2 \
  --rounds 100 --lr 0.1 --threshold 0 --output build/model.json
```

The demo prints `all_reduce=ok` from every rank after checking every element.
The training command reports global mean loss and training accuracy on rank zero,
and optionally writes that rank's final model parameters to JSON. On the bundled
toy dataset, the 100-update command reaches loss about `0.01805` and training
accuracy `1.0`. This tiny dataset is a correctness example, not an evaluation of
model generalization or distributed speedup.

`make`, `make test`, `make demo`, and `make train` wrap these commands. Run
`build/train --help` for worker options and `python3 scripts/run_local.py --help`
for launcher options. Without arguments the launcher runs the verified vector
demo. Arguments after `--` go to each worker. Launcher output is collected per
rank and printed when the group finishes; it stops and reaps peers on failure,
interrupt, or its 60-second default deadline (`--timeout` changes that deadline).

The local launcher defaults to inherited Unix stream socketpairs (`--transport
ipc`). This uses the same framing, nonblocking send/receive code, and collective
algorithm as TCP while avoiding local port allocation. To use TCP locally:

```sh
python3 scripts/run_local.py --transport tcp --workers 4
# Or select a fixed consecutive port range:
python3 scripts/run_local.py --transport tcp --workers 4 --base-port 5000 -- \
  --demo --elements 1000003 --rounds 3
```

On a Mac where the selected Xcode installation refuses to run because its license
has not been accepted, an already installed Command Line Tools toolchain can be
selected for these commands without changing system settings:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang++ \
  -DCMAKE_MAKE_PROGRAM=/Library/Developer/CommandLineTools/usr/bin/make \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build --parallel 4
```

## Training data

CSV files have one header row, exactly `--input` numeric feature columns, then
one zero-based integer label column. Labels must be less than `--classes`;
features must be finite. Quoted cells and missing values are unsupported.
Normalize features before training. For example:

```csv
x1,x2,label
-1,-1,0
1,1,1
```

Every rank loads an identical full dataset and takes rows whose index modulo
worker count equals its rank. Workers check model/training settings and a dataset
content fingerprint at startup. Local shards can be uneven or empty; the global
dataset must be nonempty. All ranks initialize from the same seed, compute local
**gradient sums**, and reduce gradients, loss sum, and sample count in one vector.
Updates divide by the **global sample count**, so a smaller shard has the correct
weight. Global loss controls early stopping consistently across ranks. The final
loss is recomputed after the last update.

Defaults preserve the original classifier shape: 64 inputs, 64 hidden units,
10 classes, seed 42, learning rate 0.05, at most 500 updates, and mean-loss
threshold 0.01. `build/train DATA.csv` runs the single-worker baseline. Model
parameters and local gradients use float; collective accumulation uses double.
Summation order differs between worker counts, so compare with a tolerance.

## How the ring works

Rank `r` sends to `(r + 1) % P` and receives from `(r - 1 + P) % P`.
A vector is divided into `P` balanced contiguous chunks; empty chunks are valid.

1. **Reduce-scatter, `P - 1` steps:** at step `s`, rank `r` sends chunk
   `(r - s + P) % P` and sums the received predecessor chunk. Each rank finishes
   with one completely reduced chunk, indexed `(r + 1) % P`.
2. **All-gather, `P - 1` steps:** forward the completed chunk and then each newly
   received chunk until every rank has the full sum.

For evenly sized chunks, each rank sends approximately `2(P-1)N/P` data elements
per collective, plus framing metadata. `P = 1` is a no-op. Collectives must occur
in the same order with the same vector length on all ranks. Empty vectors still
exchange metadata when `P > 1`.

Messages use the existing 30-byte, network-byte-order header. Chunk payloads
contain a 64-bit collective sequence, a 64-bit vector length, and big-endian
double values. Receivers validate magic/version, payload limits, sender, phase,
step, chunk index, element count, sequence, and vector length. A payload is limited
to 16 MiB, so the largest chunk holds at most 2,097,150 doubles. Larger vectors
need enough ranks to stay within that limit. Send and receive progress together
with `poll`, including partial I/O, so chunks larger than socket buffers do not
create a ring-wide send deadlock.

Setup and each message exchange have a deadline (`--timeout-ms`, default 10000).
A disconnect, malformed message, mismatched collective, or timeout fails the run;
a failed ring cannot be reused. There is no worker restart/recovery or elastic
membership. The launcher supplies fixed membership; the protocol's reserved
coordinator message types do not imply an implemented coordinator service.

## Multiple hosts over TCP

Start one worker per rank with identical options and dataset contents. Supply a
numeric IPv4 address for every rank in ring order. Rank `r` listens on
`base_port + r`; each host must permit the required inbound TCP port. Example
for two machines (replace these documentation addresses with actual addresses):

```sh
# On host 192.0.2.10:
build/train --demo --rounds 3 --workers 2 --rank 0 \
  --base-port 5000 --hosts 192.0.2.10,192.0.2.11
# On host 192.0.2.11:
build/train --demo --rounds 3 --workers 2 --rank 1 \
  --base-port 5000 --hosts 192.0.2.10,192.0.2.11
```

TCP listeners bind all IPv4 interfaces. This is a learning implementation for a
trusted network: it has no authentication, encryption, or hostile-peer isolation.
There is no MPI, GPU support, overlap of training and communication, or performance
claim. Multi-host behavior has not been validated in the development environment.

## Verification

```sh
# Unit tests + separate-process IPC integration tests (default).
ctest --test-dir build --output-on-failure

# Add the same integration checks over real localhost TCP where ports are allowed.
cmake -S . -B build -DENABLE_TCP_TESTS=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure

# Memory/undefined-behavior checks.
cmake -S . -B build-asan -DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-asan --parallel 4
ctest --test-dir build-asan --output-on-failure
```

Tests cover byte order, malformed/truncated messages and overflowing offsets,
known sums at 1/2/3/4/7 ranks, repeated collectives, empty and uneven chunks,
1,000,003-element vectors, peer disconnects/stalls, collective/configuration
mismatches, invalid CSVs, empty shards, and global early stopping. Distributed
training at 2 and 4 ranks is compared with a single-rank baseline after 100
updates (maximum absolute parameter difference below `2e-6`). The test runner
uses only the C++ standard library and Python standard library. Temporary test
directories may remain under the build directory if the host denies directory
cleanup.

The development sandbox blocks binding TCP ports even on an elevated retry;
local IPC is the verified transport here. Enable the TCP suite on an unrestricted
host to validate TCP connection setup and networking.
