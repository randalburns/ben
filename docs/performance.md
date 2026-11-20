# Performance changes in this fork

This fork plays a board 2.3 to 6.7 times faster than the BEN commit it started from
(`5098782e`, version 0.9.0.3). The factor depends on the hardware:

| Hardware | Best configuration | s/board before | s/board after | Speed-up |
|---|---|---|---|---|
| x86 laptop, CPU only | Keras networks (the default) | 23.4 | 10.1 | 2.3× |
| x86 laptop, CPU only | ONNX Runtime networks, one thread per core | 23.4 | 7.9 | 3.0× |
| Apple M5 | ONNX Runtime networks | 20.3 | 6.1 | 3.3× |
| x86 laptop with an NVIDIA GPU | ONNX Runtime networks on the GPU | 23.4 | 3.5 | 6.7× |

Bids and cards are unchanged, with the exceptions listed under
[Effect on decisions](#effect-on-decisions).

## How it was measured

- **Boards:** the 8 boards of `src/christy170a.pbn`, played by `game.py --auto True --seed 42`
  with the default configuration unless a step says otherwise.
- **Machines:**
  - **PC:** a laptop with a Ryzen AI 9 HX 370 (12 cores, 24 threads) and an RTX 5070 Laptop GPU
    (8 GB), Linux, on AC power in its Performance power profile.
  - **M5:** a MacBook with an Apple M5 (4 performance and 6 efficiency cores), 16 GB, macOS 26.
- **Runs:** each step was run twice, the second round in reverse order; the table gives the
  mean. The two rounds differed by at most 0.4 s/board on the PC (0.9 and 1.3 s/board for
  the two multi-threaded ONNX Runtime rows) and 0.8 s/board on the M5.
  On the PC the time is the mean of the "Board played in" lines; on the M5 it is the whole
  `game.py` run divided by 8, start-up included.
- **Decisions:** each run's output PBN was compared byte for byte with the others.

Eight boards is a small sample. The times are stable between rounds, but other boards will
weight bidding, sampling and play differently. Times also depend on the machine's state: in
its Balanced power profile the PC was about a third slower at every step.

## The changes, in the order applied

Each row adds one change to the row above.

| Step | Change | PC s/board | PC step | PC total | M5 s/board | M5 step | M5 total |
|---|---|---|---|---|---|---|---|
| 0 | BEN at `5098782e` (`dds3`, C# BGADLL on DDS) | 23.4 | | 1.0× | 20.3 | | 1.0× |
| 1 | ddfast for BEN's double-dummy solves | 21.0 | −10.1% | 1.11× | 19.0 | −6.2% | 1.07× |
| 2 | C++ BGADLL for PIMC | 19.8 | −5.9% | 1.18× | 16.6 | −12.5% | 1.22× |
| 3a | No GUI pauses, no sleep after PIMC | 16.7 | −15.9% | 1.41× | 13.3 | −19.8% | 1.52× |
| 3b | Sampling and claim-check speedups | 15.3 | −8.3% | 1.53× | 12.4 | −7.0% | 1.63× |
| 4a | Network scores cached during play sampling | 11.5 | −24.8% | 2.04× | 9.9 | −19.9% | 2.04× |
| 4b | Other repeated work removed | 10.1 | −11.7% | 2.31× | 9.0 | −9.1% | 2.24× |
| 5a | Networks on ONNX Runtime (CPU), 1 thread | 17.8 | +75.9% | 1.31× | 6.1 | −32.8% | 3.3× |
| | — the same with 12 threads, one per core | 7.9 | −22.1% from 4b | 2.98× | not measured | | |
| | — the same with 24 threads, all hardware threads | 11.0 | +9.3% from 4b | 2.12× | not measured | | |
| 5b | Networks on the NVIDIA GPU | 3.5 | −65.7% from 4b | 6.7× | not applicable | | |

Step 5a makes the PC slower and the M5 faster with BEN's setting of one thread per network
call; the reason, and the two rows with more threads, are under step 5a below. The PC's
step 3a is from one round: the laptop suspended during the other.

### 1. ddfast

`DDSolver` (`src/ddsolver/ddsolver.py`) runs on
[ddfast](https://github.com/randalburns/ddfast), a double-dummy solver with the `dds3`
interface, when it is built; otherwise it uses `dds3` as before. `BEN_DDS=dds3` or `ddfast`
forces one. On a recording of BEN's own solver calls for these boards (559 calls, 41,944
deals), ddfast returned the same results as `dds3` in about a quarter of the time.

ddfast needs endgame tables: 1.1 GB on disk, built once in about 3 minutes, and mapped
into memory shared between processes.

On Linux, PIMC also solves on ddfast in this step, through a `libdds.so` that ddfast
provides with DDS's C interface. On macOS that library is not built, so PIMC stays on DDS
until step 2.

### 2. C++ BGADLL

PIMC and PIMCDef run on a C++ port of BGADLL
([randalburns/bgadll-cpp](https://github.com/randalburns/bgadll-cpp)) with the same 61
`bga_*` functions, so `src/pimc/BGADLL_Native.py` loads it unchanged. It draws the same
samples in the same order as the C# library and calls ddfast directly. Differences from the
C# library: a persistent worker pool instead of new threads for every evaluation, and no
limit of 8 concurrent solves. `BEN_BGADLL=csharp` keeps the C# library.

### 3a. No GUI pauses, no sleep after PIMC

- `game.py` sleeps 0.1 s after every bid and 0.01 s around every card so a GUI client can
  redraw. `BEN_HEADLESS=1` skips these pauses. Unset, they remain, so `gameserver.py` and
  other front ends behave as before.
- `PIMC.py` and `PIMCDef.py` sleep 50 ms after `AwaitEvaluation`. The C++ BGADLL has
  already joined its workers when that call returns, so the sleep is skipped for it. The
  C# library keeps it.

On these boards the two together are about 28 s per 8 boards (100 bids, 416 cards, 192
PIMC calls).

### 3b. Sampling and claim-check speedups

- The dealing loops of `sample_cards_vec` and `shuffle_cards_bidding_info` run as numba
  kernels (`src/sample_jit.py`). The random numbers are drawn by the same numpy calls as
  before. numba is optional; without it the numpy code runs.
- `claimcheck` no longer builds the list of all card splits (up to 184,756 tuples) to
  sample 50 of them.
- `init_rollout_states_iterative` filters on first-trick hands and builds the full
  `(n, 13, 298)` state arrays once, for the samples that are kept.

### 4a. Network scores cached during play sampling

At every card, play sampling scored about 5,000 sampled hands against the whole auction with
the bidding network, and the leader's hand with the lead network. Each score depends only
on the hand and the board's auction, seat and vulnerability, and the same hands recur
through the board. The scores are now kept per board (`src/sample.py`, cleared in
`clear_sample_cache`, capped at 512 MB). Card-play predictions are kept for the board in the
same way.

This relies on the networks giving the same result for a row whatever the batch around it.
That was tested for batches of 2 to 12,000 rows; a batch of one row uses a different
TensorFlow kernel and is handled separately.

### 4b. Other repeated work removed

- During bidding, each sampled deal is solved double dummy once per bid, where different
  candidate bids roll out to the same contract (19% fewer solves on these boards).
- Bidding sampling scores each distinct hand once per auction, and checks first the player
  whose bids reject the most deals.
- The bidding network is loaded once when both sides use the same file.
- `dds_max_threads = 0` means the CPUs the process may run on, not all CPUs of the machine.

### 5a. Networks on ONNX Runtime

`config/onnxruntime.conf` runs every network on ONNX Runtime. The bidder and bid-info
networks are read from the 8744 `.keras` files and rebuilt in memory as ONNX graphs with
ONNX's LSTM operator (`src/nn/lstm_onnx.py`); each rebuilt graph is compared with the Keras
model when it loads, and the model stays on TensorFlow if they disagree. Nothing is written
to disk. The other networks use the files in `models/onnx/`.

Two observations about the existing ONNX setup, which this step works around rather than
changes:

- `models/onnx/bidder.onnx` and `bidinfo.onnx` are conversions of the GIB-BBO-8730 models
  from April 2025, not of the 8744 models that `default.conf` uses.
- tf2onnx converted the bidding LSTMs to `Loop` nodes that run one auction step per
  iteration, which is slow on a GPU.

The step's effect depends on the machine:

- **M5:** 32.8% faster. ONNX Runtime's matrix multiplies use Arm's KleidiAI kernels for the
  M5's SME2 unit.
- **PC:** 75.9% slower as BEN configures it. `src/nn/onnx_config.py` sets `intra_op_num_threads = 1`, so each
  network call uses one core, where TensorFlow uses all of them. Measured separately on the
  bidding network, ONNX Runtime with 12 threads is about 1.9 times faster than TensorFlow on
  this PC, and with 1 thread about 2.5 times slower:

  | Batch (hands × steps) | ONNX Runtime, 1 thread | ONNX Runtime, 12 threads | TensorFlow |
  |---|---|---|---|
  | 200 × 6 | 26.9 ms | 8.4 ms | 15.5 ms |
  | 5,000 × 4 | 519.5 ms | 110.2 ms | 205.4 ms |
  | 30,000 × 2 | 1,592.4 ms | 385.8 ms | 728.0 ms |

  `BEN_ONNX_THREADS=n` sets the number of threads; the default stays 1. On whole boards,
  12 threads (one per core) take 7.9 s/board, 22% less than the Keras configuration. With
  24 threads (every hardware thread) a board takes 11.0 s, more than with Keras, although
  the network calls themselves are still fast: the likely cause is ONNX Runtime's threads
  competing with ddfast's and PIMC's, which was not measured. The decisions are the same at
  every thread count.

### 5b. Networks on the NVIDIA GPU

- **TensorFlow:** the pip packages of TensorFlow 2.21 (`tensorflow[and-cuda]`) find every
  CUDA library through their RUNPATH except cuSOLVER, and without it TensorFlow registers
  no GPU. `src/nn/tf_guard.py` loads cuSOLVER first, and turns on memory growth so that
  several processes can share a GPU.
- **ONNX Runtime:** `config/gpu.conf` has the networks of step 5a; with `onnxruntime-gpu`
  installed they run on the GPU, with one CUDA memory arena shared by all sessions
  (`src/nn/onnx_config.py`).
- **Precision:** both runtimes use TF32 by default on this GPU, which rounds the operands
  of matrix multiplies to 10 bits of mantissa. It is turned off for both, so the networks
  compute in float32 as on the CPU. `BEN_TF32=1` turns it back on.

With the networks on the GPU, what remains of a board on the PC is mostly sampling code,
double-dummy solves and PIMC.

## Effect on decisions

Steps 1 to 4b and step 5b produce byte-identical PBN output to each other on each machine:
the same auctions and the same cards.

The exceptions:

- **Step 0 against the rest.** The contracts are the same. On the PC the results are the
  same too, and 14 spot cards on 5 boards are played in a different order within a suit. On
  the M5 the play differs on 5 of the 8 boards, and on one of them (3NT) declarer takes 10
  tricks instead of 11. On every M5 board the first card that differs is a different but
  equivalent card in the same suit (on the 3NT board, South's ♥5 or ♥2 at trick 1): `dds3`
  and ddfast break ties between equal cards differently in BEN's own solves, and today's
  code with `BEN_DDS=dds3` reproduces step 0's PBN byte for byte. On that one board the
  choice led the later play down a path that cost a trick. The PC's spot cards
  probably differ for the same reason; that was not checked there.
- **Step 5a on the PC, at every thread count.** Board 3 is bid to 3NT making 11 instead of 5♦ making 12. ONNX
  Runtime's CPU results differ from TensorFlow's by about 1e-6, and on that board one
  sampled hand falls on the other side of a threshold. On the M5, step 5a matches the
  earlier steps.

Each configuration gave the same output in both of its rounds.

## Running it

```bash
git clone --recurse-submodules https://github.com/randalburns/ben
bash build_native.sh        # builds ddfast and the C++ BGADLL, then runs preflight
cd src
BEN_HEADLESS=1 python game.py --boards <file>.pbn --auto True                                  # x86, CPU, Keras
BEN_HEADLESS=1 BEN_ONNX_THREADS=12 python game.py --boards <file>.pbn --auto True --config config/onnxruntime.conf  # x86, CPU, 12 cores
BEN_HEADLESS=1 python game.py --boards <file>.pbn --auto True --config config/onnxruntime.conf  # Apple silicon
BEN_HEADLESS=1 python game.py --boards <file>.pbn --auto True --config config/gpu.conf          # NVIDIA GPU
```

For the GPU: `pip install "tensorflow[and-cuda]==2.21.0"` and an `onnxruntime-gpu` built
for CUDA 12 (the PyPI package is built for CUDA 13; version 1.29.0 from Microsoft's
`onnxruntime-cuda-12` package feed was used here).

Without the submodules built, BEN falls back to `dds3` and the C# BGADLL, as before. On
macOS, the C# BGADLL (steps 0 and 1) only finds its DDS library with
`DYLD_LIBRARY_PATH=<ben>/bin/BGA/macos/arm64`.
`preflight.py` reports which solver and which PIMC engine are in use.

## Requirements and limits

- **ddfast** builds on Linux x86-64 and on macOS on Apple silicon. On x86 it uses AVX-512
  where the CPU has it and Google Highway otherwise.
- **The C++ BGADLL** builds on Linux x86-64 and macOS on Apple silicon.
- **Windows, Docker and the PyInstaller builds** are not covered: they run on `dds3` and the
  C# BGADLL as before.
- **`gpu.conf` and `onnxruntime.conf`** are copies of `default.conf` with other network
  settings, and have to be kept in step with it.
- **GPU memory:** about 3 GB per process with the shared arena, which is never returned.
  `BEN_ONNX_GPU_ARENA=large` lowers it for several processes on one GPU.
- **The numbers above** are from one set of 8 boards on two machines.

## Measured and not included

- A C++ port of the dealing code in `sample.py` was about 1 s/board faster than the numba
  version on a single core. It was left out to avoid a second implementation of that code.
- Stopping bid sampling early in long auctions would save about 11% per board, but can
  change a bid in a rare case.
- A default for ONNX Runtime's thread count other than 1. One thread per core was the best
  setting on this PC, but it was not measured on the M5, or with several BEN processes on
  one machine, where the threads of each process would compete.
