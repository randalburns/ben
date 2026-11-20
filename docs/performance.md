# BEN, one board at a time: from 31 s to 5 s

How the `maxperf` branch of this fork got BEN's time per board down from 31.2 s to
4.9 s, about 6.4× faster, without changing a single bid or card. On an Apple M5, which
has no NVIDIA GPU, the same steps take it from 20.3 s to 6.1 s, about 3.3× faster. The
steps are listed in the order they were applied.

## How it was measured

- **Boards:** the 8 boards of `src/christy170a.pbn`, played by `game.py --auto True --seed 42`.
- **Machines:**
  - **PC:** a laptop with a Ryzen AI 9 HX 370 (24 threads) and an RTX 5070 Laptop GPU.
  - **M5:** a MacBook with an Apple M5 (10 cores: 4 performance, 6 efficiency) and 16 GB, macOS 26.
  - Both on AC power, doing nothing else.
- **Runs:** every step was run twice, in reverse order the second time; the times are the mean of both.
  On the M5, each time is the whole `game.py` run divided by 8, start-up included.
- **Correctness:** each run's output PBN (every auction and every card) was compared byte for byte.

Absolute times depend on the machine's state: another session on the PC measured the same
endpoints at 20.7 and 3.5 s/board. The ratios between steps held in both sessions.

## The trajectory

| Step | What changed | PC s/board | PC step | PC total | M5 s/board | M5 step | M5 total |
|---|---|---|---|---|---|---|---|
| 0 | BEN as forked (`dds3`, C# PIMC on DDS) | 31.2 | | 1.0× | 20.3 | | 1.0× |
| 1 | + ddfast | 29.1 | −6.6% | 1.07× | 19.0 | −6.2% | 1.07× |
| 2 | + C++ BGADLL | 28.4 | −2.5% | 1.10× | 16.6 | −12.5% | 1.22× |
| 3a | + no waits | 26.2 | −7.7% | 1.19× | 13.3 | −19.8% | 1.52× |
| 3b | + Python speedups | 26.2 | ~0 | 1.19× | 12.4 | −7.0% | 1.63× |
| 4a | + play-sampling cache | 19.4 | −25.8% | 1.61× | 9.9 | −19.9% | 2.04× |
| 4b | + other redundant work | 19.4 | ~0 | 1.61× | 9.0 | −9.1% | 2.24× |
| 5a | + ONNX Runtime | not measured | | | **6.1** | −32.8% | **3.3×** |
| 5b | + cuSOLVER (networks on the GPU) | **4.9** | −74.9% (5a and 5b) | **6.4×** | no NVIDIA GPU | | |

Run-to-run noise was about ±1.5 s/board on the PC, so "~0" means too small to see end to
end. On the M5 the two rounds of a step differed by at most 0.8 s/board. The PC's 5a and 5b
were measured together, as one step from 4b.

## The steps

**1. ddfast.** BEN's own double-dummy solves (bidding, leads, card play, claims) run on
[ddfast](https://github.com/randalburns/ddfast) instead of DDS 3.1, about 4.6× faster
on BEN's real calls. On Linux, PIMC also solves on ddfast, through ddfast's `libdds.so`,
a drop-in for DDS's C interface. On macOS that library isn't built, so in step 1 PIMC
still solves on DDS.

**2. C++ BGADLL.** PIMC runs on a C++ port of BGADLL, which calls ddfast directly
([randalburns/bgadll-cpp](https://github.com/randalburns/bgadll-cpp)). PIMC's own time
drops by about a third; the board time drops less. On the M5 this step is larger,
because it is also where PIMC moves from DDS to ddfast.

**3. Python (agent 1).**
- **3a. No waits.** `BEN_HEADLESS=1` skips the 0.1 s pause after every bid and the
  0.01 s pauses around every card, which give a GUI time to redraw. It also removes a
  50 ms sleep after each PIMC call that the C++ engine doesn't need.
- **3b. Speedups.** numba kernels for dealing sampled hands, cheaper claim checks, and
  the large play-state array built once instead of copied through every filter. Each
  is faster in isolation; on the PC the total is too small to measure on a whole board.

**4. Redundant work (agent 4).**
- **4a. Play-sampling cache.** At every card, BEN re-scored about 5,000 sampled hands'
  whole auctions with the bidding network, and re-checked their opening leads. Those
  scores depend only on the hand and the board, so they're now computed once per hand
  per board and reused. This is the largest CPU-side step.
- **4b. The rest.** Each double-dummy deal solved once per bid, each distinct hand
  scored once per auction, the most selective player checked first, the bidding
  network loaded once. These help during bidding, but not visibly on the PC's boards.

**5. The networks.** Up to step 4b, the networks run on TensorFlow (Keras) on the CPU.
BEN's TensorFlow would use a GPU by itself if it found one (BEN's training scripts always
have), but on the PC TensorFlow 2.21 found none until step 5b.
- **5a. ONNX Runtime.** Every network runs on ONNX Runtime instead of Keras
  (`config/onnxruntime.conf` on the CPU; `config/gpu.conf` has the same networks). The
  bidding networks (LSTMs) are rebuilt in memory from the Keras models as graphs with
  ONNX's built-in LSTM operator, and checked against Keras when they load; the converted
  ONNX files stepped through each auction one move at a time. ONNX Runtime has much less
  overhead per call than Keras, and on the M5 its matrix multiplies use Arm's KleidiAI
  kernels for the M5's SME2 matrix unit, 3 to 8 times faster than without them on BEN's
  networks.
- **5b. cuSOLVER and the GPU.** TensorFlow 2.21's pip packages list every CUDA library
  they need except cuSOLVER, so without it TensorFlow registers no GPU. BEN now loads
  cuSOLVER itself (`src/nn/tf_guard.py`). With `onnxruntime-gpu` installed,
  `config/gpu.conf` runs every network on the GPU. All networks compute in full float32
  (TF32 off), so the GPU makes the same decisions as the CPU. The M5 has no NVIDIA GPU,
  so this step doesn't apply there.

## Results stay the same

From step 1 on, every run's PBN is byte-identical, on CPU and GPU, on both machines.
The original fork (step 0) plays the same contracts:
- **PC:** the same results; 5 spot cards on 2 boards differ, each a choice between small
  cards of equal double-dummy value.
- **M5:** the play differs on 5 of the 8 boards, and on one of them (3NT) declarer takes
  10 tricks instead of 11.

## To run it

```bash
bash build_native.sh                 # ddfast and the C++ BGADLL (submodules)
pip install "tensorflow[and-cuda]==2.21.0"   # and onnxruntime-gpu, CUDA 12 build, for the GPU
cd src && BEN_HEADLESS=1 python game.py --config config/gpu.conf --boards <file>.pbn --auto True
```

Without a GPU, use `--config config/onnxruntime.conf` instead (steps up to 5a), or leave
out `--config` for the Keras networks (steps up to 4b).

On macOS, the original C# BGADLL (steps 0 and 1) only finds its DDS library with
`DYLD_LIBRARY_PATH=<ben>/bin/BGA/macos/arm64`.

## Not included

- **The C++ port of BEN's sampling code (agent 2):** it saved about 1 s/board over the
  Python speedups, which didn't justify a second implementation to maintain.
- **Trade-offs that change decisions:** stopping bid sampling early in long auctions
  (about 11% per board) and lower-precision GPU arithmetic.
