# BEN, one board at a time: from 31 s to 5 s

How the `maxperf` branch of this fork got BEN's time per board down from 31.2 s to
4.9 s, about 6.4× faster, without changing a single bid or card. The steps are listed
in the order they were applied.

## How it was measured

- **Boards:** the 8 boards of `src/christy170a.pbn`, played by `game.py --auto True --seed 42`.
- **Machine:** a laptop with a Ryzen AI 9 HX 370 (24 threads) and an RTX 5070 Laptop GPU, on AC power, doing nothing else.
- **Runs:** every step was run twice, in reverse order the second time; the times are the mean of both.
- **Correctness:** each run's output PBN (every auction and every card) was compared byte for byte.

Absolute times depend on the machine's state: another session measured the same endpoints at
20.7 and 3.5 s/board. The ratios between steps held in both sessions.

## The trajectory

| Step | What changed | s/board | This step | Cumulative |
|---|---|---|---|---|
| 0 | BEN as forked (`dds3`, C# PIMC on DDS) | 31.2 | | 1.0× |
| 1 | + ddfast | 29.1 | −6.6% | 1.07× |
| 2 | + C++ BGADLL | 28.4 | −2.5% | 1.10× |
| 3a | + no waits | 26.2 | −7.7% | 1.19× |
| 3b | + Python speedups | 26.2 | ~0 | 1.19× |
| 4a | + play-sampling cache | 19.4 | −25.8% | 1.61× |
| 4b | + other redundant work | 19.4 | ~0 | 1.61× |
| 5 | + GPU | **4.9** | −74.9% | **6.4×** |

Run-to-run noise was about ±1.5 s/board, so "~0" means too small to see end to end.

## The steps

**1. ddfast.** BEN's own double-dummy solves (bidding, leads, card play, claims) run on
[ddfast](https://github.com/randalburns/ddfast) instead of DDS 3.1, about 4.6× faster
on BEN's real calls. PIMC also solves on ddfast, through ddfast's `libdds.so`, a
drop-in for DDS's C interface.

**2. C++ BGADLL.** PIMC runs on a C++ port of BGADLL, which calls ddfast directly
([randalburns/bgadll-cpp](https://github.com/randalburns/bgadll-cpp)). PIMC's own time
drops by about a third; the board time drops less.

**3. Python (agent 1).**
- **3a. No waits.** `BEN_HEADLESS=1` skips the 0.1 s pause after every bid and the
  0.01 s pauses around every card, which give a GUI time to redraw. It also removes a
  50 ms sleep after each PIMC call that the C++ engine doesn't need.
- **3b. Speedups.** numba kernels for dealing sampled hands, cheaper claim checks, and
  the large play-state array built once instead of copied through every filter. Each
  is faster in isolation, but the total is too small to measure on a whole board.

**4. Redundant work (agent 4).**
- **4a. Play-sampling cache.** At every card, BEN re-scored about 5,000 sampled hands'
  whole auctions with the bidding network, and re-checked their opening leads. Those
  scores depend only on the hand and the board, so they're now computed once per hand
  per board and reused. This is the largest CPU-side step.
- **4b. The rest.** Each double-dummy deal solved once per bid, each distinct hand
  scored once per auction, the most selective player checked first, the bidding
  network loaded once. These help during bidding, but not visibly on these boards.

**5. GPU.** `config/gpu.conf` runs every network on ONNX Runtime on the GPU. The
bidding networks (LSTMs) are rebuilt in memory as graphs with ONNX's built-in LSTM
operator; the converted ONNX files stepped through each auction one move at a time.
All networks compute in full float32, so the GPU makes the same decisions as the CPU.

## Results stay the same

From step 1 on, every run's PBN is byte-identical, on CPU and GPU. The original fork
(step 0) plays the same contracts with the same results; 5 spot cards on 2 boards
differ, each a choice between small cards of equal double-dummy value.

## To run it

```bash
bash build_native.sh                 # ddfast and the C++ BGADLL (submodules)
pip install "tensorflow[and-cuda]==2.21.0"   # and onnxruntime-gpu, CUDA 12 build, for the GPU
cd src && BEN_HEADLESS=1 python game.py --config config/gpu.conf --boards <file>.pbn --auto True
```

Without a GPU, leave out `--config config/gpu.conf`: the same steps up to 4b apply on the CPU.

## Not included

- **The C++ port of BEN's sampling code (agent 2):** it saved about 1 s/board over the
  Python speedups, which didn't justify a second implementation to maintain.
- **Trade-offs that change decisions:** stopping bid sampling early in long auctions
  (about 11% per board) and lower-precision GPU arithmetic.
