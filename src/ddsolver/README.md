# Double-dummy solver (`ddsolver`)

BEN's double-dummy solver wrapper, `DDSolver`, runs on one of two backends:

| Backend | What it is | Platforms | Speed on BEN's workload |
|---------|------------|-----------|-------------------------|
| **ddfast** | An exact double-dummy solver with DDS 3.x's `dds3` Python interface, built from the `extern/ddfast` submodule ([randalburns/ddfast](https://github.com/randalburns/ddfast), branch `dds-capi`) | Linux x86-64 with AVX-512 and BMI2 (Zen 4/5, Intel Ice Lake and later); Python 3.12+ | about 4× dds3 |
| **dds3** | DDS 3.1's Python extension, vendored per platform under `bin/dds3-*` (built from the DDS **v3.1.0** tag, `fe6a6581`) | Windows, Linux, macOS; Python 3.12 only | baseline |

Both return the same results: a 559-call recording of a real 8-board session
(41,944 deals plus par) replays identically on both (see
[Benchmarking](#benchmarking-record-a-real-session-replay-it)), taking 6.4 s on
ddfast and 29.7 s on dds3 (24 threads). By purpose: card play 6×, bidding 3.8×,
opening leads 3.7×, claim checks 10×, par 2×.

`DDSolver`'s public API (`solve`, `calculatepar`, `version`,
`expected_tricks_dds`, ...) is the same on both, so callers do not care which
one runs.

## Choosing the backend

```bash
BEN_DDS=ddfast   # ddfast; fails at import if it is not built
BEN_DDS=dds3     # the vendored dds3
                 # unset: ddfast if it is built, else dds3
```

`DDSolver(...).version()` says which one is in use (`ddfast` or `dds3 3.1.0`),
and so do `preflight.py` (`double dummy solver: ...`) and the header of a
`--ddsrecord` recording.

PIMC (BGADLL) picks its engine and solver separately, with `BEN_BGADLL` and
`BEN_PIMC_DDS`: see [PIMC engines](#pimc-engines-the-c-port-or-the-c-bgadll).

The Docker images and the Windows PyInstaller builds only ship dds3 (they do not
build the submodule), so they run on dds3.

## Using ddfast

### Build it

```bash
git submodule update --init --recursive extern/ddfast   # ddfast and its extern/dds
pip install nanobind                                     # in BEN's Python environment
sh extern/ddfast/python/build.sh                         # uses $PYTHON, default python3
```

`build.sh` compiles for the build machine's CPU (`-march=native`) with clang++
(or g++) and writes three libraries to `extern/ddfast/python/ddfast/`:

| File | What |
|------|------|
| `libddfast.so` | the solver, the endgame tables, the thread pool and the DDS-compatible layer |
| `_ddfast.abi3.so` | the Python module (CPython's stable ABI, so any Python 3.12+), linked against `libddfast.so` |
| `libdds.so` | DDS's C interface (`SolveBoardPBN`, ...) on `libddfast.so`, for PIMC |

`ddsolver.py` finds the package in the submodule (or an installed `ddfast`);
nothing needs to be added to `PYTHONPATH`.

### The endgame tables

ddfast needs endgame tables: about 1.1 GB on disk and 1.2 GB in memory. They are
built the first time ddfast runs, which takes about 3 minutes on all cores and
briefly uses up to about 5 GB of memory; after that they load at startup.
`DDSolver` loads them once per process from `$DDFAST_TABLES`, else
`extern/ddfast/build/tables` (gitignored in the submodule). To build them ahead
of time:

```bash
python -c "import sys; sys.path.insert(0, 'extern/ddfast/python'); import ddfast; ddfast.init('extern/ddfast/build/tables')"
```

### How DDSolver drives it

A batch of sampled deals is one `ddfast.solve_boards_pbn` call: ddfast solves the
deals on its own threads (`max_threads`, `[dds] dds_max_threads`, 0 = all CPUs)
with the GIL released once, and keeps its search state per thread itself, so no
Python thread pool or `SolverContext` is involved. Under gevent (`gameapi.py`)
the call runs on gevent's native-thread pool so the hub keeps serving other
requests.

Differences from dds3 that BEN does not see: `nodes` is always 0; with
`solutions=1` the reported card may be another equally good one, and cards with
equal scores may come in another order (BEN keys results by card and uses only
the scores). One difference it does see: ddfast rejects a doubled deal prefix
(`"N:N:..."`, which dds3 accepts), so `calculatepar` only adds `N:` when the deal
has none.

## dds3

Since the DDS **3.0.0** upgrade, the dds3 backend uses the `dds3` Python
extension instead of the old `ctypes`-loaded `dds.dll` / `libdds.so`.

| Before (DDS 2.9.x)                         | Now (DDS 3.0.0)                             |
|--------------------------------------------|---------------------------------------------|
| `ctypes.cdll.LoadLibrary("dds.dll")`       | `import dds3` (compiled pybind11 extension)  |
| `dds.py`, `ddss.py` ctypes wrappers        | removed                                      |
| `DDSSolver` (ddss fast-fork) + fallback    | removed — single `DDSolver` class            |
| `bin/ddss.dll` / `.exp` / `.lib`           | removed (ddss fast-fork dropped)              |

> **Note:** `bin/dds.dll`, `bin/libdds.so` and `bin/darwin/libdds.2.9.0.dylib`
> are **kept** — they are not used by `ddsolver`, but the C# BGADLL (PIMC)
> calls the DDS C ABI and `src/pimc/BGADLL_Native.py` may preload them, and the
> C++ port falls back to them without ddfast. See
> [The DDS C library used by the C# BGADLL](#the-dds-c-library-used-by-the-c-bgadll).

**Concurrency.** On dds3, `DDSolver` runs a Python `ThreadPoolExecutor` and issues one
`dds3.solve_board_pbn(..., context=ctx)` per board, keeping **one `SolverContext`
per worker thread** — the modern DDS 3.x model (internal batch threading was
removed). `solve_board_pbn` releases the GIL during the search, so the pool runs
truly in parallel and each thread's context keeps a warm transposition table.
The deprecated `set_max_threads` / `solve_all_boards` bindings are **not** used
(and no longer exist in current `dds3`).

## Installing the `dds3` extension

`dds3` is a compiled extension and must be built per platform from the DDS
repository (https://github.com/dds-bridge/dds). It is **not** on PyPI.

### Build dds3

```bash
# The DDS repo uses bazelisk (the `bazel` launcher); `bazel` may not be on PATH.
bazelisk build -c opt //python:_dds3            # -> bazel-bin/python/_dds3.{so,pyd}
bazelisk build -c opt //python:dds3_wheel_dist  # -> a wheel under bazel-bin/python/dist/
```

> **Match the Python version.** A compiled extension is locked to one CPython
> version (it links e.g. `python312.dll`). BEN's `ben` conda env is **Python
> 3.12**, so build the extension for 3.12, not the DDS repo's default 3.14:
> ```bash
> bazelisk build -c opt \
>   --@rules_python//python/config_settings:python_version=3.12 //python:_dds3
> ```
> This requires the 3.12 toolchain to be registered in the DDS repo's
> `MODULE.bazel` (the default only registers 3.14). Verify the result with
> `grep -a python3 _dds3.pyd` — it must reference `python312.dll`. A 3.14-built
> extension fails to import under 3.12 with a confusing `ImportError`.

See `docs/python_interface.md` in the DDS repo for per-OS prerequisites
(Windows: MSVC; Linux: GCC/clang; macOS: bazelisk).

> **Linux: build against an old glibc for portability.** A `.so` records the
> highest glibc symbol version it uses and will not load on an older glibc
> (`version 'GLIBC_2.XX' not found`). Build the Linux extension on the **oldest**
> glibc you need to support — a [manylinux](https://github.com/pypa/manylinux)
> container (`manylinux_2_28` = glibc 2.28, ships CPython 3.12) is ideal: it runs
> on every current distro **and** on the Ubuntu 24.04 Docker image. Building on
> Ubuntu 24.04 (glibc 2.39) instead yields a `.so` that needs glibc ≥ 2.38, so it
> will **not** run on Ubuntu 22.04 / Debian 12. The vendored `bin/dds3-linux/`
> binary is a manylinux_2_28 build (needs glibc ≥ 2.25), made without Bazel the
> same way as the PIMC library below, plus the two binding files and pybind11:
> ```bash
> docker run --rm -v "$PWD:/dds:ro" -v "$PWD/out:/out" quay.io/pypa/manylinux_2_28_x86_64 bash -c '
>   PY=/opt/python/cp312-cp312/bin/python; $PY -m pip install pybind11
>   mkdir -p /inc && ln -s /dds/library/src /inc/dds
>   g++ -std=c++20 -O3 -fPIC -shared -pthread -static-libstdc++ -static-libgcc -DNDEBUG \n>       -I/inc -I/dds/library/src -I/dds/library/src/api -I/dds/python/src $($PY -m pybind11 --includes) \n>       /dds/python/src/bindings.cpp /dds/python/src/converters.cpp \n>       $(find /dds/library/src -name "*.cpp" | grep -viE "test|bench|_main|fuzz") -o /out/_dds3.so
>   strip --strip-unneeded /out/_dds3.so'
> ```

> **macOS: always set a low deployment target.** Without it the extension is
> stamped with the build machine's macOS version and links libc++ symbols that
> older systems lack — the `_dds3.so` shipped before 0.8.8.8 had `minos 26.5` and
> failed on anything older with
> `Symbol not found: __ZNSt3__113__hash_memoryEPKvm`. Bazel is not needed; the
> extension is two binding files plus the library sources:
> ```bash
> # in a DDS checkout, with a Python 3.12 venv that has pybind11 installed
> mkdir -p inc && ln -sfn "$PWD/library/src" inc/dds     # for #include <dds/...>
> SDK=$(xcrun --show-sdk-path)      # see the note below about CLT 27
> MACOSX_DEPLOYMENT_TARGET=11.0 clang++ -isysroot "$SDK" -std=c++20 -O3 -flto=thin \
>   -fPIC -shared -undefined dynamic_lookup -mmacosx-version-min=11.0 -arch arm64 \
>   -Iinc -Ilibrary/src -Ilibrary/src/api -Ipython/src $(python -m pybind11 --includes) \
>   python/src/bindings.cpp python/src/converters.cpp \
>   $(find library/src -name '*.cpp' | grep -viE 'test|bench|_main|fuzz') \
>   -o _dds3.so
> otool -l _dds3.so | grep -A4 LC_BUILD_VERSION   # must say minos 11.0
> ```
> The macOS 27 Command Line Tools ship without libc++ headers (`'string' file not
> found`); point at the 26.5 SDK instead:
> `SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` and add
> `-nostdinc++ -isystem "$SDK/usr/include/c++/v1"`.
>
> Build all platforms from the same DDS commit and vendor that commit's
> `python/dds3/__init__.py` with them: v3.1.0 renamed `initialise_static_memory`
> to `initialize_static_memory`, so an `__init__.py` from one side of the rename
> does not import an extension from the other.

> **Windows without Bazel.** From an x64 Native Tools prompt, in a DDS checkout
> with `library\src` also copied (or junctioned) to `inc\dds`, and a Python 3.12
> that has pybind11 installed:
> ```bat
> cl /nologo /std:c++20 /O2 /EHsc /MT /permissive- /bigobj /DNDEBUG /DNOMINMAX ^
>    /Iinc /Ilibrary\src /Ilibrary\src\api /Ipython\src ^
>    /I<python>\Include /I<python>\Lib\site-packages\pybind11\include /MP ^
>    python\src\bindings.cpp python\src\converters.cpp <all library\src\**\*.cpp except tests> ^
>    /LD /Fe_dds3.pyd /link /LIBPATH:<python>\libs
> ```
> `/DNOMINMAX` is needed because `windows.h` otherwise defines `min`/`max` macros
> that break `std::min` in the bindings.

### Make it importable — two options

When dds3 is the backend, `ddsolver.py` imports it and finds it either way:

1. **pip install (recommended)** — install the built wheel into BEN's Python
   environment:
   ```bash
   pip install dds3-1.0.0-*.whl
   ```

2. **Vendored in `bin/`** — copy the built `dds3` package into the
   platform-specific folder so it is shipped with BEN:
   ```
   bin/dds3-win/dds3/__init__.py     + _dds3.pyd     (Windows)
   bin/dds3-linux/dds3/__init__.py   + _dds3.so      (Linux)
   bin/dds3-darwin/dds3/__init__.py  + _dds3.so      (macOS)
   ```
   `ddsolver._load_dds3()` adds the right folder to `sys.path` automatically.

The Docker images expect option 2 (`bin/dds3-linux/`). PyInstaller builds
expect option 1 (`dds3` installed in the build environment; the `.spec` files
list it under `hiddenimports`).

## PIMC engines: the C++ port or the C# BGADLL

PIMC and PIMCDef come from BGADLL ([ThorvaldAagaard/BGA](https://github.com/ThorvaldAagaard/BGA),
MIT). BEN can load either of two builds with the same C interface (the 61 `bga_*`
functions `src/pimc/BGADLL_Native.py` binds):

| Engine | What | Solver |
|--------|------|--------|
| **cpp** (the default when built) | A C++ port, `extern/bgadll-cpp` ([randalburns/bgadll-cpp](https://github.com/randalburns/bgadll-cpp)); Linux and macOS | ddfast (`ddfast::card_results`, in the `libddfast.so` BEN already loaded, so one copy of the tables), else the vendored `libdds` |
| **csharp** | BGADLL.Native (NativeAOT C#), `bin/BGA/<os>/<arch>/BGADLL.*`; every platform | a `libdds` it loads by name: see the next section |

```bash
BEN_BGADLL=cpp      # the C++ port; fails if it is not built
BEN_BGADLL=csharp   # the C# library
                    # unset: the C++ port if it is built, else the C# library
BEN_PIMC_DDS=haglund   # either engine: solve on the vendored DDS library instead of ddfast
```

Build the port:

```bash
git submodule update --init extern/bgadll-cpp   # randalburns/bgadll-cpp (not --recursive: its
                                                #  extern/BGA is only for its own correctness check)
sh extern/bgadll-cpp/build.sh   # -> extern/bgadll-cpp/bin/BGADLL.so (.dylib on macOS; C++17)
```

`preflight.py` shows which engine and solver PIMC got, e.g.
`PIMC engine: BGADLL C++ port - DDS backend: ddfast`. If ddfast is not built the port
falls back to the vendored `libdds` by itself (BEN points it there with
`BGA_DDS_LIB`); the port's own settings (`BGA_DDS_BACKEND`, `BGA_DDFAST_LIB`,
`BGA_DDS_LIB`) are described in its README.

The port makes the same samples in the same order as the C# library (same seed,
same shuffle, same constraints and weights), so it returns the same results:
byte-identical on 303 positions in its own tests, and in BEN the same output on
`src/test_pimc_native.py`'s position and the same contracts and tricks on 8 boards.
What differs is the machinery: one persistent worker pool instead of new threads
per call (ddfast keeps its search state per OS thread), no 8-call limit on
concurrent solves, ddfast's `card_results` (every card in one call) instead of
`SolveBoardPBN`, and a condition variable instead of 10 ms polling.

On 8 boards (`christy170a.pbn`, seed 42), run back to back on AC power, both on
ddfast, TensorFlow on the CPU, results identical:

| | C# BGADLL (ddfast's `libdds.so`) | C++ port |
|---|---|---|
| PIMC, 192 calls | 20.8 s (108 ms per call) | 13.1 s (68 ms per call), 1.6× faster |
| Whole run | 174.6 s | 164.7 s |

(On battery, with the CPU power-limited: 148 against 76 ms per call, 1.9×.)

## The DDS C library used by the C# BGADLL

The C# BGADLL (PIMC / PIMCDef) does not use `DDSolver`'s backend. On every platform
except Windows x64 (which uses `libbcalcdds.dll`) it calls the DDS **C ABI**
(`SolveBoardPBN`, `SetMaxThreads`, `ErrorMessage`) from a library it loads by
name (`[DllImport("dds")]`): `libdds.so` on Linux. `BGADLL_Native.py` preloads
that library before BGADLL, and BGADLL binds to the one already loaded.

### ddfast's libdds.so (Linux, the default when built)

With the C# engine (`BEN_BGADLL=csharp`, or the C++ port not built), when
`extern/ddfast/python/ddfast/libdds.so` is built (see
[Using ddfast](#using-ddfast)), BEN preloads it in place of the vendored file, so
PIMC solves on ddfast too:

```bash
BEN_PIMC_DDS=haglund   # keep the vendored DDS library below instead
                       # unset: ddfast's libdds.so if it is built
```

This is independent of `BEN_DDS`: with `BEN_DDS=dds3`, PIMC still uses ddfast's
library unless `BEN_PIMC_DDS=haglund`. `preflight.py` shows which one PIMC got:
`PIMC DDS backend: haglund on ddfast` or `... on <path of the vendored libdds>`
("haglund" is BGADLL's name for its DDS backend, whichever library provides it).

`libdds.so` shares `libddfast.so` with ddfast's Python module, so one process
holds one copy of the tables. `ddsolver.py` sets `DDFAST_TABLES` for the process
so both find the same tables whichever solves first. The library accepts any
thread index: ddfast keeps its search state per OS thread, so BGADLL's indexes
0–7 need no `SetMaxThreads`.

It agrees with the vendored DDS 3.1.0 library on 36,000 calls
(`extern/ddfast/python/tests/test_dds_capi.py`: random positions at every stage
of play, solutions 1–3, targets, modes, single-threaded and from 8 threads), and
PIMC's output on `src/test_pimc_native.py`'s position is identical on both. On
an 8-board session PIMC's time fell 28% (30.5 s to 22.0 s): each PIMC call also
spends time in BGADLL's own sampling and bookkeeping, which ddfast does not
change.

macOS and Windows have no ddfast build, so there PIMC always uses the vendored
library.

**Measuring PIMC's solves.** `DDFAST_CAPI_STATS=<file>` makes ddfast's
`libdds.so` append a summary to `<file>` when BEN exits:

```bash
DDFAST_CAPI_STATS=/tmp/pimc-dds.txt python game.py --boards christy170a.pbn --auto True
```

It reports calls and errors, time in the solver (in all and by tricks left),
the OS threads that called and each thread's first call against later ones
(ddfast sets up its search state per thread on the first call), and how many
solves overlapped (busy wall time, mean and peak concurrency). Comparing the
busy time with `pimc_declaring` + `pimc_defending` in BEN's timing summary
shows how much of PIMC is solving. On the 8 boards above: 37,235 solves, the
solver busy 5.6 s of PIMC's 21 s; BGADLL starts about 11 new threads per call
(2,127 in all), each paying 6.8 ms on its first solve against 0.58 ms later,
and never runs more than 8 solves at once (its thread-index pool). Unset, it
costs one branch per call.

### The vendored libraries

| Platform       | File                                   | DDS   |
|----------------|----------------------------------------|-------|
| Windows x64    | `bin/BGA/windows/x64/dds.dll` (fallback only) | 2.9.0 |
| Windows ARM64  | `bin/BGA/windows/arm64/dds.dll`        | 3.1.0 (`671a4aa4`) |
| Linux x64      | `bin/BGA/linux/x64/libdds.so` (+ identical `dds.dll`, `bin/libdds.so`) | 3.1.0 (`671a4aa4`) |
| Linux ARM64    | `bin/BGA/linux/arm64/libdds.so`        | 3.1.0 (`671a4aa4`) |
| macOS ARM64    | `bin/BGA/macos/arm64/libdds.2.9.0.dylib` | 2.9.0 |

> **The thread index must be accepted.** BGADLL spreads its solves over thread
> indexes 0–7. DDS builds from before commit `15eb56fd` (2026-05-22) only accept
> index 0 and return `-15` (`RETURN_THREAD_INDEX`) for the rest, which BGADLL
> scored as 0 tricks: from 0.8.8.0 to 0.8.8.8 PIMC on Linux x64 decided on about
> 1 in 8 of its samples. BGADLL 0.9.9.2+ prints a `BGADLL DDS warning` line on
> stderr the first time a solve fails, so check the log after changing any of
> these files.
>
> **Check the architecture.** Before 0.8.8.8 the Linux ARM64 and Windows ARM64
> folders held x86-64 copies, which cannot load next to an ARM64 BGADLL.

Build the Linux libraries in a manylinux container (glibc 2.28, static C++
runtime, so only libc/libm/libpthread are needed):

```bash
# from a DDS checkout; use quay.io/pypa/manylinux_2_28_aarch64 for ARM64
docker run --rm -v "$PWD:/dds:ro" -v "$PWD/out:/out" quay.io/pypa/manylinux_2_28_x86_64 bash -c '
  mkdir -p /inc && ln -s /dds/library/src /inc/dds
  g++ -std=c++20 -O3 -fPIC -shared -pthread -static-libstdc++ -static-libgcc \
      -Wl,-soname,libdds.so -I/inc -I/dds/library/src -I/dds/library/src/api \
      -o /out/libdds.so $(find /dds/library/src -name "*.cpp" | grep -viE "test|bench|_main|fuzz")
  strip --strip-unneeded /out/libdds.so'
```

The Windows ARM64 `dds.dll` is the same sources compiled with MSVC from an
`x64_arm64` developer prompt: `cl /std:c++20 /O2 /EHsc /MT /LD ... /link /MACHINE:ARM64`
(`/MT` so it needs no VC++ redistributable).

## Benchmarking: record a real session, replay it

`ddsrecorder.py` + `ddsreplay.py` turn a played session into a repeatable DDS
performance test. Every DDS call BEN makes is written to a JSON Lines file with
its **full input and output**; the replayer re-issues those calls, times them,
and checks the answers still match.

Benchmarking this way rather than on generated deals matters because DDS runtime
is dominated by *which* calls BEN makes — the trick-1 `solutions=1` batches for
bidding cost far more than the trick-11 ones for card play, and their mix is not
easy to guess. In a sample 16-deal session, trick 1 was **81 % of all DDS time**.

### Record

Recording is off unless you ask for it with `--ddsrecord`, accepted by
`game.py`, `gameserver.py`, `gameapi.py` and `table_manager_client.py`. The
value is a file or a directory (a directory gets one file per process, so
several BEN processes can record at once); a `.gz` suffix compresses.

```bash
python game.py --boards mysession.pbn --auto True --ddsrecord d:/tmp/dds-rec.jsonl
```

Where there is no command line to reach — Docker, frozen builds — the
`BEN_DDS_RECORD` environment variable does the same thing. `--ddsrecord` wins if
both are given.

The entry points arm the recorder right after parsing arguments, before any
`DDSolver` is constructed, so no call is missed. The header record is written
lazily with the first DDS call, which is how it can carry the DDS version, mode
and thread count that only become known later.

Calls are tagged with the board being played in `game.py` (so also
`gameserver.py`) and `table_manager_client.py`. `gameapi.py` is **not** tagged —
it serves requests concurrently under gevent, so a single "current board" would
mis-attribute interleaved requests. Its records still carry the complete input
and replay fine, just with `board: null`.

Records are flushed per line, so a session ended with Ctrl-C still yields a
usable file; the replayer skips a truncated final line.

Expect roughly 4 MB per 800 calls (200-sample batches dominate). Recording adds
the cost of serialising those batches, so **the recorded `ms` are not a clean
benchmark** — they are there for reference and appear in the report as
`rec ms/call`. The replay timings are the ones to compare.

### Replay

```bash
# straight benchmark, best of 3
python src/ddsolver/ddsreplay.py d:/tmp/dds-rec/dds-12345.jsonl --repeat 3

# what does the recording contain? (does not run DDS)
python src/ddsolver/ddsreplay.py rec.jsonl --list

# tune dds_max_threads for this machine
python src/ddsolver/ddsreplay.py rec.jsonl --threads 8 --threads 16 --threads 32

# is a DDS rebuild / a dds_mode change faster, and still correct?
python src/ddsolver/ddsreplay.py rec.jsonl --json new.json --baseline old.json
python src/ddsolver/ddsreplay.py rec.jsonl --dds-mode 2
```

The report breaks time down by `purpose` (`bid`, `lead`, `play`, `claimcheck`,
`claimtricks`, `contract`, `par`) and, with `--tricks`, by trick number. Other
useful filters: `--purpose`, `--board`, `--solutions`, `--min-trick`,
`--max-trick`, `--limit`, `--no-par`.

`--json` writes a machine-readable result that a later run can `--baseline`
against; the exit status is non-zero if any call returned something other than
what was recorded, so it can be used as a CI check that a DDS change is
behaviour-preserving.

Replay calls `solve_helper` / `_calculatepar_impl` rather than the public
`solve` / `calculatepar`, so it measures the solver itself and never re-triggers
the recorder.

**Comparing the backends:** replay the same recording under each:

```bash
BEN_DDS=dds3   python src/ddsolver/ddsreplay.py rec.jsonl --warmup --repeat 3
BEN_DDS=ddfast python src/ddsolver/ddsreplay.py rec.jsonl --warmup --repeat 3
```

The report's `replaying:` line names the backend. Both should report
`VERIFY: all ... calls returned the recorded result`, whichever backend made the
recording.

**On repeatability:** with `dds_mode=1` dds3 reuses a transposition table across
consecutive solves on the same `SolverContext`, and the thread pool assigns
boards to contexts nondeterministically (ddfast likewise keeps a table per
thread), so a single run carries a few percent of noise. Use `--repeat` (the
best run is reported) and `--warmup`.
