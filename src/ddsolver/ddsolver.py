"""BEN's double-dummy solver wrapper, on one of two backends:

  ddfast  an exact double-dummy solver with DDS 3.x's dds3 interface, built from the
          extern/ddfast submodule (Linux x86-64 with AVX-512; Python 3.12+).
  dds3    DDS 3.1's Python extension, vendored per platform under bin/dds3-*.

BEN_DDS=ddfast or BEN_DDS=dds3 picks one; unset, ddfast is used when it is built and dds3
otherwise. Both return the same results (see ddsreplay.py); DDSolver.version() names the one
in use. PIMC (BGADLL) chooses its own DDS library separately - see ddfast_libdds() and
pimc/BGADLL_Native.py.
"""
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from collections import Counter
from objects import Card
from colorama import Fore, Back, Style, init
from nn.timing import ModelTimer
from ddsolver.ddsrecorder import DDSRecorder

init()


# ---------------------------------------------------------------------------------------------
# Finding the backends


def _ddfast_python_dirs():
    """Directories that may hold the ddfast Python package (python/ddfast/ in the
    ddfast repository, vendored here as the extern/ddfast submodule)."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [
        os.path.abspath(os.path.join(here, "..", "..")),  # repo: src/ddsolver -> <repo>
        os.path.abspath(os.path.join(here, "..")),        # flattened (Docker): /app/ddsolver -> /app
    ]
    if os.getenv("BEN_HOME"):
        roots.append(os.getenv("BEN_HOME"))
    roots.append(os.getcwd())
    dirs = []
    for root in roots:
        for cand in (os.path.join(root, "extern", "ddfast", "python"),
                     os.path.join(root, "ddfast", "python")):
            if cand not in dirs:
                dirs.append(cand)
    return dirs


def _load_ddfast():
    """Import ddfast: installed, else from the extern/ddfast submodule, whose
    python/build.sh builds the extension into python/ddfast/."""
    last_error = None
    try:
        import ddfast
        return ddfast
    except ImportError as ex:
        last_error = ex

    for cand in _ddfast_python_dirs():
        if not os.path.isdir(os.path.join(cand, "ddfast")):
            continue
        if cand not in sys.path:
            sys.path.insert(0, cand)
        try:
            import ddfast
            return ddfast
        except ImportError as ex:
            last_error = ex

    raise ImportError(
        "Could not import the 'ddfast' double-dummy solver.\n"
        "Build its Python extension from the extern/ddfast submodule:\n"
        "  git submodule update --init --recursive extern/ddfast\n"
        "  pip install nanobind\n"
        "  sh extern/ddfast/python/build.sh\n"
        "The extension uses CPython's stable ABI (3.12 and later) and needs AVX-512 on x86-64.\n"
        f"\nUnderlying import error: {last_error!r}"
    ) from last_error


def _load_dds3():
    """Import the dds3 extension (DDS 3.x Python interface).

    Tries a normal import first (dds3 installed as a wheel). If that fails,
    falls back to a copy vendored under BEN's bin/ directory.
    """
    # Keep the real failure around. A vendored _dds3.so that exists but won't
    # load (e.g. built for a different Python version, or missing a dependent
    # shared library) raises ImportError here; swallowing it hides the cause.
    last_error = None
    try:
        import dds3
        return dds3
    except ImportError as ex:
        last_error = ex

    here = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "win32":
        plat = "win"
    elif sys.platform == "darwin":
        plat = "darwin"
    else:
        plat = "linux"

    # Look for a vendored dds3 under bin/, covering the layouts BEN runs in:
    #   repo:              src/ddsolver/ddsolver.py -> <repo>/bin       (here/../..)
    #   flattened (Docker): /app/ddsolver/ddsolver.py -> /app/bin       (here/..)
    #   plus BEN_HOME/bin and <cwd>/bin as fallbacks.
    bin_roots = [
        os.path.join(os.path.abspath(os.path.join(here, "..", "..")), "bin"),
        os.path.join(os.path.abspath(os.path.join(here, "..")), "bin"),
    ]
    if os.getenv("BEN_HOME"):
        bin_roots.append(os.path.join(os.getenv("BEN_HOME"), "bin"))
    bin_roots.append(os.path.join(os.getcwd(), "bin"))

    seen = set()
    for root in bin_roots:
        for cand in (os.path.join(root, "dds3-" + plat),
                     os.path.join(root, "dds3"),
                     root):
            if cand in seen:
                continue
            seen.add(cand)
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                import dds3
                return dds3
            except ImportError as ex:
                last_error = ex
                continue

    raise ImportError(
        "Could not import the 'dds3' extension (DDS 3.x Python interface).\n"
        "Build it from the DDS repository and install or vendor it:\n"
        "  bazel build -c opt //python:dds3_wheel_dist   # produces a wheel in dist/\n"
        "  pip install dist/dds3-*.whl\n"
        f"or place the built dds3 package in BEN's bin/dds3-{plat}/ directory.\n"
        f"A _dds3.so built for a different Python than {sys.version_info.major}."
        f"{sys.version_info.minor} will fail to load here.\n"
        "See src/ddsolver/README.md.\n"
        f"\nUnderlying import error: {last_error!r}"
    ) from last_error


# ddfast is looked for whatever BEN_DDS says: PIMC can use its libdds.so (ddfast_libdds)
# even when BEN's own solves run on dds3.
try:
    ddfast = _load_ddfast()
    _ddfast_error = None
except ImportError as _ex:
    ddfast = None
    _ddfast_error = _ex


def _choose_backend():
    choice = (os.getenv("BEN_DDS") or "").strip().lower()
    if choice not in ("", "ddfast", "dds3"):
        raise ValueError(f"BEN_DDS={choice!r}: use 'ddfast' or 'dds3' (or leave it unset)")
    if choice == "ddfast" and ddfast is None:
        raise _ddfast_error
    if choice == "dds3" or ddfast is None:
        return "dds3"
    return "ddfast"


BACKEND = _choose_backend()
dds3 = _load_dds3() if BACKEND == "dds3" else None


# ---------------------------------------------------------------------------------------------
# ddfast's tables, and its libdds.so for PIMC


def _tables_dir():
    """Where ddfast's endgame tables (about 1.1 GB) live: $DDFAST_TABLES, else
    build/tables in the ddfast checkout the package was loaded from. ddfast's own
    default (build/tables under the current directory) would put them wherever
    BEN happened to be started."""
    if os.getenv("DDFAST_TABLES"):
        return os.getenv("DDFAST_TABLES")
    if ddfast is None:
        return None
    pkg = os.path.dirname(os.path.abspath(ddfast.__file__))       # <ddfast>/python/ddfast
    root = os.path.abspath(os.path.join(pkg, "..", ".."))
    if os.path.isdir(os.path.join(root, "python")) and os.path.isdir(os.path.join(root, "include")):
        return os.path.join(root, "build", "tables")
    return None


# PIMC's DDS (ddfast's libdds.so, see ddfast_libdds) shares libddfast.so with this module and
# loads the tables on its first solve if nothing has yet, from $DDFAST_TABLES. Set it for the
# process so both find the same tables whichever solves first.
if _tables_dir() and not os.getenv("DDFAST_TABLES"):
    os.environ["DDFAST_TABLES"] = _tables_dir()

# The table files (about 1.1 GB) are mapped read-only rather than read into each process, so all
# BEN servers and other ddfast users on a machine share one copy. DDFAST_TABLE_MAP=0 (or empty)
# reads them in instead, ddfast's own default and faster to search on Linux's huge pages. Set
# for the process so PIMC's libdds, which shares the tables, maps them too if it loads first.
if os.getenv("DDFAST_TABLE_MAP") is None:
    os.environ["DDFAST_TABLE_MAP"] = "1"
MAP_TABLES = os.environ["DDFAST_TABLE_MAP"] not in ("", "0")


def ddfast_libdds():
    """Path of ddfast's libdds.so, DDS's C interface on ddfast (a drop-in libdds for BGADLL /
    PIMC), beside the ddfast package; None if it was not built (Linux only so far)."""
    if ddfast is None:
        return None
    path = os.path.join(os.path.dirname(os.path.abspath(ddfast.__file__)), "libdds.so")
    return path if sys.platform.startswith("linux") and os.path.isfile(path) else None


def ddfast_libddfast():
    """Path of ddfast's libddfast.so (libddfast.dylib on macOS): the solver and tables, shared
    by ddfast's Python module, its libdds and the C++ BGADLL, beside the ddfast package; None
    if not built."""
    if ddfast is None or sys.platform == "win32":
        return None
    name = "libddfast.dylib" if sys.platform == "darwin" else "libddfast.so"
    path = os.path.join(os.path.dirname(os.path.abspath(ddfast.__file__)), name)
    return path if os.path.isfile(path) else None


_tables_loaded = False


def _ensure_tables(verbose):
    """Load the tables once per process (ddfast would otherwise do it on the first
    solve, inside a bid or a card play). The first run builds them, which takes a
    few minutes."""
    global _tables_loaded
    if _tables_loaded:
        return
    tables = _tables_dir()
    if tables and not os.path.isdir(tables):
        sys.stderr.write(f"{Fore.YELLOW}ddfast: building the endgame tables in {tables} "
                         f"(once; a few minutes, about 1.1 GB){Style.RESET_ALL}\n")
    elif verbose:
        sys.stderr.write(f"ddfast: {'mapping' if MAP_TABLES else 'loading'} the endgame tables from {tables or 'build/tables'}\n")
    ddfast.init(tables, map_tables=MAP_TABLES)
    _tables_loaded = True


# ---------------------------------------------------------------------------------------------
# Threads


def _gevent_patched_threading():
    try:
        from gevent import monkey
        return monkey.is_module_patched('threading')
    except ImportError:
        return False


# gameapi.py runs gevent's monkey.patch_all(), which turns the threads of a
# concurrent.futures pool into greenlets sharing one OS thread, and a native call made
# straight from a greenlet holds up gevent's hub (every other request) until it returns.
# Under gevent both backends run their calls on gevent's own executor, which always uses
# native threads, so the greenlet waits cooperatively. (dds3 found by grand-slam-bridge,
# PR #198: 200 deals took 13 s instead of 2 s on 8 cores.)
_GEVENT = _gevent_patched_threading()
if _GEVENT:
    from gevent.monkey import get_original
    from gevent.threadpool import ThreadPoolExecutor as _NativeThreadPoolExecutor


def _available_cpus():
    """The CPUs this process may run on (its affinity), or all of them where that is unknown."""
    try:
        return len(os.sched_getaffinity(0)) or (os.cpu_count() or 4)
    except (AttributeError, OSError):
        return os.cpu_count() or 4


class _DDFastBackend:
    """A batch is one solve_boards_pbn call: ddfast solves the boards on its own threads
    with the GIL released once, and keeps its search state per thread by itself."""

    def __init__(self, max_threads, verbose):
        if not (max_threads and max_threads > 0):
            # 0 = one thread per CPU this process may run on. ddfast's own 0 is every CPU of
            # the machine, which oversubscribes a process confined to fewer (taskset, a
            # container's cpuset). The double dummy results do not depend on the threads.
            max_threads = _available_cpus()
        self.max_threads = max_threads
        self.workers = self.max_threads
        # Under gevent: a few native threads so concurrent requests can each have a
        # batch in flight (each batch is itself spread over ddfast's threads).
        self._gevent_pool = _NativeThreadPoolExecutor(max_workers=4) if _GEVENT else None
        _ensure_tables(verbose)

    def version(self):
        return "ddfast"

    def _call(self, fn, *args, **kwargs):
        if self._gevent_pool is not None:
            return self._gevent_pool.submit(fn, *args, **kwargs).result()
        return fn(*args, **kwargs)

    def solve_boards(self, hands_pbn, trump, first, trick_suit, trick_rank, solutions, mode):
        return self._call(ddfast.solve_boards_pbn, hands_pbn, trump=trump, first=first,
                          current_trick_suit=trick_suit, current_trick_rank=trick_rank,
                          target=-1, solutions=solutions, mode=mode, max_threads=self.max_threads)

    def par_table(self, deal, vulnerable):
        return self._call(ddfast.calc_all_tables_pbn, [deal], mode=vulnerable,
                          max_threads=self.max_threads)


# dds3 (DDS 3.0+) removed internal multi-threading from the legacy batch API, so the modern
# model is one SolverContext per worker thread: the backend runs a thread pool, and
# solve_board_pbn releases the GIL during the solve, so the threads run concurrently. Each
# pool thread keeps its own SolverContext - per OS thread, so under gevent the original
# thread-local, not the patched per-greenlet one.
_ctx_local = (get_original('_thread', '_local')() if _GEVENT else threading.local())


def _thread_context():
    ctx = getattr(_ctx_local, "ctx", None)
    if ctx is None:
        ctx = dds3.SolverContext()
        _ctx_local.ctx = ctx
    return ctx


class _DDS3Backend:
    """One solve_board_pbn call per board, on a Python thread pool with a SolverContext
    (and so a warm transposition table) per thread."""

    def __init__(self, max_threads, verbose):
        self.workers = max_threads if (max_threads and max_threads > 0) else (os.cpu_count() or 4)
        if _GEVENT:
            self._pool = _NativeThreadPoolExecutor(max_workers=self.workers)
        else:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="dds")

    def version(self):
        return "dds3 3.1.0"

    def solve_boards(self, hands_pbn, trump, first, trick_suit, trick_rank, solutions, mode):
        def _solve_one(pbn):
            return dds3.solve_board_pbn(pbn, trump=trump, first=first,
                                        current_trick_suit=trick_suit, current_trick_rank=trick_rank,
                                        target=-1, solutions=solutions, mode=mode,
                                        context=_thread_context())
        return list(self._pool.map(_solve_one, hands_pbn))

    def par_table(self, deal, vulnerable):
        return dds3.calc_all_tables_pbn([deal], mode=vulnerable)


# ---------------------------------------------------------------------------------------------


class DDSolver:

    # dds_mode is DDS's `mode`: 1 (the default) always searches for the score, even
    # when the hand to play has only one card (with its equivalents) to play. dds3
    # reuses its transposition table between calls (same trump, the same or nearly
    # the same cards); ddfast keeps its search state per thread and reuses it itself.

    def __init__(self, dds_mode=1, max_threads=0, verbose=False):
        self.dds_mode = dds_mode
        # max_threads: threads a batch is solved on. 0 = one per CPU.
        if BACKEND == "ddfast":
            self._backend = _DDFastBackend(max_threads, verbose)
        else:
            self._backend = _DDS3Backend(max_threads, verbose)
        self._workers = self._backend.workers
        if verbose:
            sys.stderr.write(f"DDSolver loaded — {self.version()} - dds mode {dds_mode} - {self._workers} solver threads"
                             f"{' (native threads under gevent)' if _GEVENT else ''}\n")
        # No-op unless --ddsrecord / BEN_DDS_RECORD asked for a recording. If the
        # entry point already opened one, this just labels it with the solver
        # actually in use. See ddsrecorder.py / ddsreplay.py.
        DDSRecorder.configure(dds_version=self.version(), dds_mode=dds_mode,
                              threads=self._workers)

    def version(self):
        return self._backend.version()

    def calculatepar(self, hand, vuln, print_result=True):
        with ModelTimer.time_call('dds_par'):
            t0 = time.perf_counter()
            result = self._calculatepar_impl(hand, vuln, print_result)
            if DDSRecorder.enabled:
                DDSRecorder.record_par(hand, vuln, result,
                                       (time.perf_counter() - t0) * 1000)
            return result

    def _calculatepar_impl(self, hand, vuln, print_result=True):
        # vulnerable
        # 0: None 1: Both 2: NS 3: EW
        v = 0
        if vuln[0]: v = 2
        if vuln[1]: v = 3
        if vuln[0] and vuln[1]: v = 1

        # Callers pass the deal both with and without the "N:" prefix. dds3 accepts
        # a doubled "N:N:..."; ddfast rejects it as a PBN error.
        deal = hand if hand[:2].upper() in ("N:", "E:", "S:", "W:") else "N:" + hand

        # calc_all_tables_pbn computes the DD table and (mode != -1) the par score.
        try:
            result = self._backend.par_table(deal, v)
        except Exception as e:
            sys.stderr.write(f"Error calculating par: {e}, Hand {hand.encode('utf-8')}\n")
            raise

        par_results = result.get("par_results") or []
        if not par_results:
            sys.stderr.write(f"{Fore.RED}No par result for hand {hand.encode('utf-8')}{Style.RESET_ALL}\n")
            return None

        par = par_results[0]
        # par_score is a 2-element list of strings, e.g. "NS 420" / "EW -420".
        ns_par = par["par_score"][0]
        ew_par = par["par_score"][1]

        if print_result:
            print("NS score: {}".format(ns_par))
            print("EW score: {}".format(ew_par))

        return int(ns_par.split()[1])

    # Solutions
    #1	Find the maximum number of tricks for the side to play.  Return only one of the optimum cards and its score.
    #2	Find the maximum number of tricks for the side to play.  Return all optimum cards and their scores.
    #3	Return all cards that can be legally played, with their scores in descending order.

    @staticmethod
    def _trick_number(hands_pbn, current_trick):
        """Derive trick number (1-13) from remaining cards in PBN hand."""
        pbn = hands_pbn[0]
        if ':' in pbn:
            pbn = pbn.split(':', 1)[1]
        remaining = sum(1 for c in pbn if c not in '. ')
        return (52 - remaining - len(current_trick)) // 4 + 1

    def solve(self, strain_i, leader_i, current_trick, hands_pbn, solutions, purpose=""):
        """Solve a batch of sampled hands with double-dummy.

        purpose is a short tag describing *why* the solver was invoked (e.g. "play",
        "claimcheck", "lead", "bid"). It is folded into the timing label so the
        MODEL TIMING SUMMARY separates, say, card-play evaluations from claim
        checks at the same trick instead of lumping them into one opaque count.
        """
        trick = self._trick_number(hands_pbn, current_trick)
        label = f'dds_solve_t{trick:02d}_{purpose}' if purpose else f'dds_solve_t{trick:02d}'
        with ModelTimer.time_call(label, items=len(hands_pbn)):
            t0 = time.perf_counter()
            results = self.solve_helper(strain_i, leader_i, current_trick, hands_pbn, solutions)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        if DDSRecorder.enabled:
            DDSRecorder.record_solve(strain_i, leader_i, current_trick, hands_pbn,
                                     solutions, purpose, trick, results, elapsed_ms)

        return results

    def solve_helper(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        card_rank = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]

        trump = (strain_i - 1) % 5

        # The current trick is the same for every board in the batch.
        trick_suit = [0, 0, 0]
        trick_rank = [0, 0, 0]
        for i in range(min(3, len(current_trick))):
            trick_suit[i] = current_trick[i] // 13
            trick_rank[i] = 14 - current_trick[i] % 13

        try:
            solved = self._backend.solve_boards(list(hands_pbn), trump, leader_i, tuple(trick_suit),
                                                tuple(trick_rank), solutions, self.dds_mode)
        except Exception as e:
            print(f"{Fore.RED}DDS error: {e} {hands_pbn[0].encode('utf-8')} {current_trick} {leader_i}{Style.RESET_ALL}")
            return None

        if solutions == 1:
            # Just return the maximum number of the side to play for each sample
            card_results = {}
            card_results["max"] = []
            card_results["min"] = []
            for fut in solved:
                card_results["max"].append(fut["score"][0])
                card_results["min"].append(fut["score"][fut["cards"] - 1])

        else:
            card_results = {}
            for fut in solved:
                suits = fut["suit"]
                ranks = fut["rank"]
                scores = fut["score"]
                equals = fut["equals"]
                for i in range(fut["cards"]):
                    suit_i = suits[i]
                    score = scores[i]
                    card = suit_i * 13 + 14 - ranks[i]
                    if card not in card_results:
                        card_results[card] = []
                    card_results[card].append(score)
                    eq_cards_encoded = equals[i]
                    if eq_cards_encoded:
                        for k, rank_code in enumerate(card_rank):
                            if rank_code & eq_cards_encoded > 0:
                                eq_card = suit_i * 13 + k
                                if eq_card not in card_results:
                                    card_results[eq_card] = []
                                card_results[eq_card].append(score)

        return card_results


    def expected_tricks_dds(self, card_results):
        return {card:round((sum(values)/len(values)),2) for card, values in card_results.items()}

    def expected_tricks_dds_probability(self, card_results, probabilities_list : List[float]):
        # Convert to plain Python list to avoid numpy scalar overhead
        probs = [float(p) for p in probabilities_list]
        return {card: round(sum(p*res for p, res in zip(probs, result_list)),2) for card, result_list in card_results.items()}

    def p_made_target(self, tricks_needed):

        def fun(card_results):
            return {card:round(sum(1 for x in values if x >= tricks_needed)/len(values),3) for card, values in card_results.items()}
        return fun

    def print_dd_results(self, dd_solved, print_result=True, xcards=False):
        print("DD Result\n".join(
            f"{Card.from_code(int(k))}: [{', '.join(f'{x:>2}' for x in v[:20])}{' ]' if len(v) <= 20 else ' ...]'}"
            for k, v in dd_solved.items()
        ))

        # Create a new dictionary to store sorted counts for each key
        sorted_counts_dict = {}

        # Loop through the dictionary and process each key-value pair
        for key, array in dd_solved.items():
            # Use Counter to count the occurrences of each element
            element_count = Counter(array)

            # Sort the counts by frequency in descending order
            sorted_counts = sorted(element_count.items(), key=lambda x: x[1], reverse=True)

            # Store the sorted result in the new dictionary
            sorted_counts_dict[key] = sorted_counts

        # Print the sorted counts for each key
        for key, sorted_counts in sorted_counts_dict.items():
            print(f"Sorted counts for {Card.from_code(int(key), xcards)} DD:")
            for value, count in sorted_counts:
                print(f"  Tricks: {value}, Count: {count}")
