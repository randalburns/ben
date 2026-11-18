import sys
import os
import time
from typing import List
from collections import Counter
from objects import Card
from colorama import Fore, Back, Style, init
from nn.timing import ModelTimer
from ddsolver.ddsrecorder import DDSRecorder

init()


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
    """Import ddfast, an exact double-dummy solver with DDS 3.x's dds3 interface.

    Tries a normal import first (ddfast installed), then the extern/ddfast
    submodule, whose extension python/build.sh builds into python/ddfast/.
    """
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
        "  sh extern/ddfast/python/build.sh      # -> extern/ddfast/python/ddfast/_ddfast.abi3.so\n"
        "The extension uses CPython's stable ABI (3.12 and later) and needs AVX-512 on x86-64.\n"
        f"\nUnderlying import error: {last_error!r}"
    ) from last_error


ddfast = _load_ddfast()


def _tables_dir():
    """Where ddfast's endgame tables (about 1.1 GB) live: $DDFAST_TABLES, else
    build/tables in the ddfast checkout the package was loaded from. ddfast's own
    default (build/tables under the current directory) would put them wherever
    BEN happened to be started."""
    if os.getenv("DDFAST_TABLES"):
        return os.getenv("DDFAST_TABLES")
    pkg = os.path.dirname(os.path.abspath(ddfast.__file__))       # <ddfast>/python/ddfast
    root = os.path.abspath(os.path.join(pkg, "..", ".."))
    if os.path.isdir(os.path.join(root, "python")) and os.path.isdir(os.path.join(root, "include")):
        return os.path.join(root, "build", "tables")
    return None


_initialised = False


def _ensure_tables(verbose):
    """Load the tables once per process (ddfast would otherwise do it on the first
    solve, inside a bid or a card play). The first run builds them, which takes a
    few minutes."""
    global _initialised
    if _initialised:
        return
    tables = _tables_dir()
    if tables and not os.path.isdir(tables):
        sys.stderr.write(f"{Fore.YELLOW}ddfast: building the endgame tables in {tables} "
                         f"(once; a few minutes, about 1.1 GB){Style.RESET_ALL}\n")
    elif verbose:
        sys.stderr.write(f"ddfast: loading the endgame tables from {tables or 'build/tables'}\n")
    ddfast.init(tables)
    _initialised = True


def _gevent_patched_threading():
    try:
        from gevent import monkey
        return monkey.is_module_patched('threading')
    except ImportError:
        return False


# gameapi.py runs gevent's monkey.patch_all(). A ddfast call releases the GIL, but
# made straight from a greenlet it would still hold up gevent's hub (every other
# request) until the batch is solved. Under gevent the call runs on gevent's
# native-thread pool instead, so the greenlet waits cooperatively.
_GEVENT = _gevent_patched_threading()
if _GEVENT:
    from gevent.threadpool import ThreadPoolExecutor as _NativeThreadPoolExecutor


class DDSolver:

    # dds_mode is DDS's `mode`: 1 (the default) always searches for the score, even
    # when the hand to play has only one card (with its equivalents) to play.
    # ddfast keeps its search state per thread and reuses it by itself, so DDS's
    # transposition-table reuse behind the mode needs no context here.

    def __init__(self, dds_mode=1, max_threads=0, verbose=False):
        self.dds_mode = dds_mode
        # max_threads: threads ddfast solves a batch on. 0 = one per CPU.
        self._max_threads = max_threads if (max_threads and max_threads > 0) else 0
        self._workers = self._max_threads or (os.cpu_count() or 4)
        # Under gevent: a few native threads so concurrent requests can each have
        # a batch in flight (each batch is itself spread over ddfast's threads).
        self._gevent_pool = _NativeThreadPoolExecutor(max_workers=4) if _GEVENT else None
        _ensure_tables(verbose)
        if verbose:
            sys.stderr.write(f"DDSolver loaded — {self.version()} - dds mode {dds_mode} - {self._workers} solver threads"
                             f"{' (native threads under gevent)' if _GEVENT else ''}\n")
        # No-op unless --ddsrecord / BEN_DDS_RECORD asked for a recording. If the
        # entry point already opened one, this just labels it with the solver
        # actually in use. See ddsrecorder.py / ddsreplay.py.
        DDSRecorder.configure(dds_version=self.version(), dds_mode=dds_mode,
                              threads=self._workers)

    def version(self):
        return "ddfast"

    def _call(self, fn, *args, **kwargs):
        if self._gevent_pool is not None:
            return self._gevent_pool.submit(fn, *args, **kwargs).result()
        return fn(*args, **kwargs)

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

        # Callers pass the deal both with and without the "N:" prefix. dds3 accepted
        # a doubled "N:N:..."; ddfast rejects it as a PBN error.
        deal = hand if hand[:2].upper() in ("N:", "E:", "S:", "W:") else "N:" + hand

        # calc_all_tables_pbn computes the DD table and (mode != -1) the par score.
        try:
            result = self._call(ddfast.calc_all_tables_pbn, [deal], mode=v,
                                max_threads=self._max_threads)
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

        # One call per batch: ddfast solves the boards on its own threads with the
        # GIL released once.
        try:
            solved = self._call(
                ddfast.solve_boards_pbn,
                list(hands_pbn),
                trump=trump,
                first=leader_i,
                current_trick_suit=tuple(trick_suit),
                current_trick_rank=tuple(trick_rank),
                target=-1,
                solutions=solutions,
                mode=self.dds_mode,
                max_threads=self._max_threads,
            )
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
                for i in range(fut["cards"]):
                    suit_i = fut["suit"][i]
                    card = suit_i * 13 + 14 - fut["rank"][i]
                    if card not in card_results:
                        card_results[card] = []
                    card_results[card].append(fut["score"][i])
                    eq_cards_encoded = fut["equals"][i]
                    for k, rank_code in enumerate(card_rank):
                        if rank_code & eq_cards_encoded > 0:
                            eq_card = suit_i * 13 + k
                            if eq_card not in card_results:
                                card_results[eq_card] = []
                            card_results[eq_card].append(fut["score"][i])

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
