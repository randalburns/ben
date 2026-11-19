"""
Native ctypes wrapper for BGADLL - provides the same interface as the pythonnet
classes (PIMC, PIMCDef, Hand, Play, Card, Constraints, Extensions, Macros)
so that PIMC.py and PIMCDef.py can work on all platforms without pythonnet.

Two engines have the same C ABI (the bga_* functions):
  cpp     the C++ port (extern/BGA/BGADLL.Cpp, Linux), solving on ddfast
  csharp  BGADLL.Native (NativeAOT C#) under bin/BGA, solving on a libdds
BEN_BGADLL=cpp|csharp picks one; unset, the C++ port if it is built.
"""
import ctypes
import os
import sys
import platform
from ctypes import c_int, c_double, c_char_p, c_void_p, POINTER


def _find_native_lib():
    """Find the platform-specific BGADLL native library."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(script_dir, "../..")

    if "src" in script_dir and "pimc" in script_dir:
        bin_folder = os.path.join(parent_dir, 'bin')
    else:
        ben_home = os.getenv('BEN_HOME') or '.'
        bin_folder = os.path.join(ben_home, 'bin') if ben_home != '.' else 'bin'

    machine = platform.machine().lower()
    if sys.platform == 'win32':
        if machine in ('amd64', 'x86_64', 'x64'):
            arch = 'x64'
        elif machine in ('arm64', 'aarch64'):
            arch = 'arm64'
        else:
            arch = 'x64'
        lib_name = 'BGADLL.dll'
        platform_dir = 'windows'
    elif sys.platform == 'darwin':
        arch = 'arm64'
        lib_name = 'BGADLL.dylib'
        platform_dir = 'macos'
    else:
        if machine in ('aarch64', 'arm64'):
            arch = 'arm64'
        else:
            arch = 'x64'
        lib_name = 'BGADLL.so'
        platform_dir = 'linux'

    return os.path.join(bin_folder, 'BGA', platform_dir, arch, lib_name)


def _find_cpp_lib():
    """The C++ port of BGADLL (extern/BGA, branch cpp-ddfast-port, built by
    BGADLL.Cpp/build.sh): the same C ABI as the C# library, with ddfast as its
    double-dummy solver. Linux only; None if it is not built."""
    if not sys.platform.startswith('linux'):
        return None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    roots = [os.path.abspath(os.path.join(script_dir, '..', '..')),   # repo: src/pimc -> <repo>
             os.path.abspath(os.path.join(script_dir, '..'))]         # flattened: /app/pimc -> /app
    if os.getenv('BEN_HOME'):
        roots.append(os.getenv('BEN_HOME'))
    for root in roots:
        path = os.path.join(root, 'extern', 'BGA', 'BGADLL.Cpp', 'bin', 'BGADLL.so')
        if os.path.isfile(path):
            return path
    return None


def _choose_engine():
    """BEN_BGADLL=cpp or csharp picks the PIMC engine; unset, the C++ port if it is
    built, else the C# library under bin/BGA."""
    choice = os.getenv('BEN_BGADLL', '').strip().lower()
    if choice not in ('', 'cpp', 'csharp'):
        raise ValueError(f"BEN_BGADLL={choice!r}: use 'cpp' or 'csharp' (or leave it unset)")
    cpp = _find_cpp_lib()
    if choice == 'cpp' and cpp is None:
        raise FileNotFoundError("BEN_BGADLL=cpp but extern/BGA/BGADLL.Cpp/bin/BGADLL.so is not built "
                                "(git submodule update --init extern/BGA; sh extern/BGA/BGADLL.Cpp/build.sh)")
    if choice != 'csharp' and cpp is not None:
        return 'cpp', cpp
    return 'csharp', _find_native_lib()


# engine: "cpp" (the C++ port) or "csharp" (BGADLL.Native, NativeAOT C#).
engine, _lib_path = _choose_engine()
_lib = None
# The double-dummy solver BGADLL ended up with: "ddfast", or the path of the libdds it uses.
dds_library = None


def _preload_ddfast_libdds():
    """C# engine: preload ddfast's libdds.so (DDS's C interface on ddfast) so BGADLL's
    [DllImport("dds")] binds to it: the loader matches the request to the library
    already loaded under the soname libdds.so, and the old libdds beside BGADLL is
    never opened. It shares libddfast.so (solver and tables) with BEN's DDSolver.
    BEN_PIMC_DDS=haglund keeps the vendored DDS library (3.1.0 on Linux). Returns
    True if preloaded."""
    global dds_library
    if os.getenv('BEN_PIMC_DDS', '').lower() == 'haglund':
        return False
    try:
        from ddsolver.ddsolver import ddfast_libdds
    except ImportError:
        return False
    path = ddfast_libdds()
    if path is None:
        return False
    ctypes.CDLL(path)
    dds_library = "ddfast"
    return True


def _vendored_libdds():
    """The DDS library shipped beside the C# BGADLL (DDS 3.1.0 on Linux x64)."""
    machine = platform.machine().lower()
    arch = 'arm64' if machine in ('aarch64', 'arm64') else 'x64'
    path = os.path.join(os.path.dirname(os.path.abspath(_find_native_lib())), '..', arch, 'libdds.so')
    path = os.path.abspath(path)
    return path if os.path.isfile(path) else None


def _setup_cpp_solver():
    """Point the C++ port at its solvers before it is loaded. It opens them itself
    (dlopen), using a library already in the process first: BEN's DDSolver has
    loaded libddfast.so, so PIMC shares its tables. BEN_PIMC_DDS=haglund makes it use
    the vendored DDS library instead (BGA_DDS_BACKEND=haglund); the vendored library
    is also its fallback when ddfast is not available."""
    try:
        from ddsolver.ddsolver import ddfast_libddfast   # loads libddfast.so, sets DDFAST_TABLES
        libddfast = ddfast_libddfast()
    except ImportError:
        libddfast = None
    if libddfast:
        os.environ.setdefault('BGA_DDFAST_LIB', libddfast)
    libdds = _vendored_libdds()
    if libdds:
        os.environ.setdefault('BGA_DDS_LIB', libdds)
    if os.getenv('BEN_PIMC_DDS', '').lower() == 'haglund':
        os.environ.setdefault('BGA_DDS_BACKEND', 'haglund')


def _cpp_backend(lib):
    lib.bga_dds_backend.argtypes = []
    lib.bga_dds_backend.restype = c_void_p
    name = ctypes.string_at(lib.bga_dds_backend()).decode('utf-8', 'replace')
    if name == 'haglund':
        return os.getenv('BGA_DDS_LIB') or 'libdds.so'
    return name


def _get_lib():
    global _lib, dds_library
    if _lib is None and engine == 'cpp':
        return _get_cpp_lib()
    if _lib is None:
        if not os.path.isfile(_lib_path):
            raise FileNotFoundError(f"BGADLL native library not found: {_lib_path}")
        # BGADLL depends on dds.dll (Haglund) - preload it so NativeAOT P/Invoke can resolve it
        lib_dir = os.path.dirname(os.path.abspath(_lib_path))
        bin_dir = os.path.dirname(os.path.dirname(os.path.dirname(lib_dir)))  # bin/BGA/platform/arch -> bin/
        if sys.platform == 'win32':
            os.add_dll_directory(lib_dir)
            os.add_dll_directory(os.path.abspath(bin_dir))
            # Look for dds.dll next to BGADLL first, then in bin/
            for dds_dir in [lib_dir, bin_dir]:
                dds_path = os.path.join(dds_dir, 'dds.dll')
                if os.path.isfile(dds_path):
                    ctypes.WinDLL(dds_path)
                    break
        elif _preload_ddfast_libdds():
            pass
        else:
            # On Linux/macOS, preload dds library from next to BGADLL or bin/
            if sys.platform == 'darwin':
                candidates = [
                    os.path.join(lib_dir, 'libdds.2.9.0.dylib'),
                    os.path.join(lib_dir, 'libdds.dylib'),
                    os.path.join(bin_dir, 'darwin', 'libdds.2.9.0.dylib'),
                ]
            else:
                candidates = [
                    os.path.join(lib_dir, 'libdds.so'),
                    os.path.join(bin_dir, 'libdds.so'),
                ]
            for dds_path in candidates:
                if os.path.isfile(dds_path):
                    ctypes.CDLL(dds_path)
                    dds_library = dds_path
                    # BGADLL's [DllImport("dds")] resolves a platform-named file
                    # next to the assembly: libdds.so on Linux, libdds.dylib /
                    # dds.dylib on macOS. The vendored file may have a versioned
                    # name (e.g. libdds.2.9.0.dylib), so create the expected
                    # symlink(s) beside it if missing.
                    if sys.platform == 'darwin':
                        link_names = ['libdds.dylib', 'dds.dylib']
                    else:
                        link_names = ['libdds.so']
                    for link_name in link_names:
                        link = os.path.join(os.path.dirname(dds_path), link_name)
                        if not os.path.exists(link):
                            try:
                                os.symlink(os.path.basename(dds_path), link)
                            except OSError:
                                pass
                    break
        _lib = ctypes.CDLL(_lib_path)
        _setup_functions(_lib)
    return _lib


def _get_cpp_lib():
    global _lib, dds_library
    _setup_cpp_solver()
    _lib = ctypes.CDLL(_lib_path)
    _setup_functions(_lib)
    dds_library = _cpp_backend(_lib)
    return _lib


def _setup_functions(lib):
    """Set up function signatures for all FFI exports."""
    # String management
    lib.bga_free_string.argtypes = [c_void_p]
    lib.bga_free_string.restype = None

    # PIMC
    lib.bga_pimc_create.argtypes = [c_int, c_int]
    lib.bga_pimc_create.restype = c_void_p
    lib.bga_pimc_create_default.argtypes = []
    lib.bga_pimc_create_default.restype = c_void_p
    lib.bga_pimc_destroy.argtypes = [c_void_p]
    lib.bga_pimc_destroy.restype = None
    lib.bga_pimc_version.argtypes = [c_void_p]
    lib.bga_pimc_version.restype = c_void_p
    lib.bga_pimc_clear.argtypes = [c_void_p]
    lib.bga_pimc_clear.restype = None
    lib.bga_pimc_setup.argtypes = [c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    lib.bga_pimc_setup.restype = c_int
    lib.bga_pimc_evaluate.argtypes = [c_void_p, c_int]
    lib.bga_pimc_evaluate.restype = c_int
    lib.bga_pimc_await.argtypes = [c_void_p, c_int]
    lib.bga_pimc_await.restype = c_int
    lib.bga_pimc_get_playouts.argtypes = [c_void_p]
    lib.bga_pimc_get_playouts.restype = c_int
    lib.bga_pimc_get_combinations.argtypes = [c_void_p]
    lib.bga_pimc_get_combinations.restype = c_int
    lib.bga_pimc_get_examined.argtypes = [c_void_p]
    lib.bga_pimc_get_examined.restype = c_int
    lib.bga_pimc_get_evaluating.argtypes = [c_void_p]
    lib.bga_pimc_get_evaluating.restype = c_int
    lib.bga_pimc_get_legal_moves.argtypes = [c_void_p]
    lib.bga_pimc_get_legal_moves.restype = c_void_p
    lib.bga_pimc_get_legal_moves_to_string.argtypes = [c_void_p]
    lib.bga_pimc_get_legal_moves_to_string.restype = c_void_p
    lib.bga_pimc_output_sort.argtypes = [c_void_p]
    lib.bga_pimc_output_sort.restype = None
    lib.bga_pimc_output_get_tricks.argtypes = [c_void_p, c_char_p, c_void_p, c_void_p, c_int]
    lib.bga_pimc_output_get_tricks.restype = c_int

    # PIMCDef
    lib.bga_pimcdef_create.argtypes = [c_int, c_int]
    lib.bga_pimcdef_create.restype = c_void_p
    lib.bga_pimcdef_create_default.argtypes = []
    lib.bga_pimcdef_create_default.restype = c_void_p
    lib.bga_pimcdef_destroy.argtypes = [c_void_p]
    lib.bga_pimcdef_destroy.restype = None
    lib.bga_pimcdef_version.argtypes = [c_void_p]
    lib.bga_pimcdef_version.restype = c_void_p
    lib.bga_pimcdef_clear.argtypes = [c_void_p]
    lib.bga_pimcdef_clear.restype = None
    lib.bga_pimcdef_setup.argtypes = [c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    lib.bga_pimcdef_setup.restype = c_int
    lib.bga_pimcdef_evaluate.argtypes = [c_void_p, c_int]
    lib.bga_pimcdef_evaluate.restype = c_int
    lib.bga_pimcdef_await.argtypes = [c_void_p, c_int]
    lib.bga_pimcdef_await.restype = c_int
    lib.bga_pimcdef_get_playouts.argtypes = [c_void_p]
    lib.bga_pimcdef_get_playouts.restype = c_int
    lib.bga_pimcdef_get_combinations.argtypes = [c_void_p]
    lib.bga_pimcdef_get_combinations.restype = c_int
    lib.bga_pimcdef_get_examined.argtypes = [c_void_p]
    lib.bga_pimcdef_get_examined.restype = c_int
    lib.bga_pimcdef_get_evaluating.argtypes = [c_void_p]
    lib.bga_pimcdef_get_evaluating.restype = c_int
    lib.bga_pimcdef_get_legal_moves.argtypes = [c_void_p]
    lib.bga_pimcdef_get_legal_moves.restype = c_void_p
    lib.bga_pimcdef_get_legal_moves_to_string.argtypes = [c_void_p]
    lib.bga_pimcdef_get_legal_moves_to_string.restype = c_void_p
    lib.bga_pimcdef_output_sort.argtypes = [c_void_p]
    lib.bga_pimcdef_output_sort.restype = None
    lib.bga_pimcdef_output_get_tricks.argtypes = [c_void_p, c_char_p, c_void_p, c_void_p, c_int]
    lib.bga_pimcdef_output_get_tricks.restype = c_int

    # Hand
    lib.bga_hand_create.argtypes = []
    lib.bga_hand_create.restype = c_void_p
    lib.bga_hand_destroy.argtypes = [c_void_p]
    lib.bga_hand_destroy.restype = None
    lib.bga_hand_parse.argtypes = [c_char_p]
    lib.bga_hand_parse.restype = c_void_p
    lib.bga_hand_to_string.argtypes = [c_void_p]
    lib.bga_hand_to_string.restype = c_void_p
    lib.bga_hand_count.argtypes = [c_void_p]
    lib.bga_hand_count.restype = c_int
    lib.bga_hand_union.argtypes = [c_void_p, c_void_p]
    lib.bga_hand_union.restype = c_void_p
    lib.bga_hand_except.argtypes = [c_void_p, c_void_p]
    lib.bga_hand_except.restype = c_void_p
    lib.bga_hand_remove.argtypes = [c_void_p, c_void_p]
    lib.bga_hand_remove.restype = c_int
    lib.bga_hand_add.argtypes = [c_void_p, c_void_p]
    lib.bga_hand_add.restype = None
    lib.bga_hand_add_range.argtypes = [c_void_p, c_void_p]
    lib.bga_hand_add_range.restype = None

    # Play
    lib.bga_play_create.argtypes = []
    lib.bga_play_create.restype = c_void_p
    lib.bga_play_destroy.argtypes = [c_void_p]
    lib.bga_play_destroy.restype = None
    lib.bga_play_clear.argtypes = [c_void_p]
    lib.bga_play_clear.restype = None
    lib.bga_play_add.argtypes = [c_void_p, c_void_p]
    lib.bga_play_add.restype = None
    lib.bga_play_add_range.argtypes = [c_void_p, c_void_p]
    lib.bga_play_add_range.restype = None
    lib.bga_play_count.argtypes = [c_void_p]
    lib.bga_play_count.restype = c_int
    lib.bga_play_list_as_string.argtypes = [c_void_p]
    lib.bga_play_list_as_string.restype = c_void_p

    # Card
    lib.bga_card_create.argtypes = [c_char_p]
    lib.bga_card_create.restype = c_void_p
    lib.bga_card_destroy.argtypes = [c_void_p]
    lib.bga_card_destroy.restype = None
    lib.bga_card_to_string.argtypes = [c_void_p]
    lib.bga_card_to_string.restype = c_void_p

    # Constraints
    lib.bga_constraints_create.argtypes = [c_int] * 10
    lib.bga_constraints_create.restype = c_void_p
    lib.bga_constraints_destroy.argtypes = [c_void_p]
    lib.bga_constraints_destroy.restype = None
    lib.bga_constraints_to_string.argtypes = [c_void_p]
    lib.bga_constraints_to_string.restype = c_void_p
    lib.bga_constraints_set.argtypes = [c_void_p] + [c_int] * 10
    lib.bga_constraints_set.restype = None
    lib.bga_constraints_get.argtypes = [c_void_p, c_void_p]
    lib.bga_constraints_get.restype = None
    lib.bga_constraints_set_prop.argtypes = [c_void_p, c_int, c_int]
    lib.bga_constraints_set_prop.restype = None
    lib.bga_constraints_get_prop.argtypes = [c_void_p, c_int]
    lib.bga_constraints_get_prop.restype = c_int


def _read_string(ptr):
    """Read a UTF-8 string from a native pointer and free it."""
    if not ptr:
        return ""
    lib = _get_lib()
    s = ctypes.string_at(ptr).decode('utf-8')
    lib.bga_free_string(ptr)
    return s


# ===== Wrapper classes mimicking pythonnet interface =====

class TricksWeightEntry:
    """Mimics .NET ValueTuple with Item1 (tricks) and Item2 (weight)."""
    def __init__(self, tricks, weight):
        self.Item1 = tricks
        self.Item2 = weight
        self.tricks = tricks
        self.weight = weight


class _OutputWrapper:
    """Wraps PIMC/PIMCDef Output property - provides SortResults and GetTricksWithWeights."""
    def __init__(self, pimc_handle, is_def=False):
        self._handle = pimc_handle
        self._prefix = "bga_pimcdef" if is_def else "bga_pimc"

    def SortResults(self):
        lib = _get_lib()
        sort_fn = getattr(lib, f"{self._prefix}_output_sort")
        sort_fn(self._handle)

    def GetTricksWithWeights(self, card):
        lib = _get_lib()
        get_fn = getattr(lib, f"{self._prefix}_output_get_tricks")
        card_str = str(card).encode('utf-8')
        max_entries = 256
        tricks_buf = (c_int * max_entries)()
        weights_buf = (c_double * max_entries)()
        count = get_fn(self._handle, card_str,
                       ctypes.cast(tricks_buf, c_void_p),
                       ctypes.cast(weights_buf, c_void_p),
                       max_entries)
        return [TricksWeightEntry(tricks_buf[i], weights_buf[i]) for i in range(count)]


class NativePIMC:
    """Wraps bga_pimc_* functions with the same interface as pythonnet PIMC class."""
    def __init__(self, maxThreads=0, verbose=False):
        self._lib = _get_lib()
        if maxThreads == 0 and not verbose:
            self._handle = self._lib.bga_pimc_create_default()
        else:
            self._handle = self._lib.bga_pimc_create(maxThreads, 1 if verbose else 0)
        self.Output = _OutputWrapper(self._handle, is_def=False)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and self._lib:
            try:
                self._lib.bga_pimc_destroy(self._handle)
            except Exception:
                pass

    def version(self):
        return _read_string(self._lib.bga_pimc_version(self._handle))

    def Clear(self):
        self._lib.bga_pimc_clear(self._handle)

    def SetupEvaluation(self, hands, oppos, current_trick, previous_tricks, consts, player, maxPlayout, autoplaySingleton, useStrategy):
        hand_ptrs = (c_void_p * len(hands))(*[h._handle for h in hands])
        result = self._lib.bga_pimc_setup(
            self._handle,
            ctypes.cast(hand_ptrs, c_void_p), len(hands),
            oppos._handle,
            current_trick._handle,
            previous_tricks._handle,
            consts[0]._handle,
            consts[1]._handle,
            int(player),
            maxPlayout,
            1 if autoplaySingleton else 0,
            1 if useStrategy else 0
        )
        if result != 0:
            raise RuntimeError("PIMC SetupEvaluation failed")

    def Evaluate(self, trump):
        result = self._lib.bga_pimc_evaluate(self._handle, int(trump))
        if result != 0:
            raise RuntimeError("PIMC Evaluate failed")

    def AwaitEvaluation(self, maxWaitMs):
        self._lib.bga_pimc_await(self._handle, maxWaitMs)

    @property
    def Playouts(self):
        return self._lib.bga_pimc_get_playouts(self._handle)

    @property
    def Combinations(self):
        return self._lib.bga_pimc_get_combinations(self._handle)

    @property
    def Examined(self):
        return self._lib.bga_pimc_get_examined(self._handle)

    @property
    def Evaluating(self):
        return self._lib.bga_pimc_get_evaluating(self._handle) != 0

    @property
    def LegalMoves(self):
        ptr = self._lib.bga_pimc_get_legal_moves(self._handle)
        s = _read_string(ptr)
        if not s:
            return []
        return s.split('|')

    @property
    def LegalMovesToString(self):
        return _read_string(self._lib.bga_pimc_get_legal_moves_to_string(self._handle))


class NativePIMCDef:
    """Wraps bga_pimcdef_* functions with the same interface as pythonnet PIMCDef class."""
    def __init__(self, maxThreads=0, verbose=False):
        self._lib = _get_lib()
        if maxThreads == 0 and not verbose:
            self._handle = self._lib.bga_pimcdef_create_default()
        else:
            self._handle = self._lib.bga_pimcdef_create(maxThreads, 1 if verbose else 0)
        self.Output = _OutputWrapper(self._handle, is_def=True)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and self._lib:
            try:
                self._lib.bga_pimcdef_destroy(self._handle)
            except Exception:
                pass

    def version(self):
        return _read_string(self._lib.bga_pimcdef_version(self._handle))

    def Clear(self):
        self._lib.bga_pimcdef_clear(self._handle)

    def SetupEvaluation(self, hands, oppos, current_trick, previous_tricks, consts, player, maxPlayout, autoplaySingleton, overDummy):
        hand_ptrs = (c_void_p * len(hands))(*[h._handle for h in hands])
        result = self._lib.bga_pimcdef_setup(
            self._handle,
            ctypes.cast(hand_ptrs, c_void_p), len(hands),
            oppos._handle,
            current_trick._handle,
            previous_tricks._handle,
            consts[0]._handle,
            consts[1]._handle,
            int(player),
            maxPlayout,
            1 if autoplaySingleton else 0,
            1 if overDummy else 0
        )
        if result != 0:
            raise RuntimeError("PIMCDef SetupEvaluation failed")

    def Evaluate(self, trump):
        result = self._lib.bga_pimcdef_evaluate(self._handle, int(trump))
        if result != 0:
            raise RuntimeError("PIMCDef Evaluate failed")

    def AwaitEvaluation(self, maxWaitMs):
        self._lib.bga_pimcdef_await(self._handle, maxWaitMs)

    @property
    def Playouts(self):
        return self._lib.bga_pimcdef_get_playouts(self._handle)

    @property
    def Combinations(self):
        return self._lib.bga_pimcdef_get_combinations(self._handle)

    @property
    def Examined(self):
        return self._lib.bga_pimcdef_get_examined(self._handle)

    @property
    def Evaluating(self):
        return self._lib.bga_pimcdef_get_evaluating(self._handle) != 0

    @property
    def LegalMoves(self):
        ptr = self._lib.bga_pimcdef_get_legal_moves(self._handle)
        s = _read_string(ptr)
        if not s:
            return []
        return s.split('|')

    @property
    def LegalMovesToString(self):
        return _read_string(self._lib.bga_pimcdef_get_legal_moves_to_string(self._handle))


class NativeHand:
    """Wraps bga_hand_* functions with the same interface as pythonnet Hand class."""
    def __init__(self, handle=None):
        self._lib = _get_lib()
        if handle is not None:
            self._handle = handle
        else:
            self._handle = self._lib.bga_hand_create()

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and self._lib:
            try:
                self._lib.bga_hand_destroy(self._handle)
            except Exception:
                pass

    def ToString(self):
        return _read_string(self._lib.bga_hand_to_string(self._handle))

    def __str__(self):
        return self.ToString()

    @property
    def Count(self):
        return self._lib.bga_hand_count(self._handle)

    def Union(self, other):
        new_handle = self._lib.bga_hand_union(self._handle, other._handle)
        return NativeHand(handle=new_handle)

    def Except(self, other):
        new_handle = self._lib.bga_hand_except(self._handle, other._handle)
        return NativeHand(handle=new_handle)

    def Remove(self, card):
        return self._lib.bga_hand_remove(self._handle, card._handle) != 0

    def Add(self, card):
        self._lib.bga_hand_add(self._handle, card._handle)

    def AddRange(self, other):
        self._lib.bga_hand_add_range(self._handle, other._handle)

    def Any(self, predicate):
        """Check if any card in the hand matches a predicate.
        For compatibility - iterates by converting to string and checking."""
        # This is used in PIMC code to check suit presence
        # We implement it by checking the string representation
        raise NotImplementedError("Use native methods instead")


class NativePlay:
    """Wraps bga_play_* functions with the same interface as pythonnet Play class."""
    def __init__(self, handle=None):
        self._lib = _get_lib()
        if handle is not None:
            self._handle = handle
        else:
            self._handle = self._lib.bga_play_create()

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and self._lib:
            try:
                self._lib.bga_play_destroy(self._handle)
            except Exception:
                pass

    def Clear(self):
        self._lib.bga_play_clear(self._handle)

    def Add(self, card):
        self._lib.bga_play_add(self._handle, card._handle)

    def AddRange(self, other):
        self._lib.bga_play_add_range(self._handle, other._handle)

    @property
    def Count(self):
        return self._lib.bga_play_count(self._handle)

    def ListAsString(self):
        return _read_string(self._lib.bga_play_list_as_string(self._handle))


class NativeCard:
    """Wraps bga_card_* functions with the same interface as pythonnet Card class."""
    def __init__(self, card_str):
        self._lib = _get_lib()
        if isinstance(card_str, str):
            card_str = card_str.encode('utf-8')
        self._handle = self._lib.bga_card_create(card_str)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and self._lib:
            try:
                self._lib.bga_card_destroy(self._handle)
            except Exception:
                pass

    def ToString(self):
        return _read_string(self._lib.bga_card_to_string(self._handle))

    def __str__(self):
        return self.ToString()


class NativeConstraints:
    """Wraps bga_constraints_* functions with the same interface as pythonnet Constraints class."""
    _PROPS = ['MinClubs', 'MaxClubs', 'MinDiamonds', 'MaxDiamonds',
              'MinHearts', 'MaxHearts', 'MinSpades', 'MaxSpades', 'MinHCP', 'MaxHCP']

    def __init__(self, minC=0, maxC=13, minD=0, maxD=13, minH=0, maxH=13, minS=0, maxS=13, minHCP=0, maxHCP=37):
        self._lib = _get_lib()
        self._handle = self._lib.bga_constraints_create(
            minC, maxC, minD, maxD, minH, maxH, minS, maxS, minHCP, maxHCP)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and self._lib:
            try:
                self._lib.bga_constraints_destroy(self._handle)
            except Exception:
                pass

    def ToString(self):
        return _read_string(self._lib.bga_constraints_to_string(self._handle))

    def __str__(self):
        return self.ToString()

    def _get_prop(self, index):
        return self._lib.bga_constraints_get_prop(self._handle, index)

    def _set_prop(self, index, value):
        self._lib.bga_constraints_set_prop(self._handle, index, value)

    @property
    def MinClubs(self): return self._get_prop(0)
    @MinClubs.setter
    def MinClubs(self, v): self._set_prop(0, v)
    @property
    def MaxClubs(self): return self._get_prop(1)
    @MaxClubs.setter
    def MaxClubs(self, v): self._set_prop(1, v)
    @property
    def MinDiamonds(self): return self._get_prop(2)
    @MinDiamonds.setter
    def MinDiamonds(self, v): self._set_prop(2, v)
    @property
    def MaxDiamonds(self): return self._get_prop(3)
    @MaxDiamonds.setter
    def MaxDiamonds(self, v): self._set_prop(3, v)
    @property
    def MinHearts(self): return self._get_prop(4)
    @MinHearts.setter
    def MinHearts(self, v): self._set_prop(4, v)
    @property
    def MaxHearts(self): return self._get_prop(5)
    @MaxHearts.setter
    def MaxHearts(self, v): self._set_prop(5, v)
    @property
    def MinSpades(self): return self._get_prop(6)
    @MinSpades.setter
    def MinSpades(self, v): self._set_prop(6, v)
    @property
    def MaxSpades(self): return self._get_prop(7)
    @MaxSpades.setter
    def MaxSpades(self, v): self._set_prop(7, v)
    @property
    def MinHCP(self): return self._get_prop(8)
    @MinHCP.setter
    def MinHCP(self, v): self._set_prop(8, v)
    @property
    def MaxHCP(self): return self._get_prop(9)
    @MaxHCP.setter
    def MaxHCP(self, v): self._set_prop(9, v)


class NativeExtensions:
    """Wraps bga_hand_parse - mimics Extensions.Parse()."""
    @staticmethod
    def Parse(pbn):
        lib = _get_lib()
        if isinstance(pbn, str):
            pbn = pbn.encode('utf-8')
        handle = lib.bga_hand_parse(pbn)
        return NativeHand(handle=handle)


class _TrumpEnum:
    """Mimics Macros.Trump enum values."""
    Club = 0
    Diamond = 1
    Heart = 2
    Spade = 3
    No = 4


class _PlayerEnum:
    """Mimics Macros.Player enum values."""
    North = 0
    East = 1
    South = 2
    West = 3


class NativeMacros:
    """Mimics the Macros class with Trump and Player enums."""
    Trump = _TrumpEnum()
    Player = _PlayerEnum()


def is_available():
    """Check if the native BGADLL library is available for this platform."""
    return os.path.isfile(_lib_path)
