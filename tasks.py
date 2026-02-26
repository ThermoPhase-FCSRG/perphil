import glob
import os
from pathlib import Path
import shlex
import shutil
import re
import sys
import platform
from invoke import task
from invoke.context import Context
from invoke.exceptions import Exit, ThreadException
from rich import print

_PACKAGE_NAME = "perphil"
_HOST_SYSTEM = platform.system()
_SUPPORTED_SYSTEMS = (
    "Linux",
    "Darwin",
)

VENV_DIR = ".venv"


def _task_screen_log(message: str, bold: bool = True, color: str = "blue") -> None:
    """
    Convenient function to display a message on terminal during task executions.
    """
    rich_delimiters = f"bold {color}" if bold else f"{color}"
    print(f"[{rich_delimiters}]{message}[/{rich_delimiters}]")


def _run(c: Context, command: str, pty: bool = False, **kwargs):
    """
    Wrapper around invoke Context.run with optional PTY fallback.

    Some environments (CI containers, restricted shells) may not provide PTY
    devices. In that case, retry non-interactively so tasks remain usable.
    """
    if not pty:
        return c.run(command, **kwargs)
    try:
        return c.run(command, pty=True, **kwargs)
    except (OSError, ThreadException) as exc:
        # Handle both direct OSError and ThreadException wrapping OSError
        is_pty_error = False
        if isinstance(exc, OSError):
            # Errno 5 (EIO) often indicates PTY problems (input/output error)
            is_pty_error = "pty" in str(exc).lower() or exc.errno == 5
        elif isinstance(exc, ThreadException):
            # Check if the ThreadException wraps an OSError with errno 5
            exc_str = str(exc).lower()
            if "keyboardinterrupt" in exc_str:
                raise
            is_pty_error = "pty" in exc_str or "errno 5" in exc_str or "entrada/saída" in exc_str

        if not is_pty_error:
            raise
        _task_screen_log(
            "PTY unavailable for this command; retrying without PTY.",
            color="yellow",
            bold=False,
        )
        return c.run(command, pty=False, **kwargs)


def _platform_sanity_check() -> None:
    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise Exit(f"The code is running on unsupported operating system: {host_system}", code=1)


def _venv_activate_prefix() -> str:
    """
    Return the shell prefix to activate our venv.
    Any c.run(...) should be prefixed by this (except for install_system_packages).
    """
    return f"source {VENV_DIR}/bin/activate && "


def _runtime_cache_exports(disable_loopy_cache: bool = False) -> str:
    """
    Return shell exports for runtime cache configuration.

    We keep runtime caches inside the repository to avoid permission issues
    with user-level cache directories in constrained/CI environments.
    """
    cache_root = Path(".cache/perphil-runtime/xdg").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    exports = [f"XDG_CACHE_HOME={shlex.quote(str(cache_root))}"]
    if disable_loopy_cache:
        exports.append("LOOPY_NO_CACHE=1")
    return " ".join(exports)


def _pip_cache_exports() -> str:
    """
    Return shell exports for pip cache directory.
    """
    pip_cache = Path(".cache/pip").resolve()
    pip_cache.mkdir(parents=True, exist_ok=True)
    return f"PIP_CACHE_DIR={shlex.quote(str(pip_cache))}"


def _sanitize_petsc_optflags_for_macos(cfg_flags: list[str]) -> list[str]:
    """
    Remove CPU-specific native tuning flags from PETSc OPTFLAGS on macOS.

    Rationale:
    - Firedrake emits -march=native/-mtune=native in C/C++ OPTFLAGS.
    - During pip builds (e.g. petsc4py), Python may request universal2 extension
      builds on macOS, adding both -arch arm64 and -arch x86_64.
    - The host-native CPU flag (e.g. apple-m3) then leaks into the x86_64 compile
      slice and fails with "unknown target CPU".
    """
    if platform.system() != "Darwin":
        return cfg_flags

    cleaned: list[str] = []
    native_flag_pattern = re.compile(r"^-m(?:arch|tune)=native$")
    for flag in cfg_flags:
        if not (
            flag.startswith("--COPTFLAGS=")
            or flag.startswith("--CXXOPTFLAGS=")
            or flag.startswith("--FOPTFLAGS=")
        ):
            cleaned.append(flag)
            continue

        opt_name, opt_value = flag.split("=", 1)
        parts = [part for part in shlex.split(opt_value) if not native_flag_pattern.match(part)]
        cleaned.append(f"{opt_name}={' '.join(parts)}")
    return cleaned


def _find_working_fortran_compiler(c: Context) -> tuple[str, str]:
    """
    Return (FC, OMPI_FC) suitable for build systems that require explicit Fortran.

    On macOS/Homebrew OpenMPI, using mpi wrappers can require OMPI_FC to point to
    the actual gfortran executable.
    """
    if platform.system() == "Darwin":
        fc_candidates = ["mpifort", "mpif90", "gfortran"]
    else:
        fc_candidates = ["mpifort", "mpif90", "gfortran"]

    for pattern in (
        "/opt/homebrew/opt/gcc/bin/gfortran-*",
        "/usr/local/opt/gcc/bin/gfortran-*",
    ):
        fc_candidates.extend(sorted(glob.glob(pattern), reverse=True))
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not path_dir:
            continue
        fc_candidates.extend(sorted(glob.glob(os.path.join(path_dir, "gfortran-*")), reverse=True))

    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for candidate in fc_candidates:
        if candidate and candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)

    gfortran_fallback = ""
    for candidate in ordered_candidates:
        candidate_path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if not candidate_path:
            continue
        base = os.path.basename(candidate_path)
        if base != "gfortran" and not base.startswith("gfortran-"):
            continue
        probe = c.run(f"{shlex.quote(candidate_path)} --version", hide=True, warn=True)
        if probe is not None and not probe.failed:
            gfortran_fallback = candidate_path
            break

    for candidate in ordered_candidates:
        candidate_path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if not candidate_path:
            continue
        probe = c.run(f"{shlex.quote(candidate_path)} --version", hide=True, warn=True)
        if probe is not None and not probe.failed:
            ompi_fc = gfortran_fallback if platform.system() == "Darwin" else ""
            return candidate_path, ompi_fc

    raise Exit(
        "No working Fortran compiler wrapper was found. "
        "Please ensure OpenMPI Fortran wrappers (mpifort/mpif90) or gfortran are available."
    )


def _find_working_bison(c: Context) -> str:
    """
    Return a GNU Bison executable path with version >= 3, or empty string if unavailable.

    On macOS this prefers Homebrew's keg-only bison path when present.
    """
    candidates: list[str] = []
    env_bison = os.environ.get("BISON", "").strip()
    if env_bison:
        candidates.append(env_bison)
    candidates.extend(
        [
            "bison",
            "/opt/homebrew/opt/bison/bin/bison",
            "/usr/local/opt/bison/bin/bison",
        ]
    )

    brew_prefix = c.run("brew --prefix bison", hide=True, warn=True)
    if brew_prefix is not None and not brew_prefix.failed:
        prefix = (brew_prefix.stdout or "").strip()
        if prefix:
            candidates.append(os.path.join(prefix, "bin", "bison"))

    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)

    for candidate in ordered_candidates:
        candidate_path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if not candidate_path:
            continue

        probe = c.run(f"{shlex.quote(candidate_path)} --version", hide=True, warn=True)
        if probe is None or probe.failed:
            continue

        first_line = (probe.stdout or "").splitlines()
        version_line = first_line[0] if first_line else ""
        match = re.search(r"GNU Bison\)?\s+(\d+)\.(\d+)", version_line)
        if not match:
            continue
        major, minor = int(match.group(1)), int(match.group(2))
        if (major, minor) >= (3, 0):
            return candidate_path

    return ""


def _ensure_venv_exists() -> None:
    if not Path(VENV_DIR, "bin", "activate").is_file():
        raise Exit(
            f"Virtualenv '{VENV_DIR}/' not found. Run 'inv create-venv' (or setup-firedrake) first."
        )


@task(
    help={
        "python": "Python interpreter to use for the venv (e.g., python3.12). Defaults to the active interpreter (sys.executable).",
        "force": "Recreate the virtualenv if it already exists.",
    }
)
def create_venv(c: Context, python: str | None = None, force: bool = False) -> None:
    """
    Create a Python 3.10+ virtualenv in ./.venv if it does not already exist.
    """
    _platform_sanity_check()

    need_recreate = force
    venv_python = Path(VENV_DIR, "bin", "python")
    if os.path.isdir(VENV_DIR) and not force:
        if not venv_python.is_file():
            _task_screen_log(
                f"Virtualenv exists at '{VENV_DIR}/' but is incomplete; recreating.",
                color="yellow",
            )
            need_recreate = True
        else:
            venv_ver = c.run(
                f"{shlex.quote(str(venv_python))} -c \"import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))\"",
                hide=True,
                warn=True,
            )
            if venv_ver is None or getattr(venv_ver, "failed", False):
                _task_screen_log(
                    f"Could not query Python version from '{venv_python}'; recreating virtualenv.",
                    color="yellow",
                )
                need_recreate = True
            else:
                venv_major_minor = (getattr(venv_ver, "stdout", "") or "").strip()
                try:
                    vmaj, vmin = [int(x) for x in venv_major_minor.split(".")[:2]]
                except Exception:
                    vmaj, vmin = (0, 0)
                if (vmaj, vmin) < (3, 10) or (vmaj, vmin) >= (3, 13):
                    _task_screen_log(
                        f"Existing virtualenv uses Python {vmaj}.{vmin}; recreating with Python 3.11/3.12 for Firedrake compatibility.",
                        color="yellow",
                    )
                    need_recreate = True
                else:
                    # Self-heal toolchain in existing env (covers missing setuptools.build_meta errors)
                    c.run(
                        f"{shlex.quote(str(venv_python))} -m ensurepip --upgrade",
                        hide=True,
                        warn=True,
                    )
                    _run(
                        c,
                        f"{_venv_activate_prefix()} {_pip_cache_exports()} python -m pip install --upgrade pip setuptools wheel",
                        pty=True,
                    )
                    print(f"Virtualenv already exists at '{VENV_DIR}/'")
                    return

    # Resolve default interpreter if not provided
    if not python:
        python = sys.executable or "python"

    # Check selected Python interpreter version
    py_ver = c.run(
        f"{python} -c \"import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))\"",
        hide=True,
        warn=True,
    )
    # Guard against a None result and check .failed safely to satisfy type checkers
    if py_ver is None or getattr(py_ver, "failed", False):
        raise Exit(f"Failed to execute '{python}'. Is it installed and on PATH?")
    major_minor = (getattr(py_ver, "stdout", "") or "").strip()
    try:
        maj, minr = [int(x) for x in major_minor.split(".")[:2]]
    except Exception:
        raise Exit(f"Could not parse Python version from '{python}': '{major_minor}'")
    if (maj, minr) < (3, 10):
        raise Exit(f"Python 3.10+ is required (found {maj}.{minr}) for Firedrake.")
    # Firedrake 2025.10.0 does not provide wheels for some deps (e.g., VTK) on Python >=3.13
    # Recommend using Python 3.11 or 3.12
    if (maj, minr) >= (3, 13):
        # Attempt automatic fallback to a supported interpreter
        candidates = ["python3.12", "python3.11"]
        for cand in candidates:
            if shutil.which(cand):
                _task_screen_log(
                    f"Python {maj}.{minr} detected. Auto-selecting '{cand}' for the venv to ensure Firedrake wheels.",
                    color="yellow",
                )
                python = cand
                py_ver = c.run(
                    f"{python} -c \"import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))\"",
                    hide=True,
                    warn=True,
                )
                if py_ver is None or getattr(py_ver, "failed", False):
                    continue
                major_minor = (getattr(py_ver, "stdout", "") or "").strip()
                try:
                    maj, minr = [int(x) for x in major_minor.split(".")[:2]]
                except Exception:
                    continue
                if (maj, minr) < (3, 13):
                    break
        else:
            raise Exit(
                f"Python {maj}.{minr} detected. Firedrake 2025.10.0 currently lacks prebuilt wheels (e.g., VTK) for >=3.13. "
                "Please create the venv with Python 3.11 or 3.12 (e.g., --python python3.12)."
            )

    if os.path.isdir(VENV_DIR) and need_recreate:
        _task_screen_log(f"Removing existing virtualenv at '{VENV_DIR}/' …", color="yellow")
        shutil.rmtree(VENV_DIR)

    _task_screen_log(f"Creating virtualenv in '{VENV_DIR}/' …")
    _run(c, f"{python} -m venv {VENV_DIR}", pty=True)
    _run(
        c,
        f"{_venv_activate_prefix()} {_pip_cache_exports()} python -m pip install --upgrade pip setuptools wheel",
        pty=True,
    )
    _task_screen_log("✔ Virtualenv created.", color="yellow")


@task(pre=[create_venv])
def install_deps(c):
    """
    Install Python dependencies from pyproject.toml into the venv.
    """
    _task_screen_log("Installing Python dependencies for perphil …")
    _run(
        c,
        f"{_venv_activate_prefix()} {_pip_cache_exports()} python -m pip install --no-build-isolation -e '.[dev]'",
        pty=True,
    )
    _task_screen_log("✔ Python-level dependencies installed.", color="yellow")


@task(
    pre=[install_deps],
    help={
        "ref": "Firedrake release tag to pin (e.g., 2025.10.0). Defaults to 'latest' or FIREDRAKE_REF env.",
        "force": "Redownload even if 'firedrake-configure' already exists.",
    },
)
def download_firedrake_configure(c: Context, ref: str = "", force: bool = False) -> None:
    """
    Ensure a usable 'firedrake-configure' script is present in the repo root.

    You can pin a specific Firedrake release by providing --ref=<tag> (e.g., 2025.10.0)
    or setting FIREDRAKE_REF in the environment. When omitted, we use the latest
    release, falling back to the 'main' branch script.
    """
    # Detect requested reference (CLI arg takes precedence over env)
    requested_ref = (ref or os.environ.get("FIREDRAKE_REF", "")).strip()
    requested_is_latest = not requested_ref or requested_ref.lower() == "latest"

    # 0) If file already exists and not forcing, keep it unless a specific ref was requested.
    if os.path.isfile("firedrake-configure") and not force and requested_is_latest:
        _task_screen_log("Found existing firedrake-configure; skipping download.", color="green")
        # Ensure it's executable
        c.run("chmod +x firedrake-configure", warn=True)
        return
    if os.path.isfile("firedrake-configure") and not force and not requested_is_latest:
        _task_screen_log(
            f"Explicit ref '{requested_ref}' requested; refreshing firedrake-configure.",
            color="yellow",
        )

    # Build curl headers (GitHub API can be picky/rate-limited)
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = "-H 'User-Agent: perphil-ci' -H 'Accept: application/vnd.github+json'"
    if gh_token:
        headers += f" -H 'Authorization: Bearer {gh_token}' -H 'X-GitHub-Api-Version: 2022-11-28'"

    def _download_from_ref(tag: str) -> bool:
        _task_screen_log(f"Attempting firedrake-configure@{tag} …")
        raw_url = (
            "https://raw.githubusercontent.com/"
            f"firedrakeproject/firedrake/{tag}/scripts/firedrake-configure"
        )
        dl = c.run(f"curl -fsSL {raw_url} -o firedrake-configure", warn=True, echo=True)
        if dl is not None and not dl.failed and os.path.isfile("firedrake-configure"):
            c.run("chmod +x firedrake-configure", warn=True)
            _task_screen_log(f"✔ downloaded firedrake-configure@{tag}", color="yellow")
            return True
        return False

    # 1) If user requested a specific tag, try that first
    if requested_ref and requested_ref.lower() != "latest":
        if _download_from_ref(requested_ref):
            return
        raise Exit(
            f"Failed to download firedrake-configure for tag '{requested_ref}'. "
            "Please verify the tag exists: https://github.com/firedrakeproject/firedrake/releases"
        )

    # 2) Otherwise, try GitHub API for the latest tag
    _task_screen_log("Looking up the latest Firedrake release…")
    cmd = (
        f"curl -s {headers} https://api.github.com/repos/firedrakeproject/firedrake/releases/latest "
        "| grep -E '\"tag_name\"' | cut -d '\"' -f 4"
    )
    result = c.run(cmd, hide=True, warn=True)
    # Guard against c.run returning None or a failed result before accessing stdout
    if result is None or getattr(result, "failed", False):
        latest_tag = ""
    else:
        latest_tag = (getattr(result, "stdout", "") or "").strip()
    if latest_tag and _download_from_ref(latest_tag):
        return

    # 3) Fallback: try 'main' branch if latest release lookup/download failed
    _task_screen_log("Falling back to firedrake 'main' branch script …", color="yellow")
    fallback_url = (
        "https://raw.githubusercontent.com/"
        "firedrakeproject/firedrake/main/scripts/firedrake-configure"
    )
    dl_fb = c.run(f"curl -fsSL {fallback_url} -o firedrake-configure", warn=True, echo=True)
    # c.run may return None in some contexts; check for None and use getattr to safely access .failed
    if (
        dl_fb is not None
        and not getattr(dl_fb, "failed", False)
        and os.path.isfile("firedrake-configure")
    ):
        c.run("chmod +x firedrake-configure", warn=True)
        _task_screen_log("✔ downloaded firedrake-configure@main", color="yellow")
        return

    # 4) If still nothing, give a clear error with next-step guidance
    raise Exit(
        "Failed to obtain firedrake-configure (explicit ref, latest lookup, and fallback all failed). "
        "You can: (a) commit a known-good 'firedrake-configure' to the repo root, or "
        "(b) re-run with network access and a valid GITHUB_TOKEN."
    )


@task(pre=[download_firedrake_configure])
def install_system_packages(c):
    """
    Install all system-level dependencies that firedrake-configure recommends,
    plus ensure OpenMPI development headers are present on Linux.
    If a package is already installed, skip it.
    """
    _ensure_venv_exists()
    system = platform.system()
    prefix = _venv_activate_prefix()
    # 1) ask firedrake-configure which packages it wants
    result = c.run(
        f"{prefix} python firedrake-configure --show-system-packages",
        hide=True,
        warn=True,
    )
    if result is None or getattr(result, "failed", False):
        raise Exit("Failed to query `firedrake-configure --show-system-packages`")

    base_pkgs = (getattr(result, "stdout", "") or "").strip().split()
    if system == "Linux":
        # Ensure Fortran compiler is available for MPI Fortran support
        all_pkgs = base_pkgs + ["libopenmpi-dev", "openmpi-bin", "gfortran"]

        missing = []
        for pkg in all_pkgs:
            # Check via dpkg-query if pkg is installed
            check = c.run(f"dpkg-query -W -f='${{Status}}' {pkg}", hide=True, warn=True)
            # dpkg-query -W will succeed only if the package is installed.
            # Even if it succeeds, we still check for “install ok installed” in stdout.
            if check.failed or "install ok installed" not in check.stdout.lower():
                missing.append(pkg)

        if missing:
            _task_screen_log("Detected Linux. Installing missing system packages via apt …")
            # Update and install only the missing ones
            pkgs_str = " ".join(missing)
            _run(
                c,
                f'sudo sh -c "apt update && apt install -y {pkgs_str}"',
                pty=True,
            )
            _task_screen_log(f"✔ Installed: {pkgs_str}", color="yellow")
        else:
            _task_screen_log("✔ All system packages are already installed.", color="green")

    elif system == "Darwin":
        # macOS needs gcc (which provides gfortran) for OpenMPI to have Fortran support
        # Also install modern bison (keg-only via Homebrew) so PETSc does not have
        # to build bison from source, which can stall on newer macOS releases.
        all_pkgs = list(dict.fromkeys(base_pkgs + ["gcc", "bison"]))

        missing = []
        for pkg in all_pkgs:
            # Check via `brew list --versions <pkg>` whether pkg is present
            check = c.run(f"brew list --versions {pkg}", hide=True, warn=True)
            # If `brew list --versions` returns an empty string or fails, the pkg is not installed
            if check.failed or not check.stdout.strip():
                missing.append(pkg)

        if missing:
            _task_screen_log("Detected macOS. Installing missing packages via brew …")
            # First update Homebrew
            c.run("brew update", echo=True)
            pkgs_str = " ".join(missing)
            _run(
                c,
                f"brew install {pkgs_str}",
                pty=True,
            )
            _task_screen_log(f"✔ Installed: {pkgs_str}", color="yellow")
        else:
            _task_screen_log("✔ All Homebrew packages are already installed.", color="green")

    else:
        raise Exit(f"Unsupported OS: {system}. Please install system packages manually.")


@task(pre=[install_system_packages])
def install_petsc(c):
    """
    Clone and build PETSc using exactly the configure options that Firedrake wants.
    """
    prefix = _venv_activate_prefix()
    prefix_down = f"source ../{VENV_DIR}/bin/activate && "

    _task_screen_log("Installing PETSc")

    print("Determining PETSc version from firedrake-configure …")
    result = c.run(
        f"{prefix} python firedrake-configure --show-petsc-version",
        hide=True,
        warn=True,
        echo=True,
        pty=False,
    )
    petsc_version = result.stdout.strip()
    if not petsc_version:
        raise Exit("Could not determine PETSc version from firedrake-configure.")

    petsc_repo = "https://gitlab.com/petsc/petsc.git"
    petsc_dir = f"petsc-{petsc_version}"
    abs_petsc = os.path.abspath(petsc_dir)
    arch = "arch-firedrake-default"
    if not os.path.isdir(petsc_dir):
        print(f"Cloning PETSc {petsc_version} …")
        c.run(f"git clone --branch {petsc_version} {petsc_repo} {petsc_dir}", echo=True)

    # Short-circuit: if cached PETSc build is present, skip configure/build
    arch_lib_dir = os.path.join(abs_petsc, arch, "lib")
    if os.path.isdir(arch_lib_dir):
        try:
            lib_files = [name for name in os.listdir(arch_lib_dir) if name.startswith("libpetsc")]
        except FileNotFoundError:
            lib_files = []
        if lib_files:
            _task_screen_log(
                f"✔ Found cached PETSc build at '{arch_lib_dir}', skipping configure/build.",
                color="green",
            )
            os.environ["PETSC_DIR"] = abs_petsc
            os.environ["PETSC_ARCH"] = arch

            # Create a symlink 'petsc' -> petsc_dir so that firedrake-configure --show-env works correctly
            if not os.path.exists("petsc"):
                os.symlink(petsc_dir, "petsc")
                _task_screen_log(f"✔ Created symlink 'petsc' -> '{petsc_dir}'", color="yellow")
            elif os.path.islink("petsc") and os.readlink("petsc") != petsc_dir:
                os.remove("petsc")
                os.symlink(petsc_dir, "petsc")
                _task_screen_log(f"✔ Updated symlink 'petsc' -> '{petsc_dir}'", color="yellow")

            print(f"→ Exported PETSC_DIR={abs_petsc}")
            print(f"→ Exported PETSC_ARCH={arch}")
            return

    print("Gathering PETSc configure flags …")
    cfg_stdout = c.run(
        f"{prefix} python firedrake-configure --show-petsc-configure-options",
        hide=True,
        warn=True,
        echo=True,
        pty=False,
    ).stdout.strip()
    cfg_flags = shlex.split(cfg_stdout)
    if not cfg_flags:
        raise Exit("No PETSc configure options found.")
    cfg_flags = _sanitize_petsc_optflags_for_macos(cfg_flags)

    # Prefer system GNU Bison on macOS instead of PETSc's source build path.
    # This avoids long/hanging "Running configure on BISON" failures.
    download_bison_requested = any(
        flag == "--download-bison" or flag.startswith("--download-bison=") for flag in cfg_flags
    )
    if platform.system() == "Darwin" and download_bison_requested:
        bison_exec = _find_working_bison(c)
        if not bison_exec:
            raise Exit(
                "PETSc requested '--download-bison', but no GNU Bison >= 3 was found. "
                "Install it with 'brew install bison' (or run 'inv install-system-packages') "
                "and retry."
            )
        cfg_flags = [
            flag
            for flag in cfg_flags
            if not (
                flag == "--download-bison"
                or flag.startswith("--download-bison=")
                or flag.startswith("--with-bison-exec=")
            )
        ]
        cfg_flags.append(f"--with-bison-exec={bison_exec}")
        _task_screen_log(
            f"Using system Bison at '{bison_exec}' instead of '--download-bison'.",
            color="yellow",
        )

    # ------------------------------------------------------------------
    # Fix ScaLAPACK + Fortran mismatch on macOS:
    # Firedrake currently emits '--with-fortran-bindings=0' while also
    # requesting ScaLAPACK (via --with-scalapack-dir=… or --download-scalapack).
    # PETSc's configure refuses ScaLAPACK when Fortran bindings are disabled,
    # producing: "Cannot use ScaLAPACK without Fortran".
    # We patch the flag list here: if any scalapack-related flag is present
    # we force '--with-fortran-bindings=1' and drop any disabling flag.
    # ------------------------------------------------------------------
    has_scalapack = any("scalapack" in f for f in cfg_flags)
    if has_scalapack:
        new_flags: list[str] = []
        for f in cfg_flags:
            if f.startswith("--with-fortran-bindings="):
                # Replace disable with enable
                continue
            new_flags.append(f)
        # Append enabled Fortran bindings explicitly
        new_flags.append("--with-fortran-bindings=1")
        cfg_flags = new_flags
        _task_screen_log(
            "Adjusted PETSc flags: enabling Fortran bindings for ScaLAPACK.",
            color="yellow",
        )

    fortran_requested = any(f == "--with-fortran-bindings=1" for f in cfg_flags)

    print("Configuring PETSc …")
    with c.cd(petsc_dir):
        # Join configure flags into a single invocation
        cfg_joined = " ".join(shlex.quote(flag) for flag in cfg_flags)
        # Determine compilers from environment and strip any ccache prefix
        raw_cc = os.environ.get("CC", "mpicc").strip()
        cc = raw_cc.split(" ", 1)[-1] if " " in raw_cc else raw_cc
        raw_cxx = os.environ.get("CXX", "mpicxx").strip()
        cxx = raw_cxx.split(" ", 1)[-1] if " " in raw_cxx else raw_cxx
        compiler_args = [f"CC={cc}", f"CXX={cxx}"]
        ompi_fc_env = ""
        if fortran_requested:
            raw_fc = os.environ.get("FC", "mpif90").strip()
            requested_fc = raw_fc.split(" ", 1)[-1] if " " in raw_fc else raw_fc
            if platform.system() == "Darwin":
                fc_candidates = ["mpifort", requested_fc, "mpif90", "gfortran"]
            else:
                fc_candidates = [requested_fc, "mpifort", "mpif90", "gfortran"]

            # Homebrew may expose only versioned Fortran compiler names (e.g. gfortran-15).
            for pattern in (
                "/opt/homebrew/opt/gcc/bin/gfortran-*",
                "/usr/local/opt/gcc/bin/gfortran-*",
            ):
                fc_candidates.extend(sorted(glob.glob(pattern), reverse=True))
            for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                if not path_dir:
                    continue
                fc_candidates.extend(
                    sorted(glob.glob(os.path.join(path_dir, "gfortran-*")), reverse=True)
                )

            seen: set[str] = set()
            ordered_candidates: list[str] = []
            for candidate in fc_candidates:
                if candidate and candidate not in seen:
                    ordered_candidates.append(candidate)
                    seen.add(candidate)

            # Keep a working gfortran fallback for OpenMPI wrappers that require OMPI_FC.
            gfortran_fallback = ""
            for candidate in ordered_candidates:
                candidate_path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
                if not candidate_path:
                    continue
                base = os.path.basename(candidate_path)
                if base != "gfortran" and not base.startswith("gfortran-"):
                    continue
                probe = c.run(f"{shlex.quote(candidate_path)} --version", hide=True, warn=True)
                if not probe.failed:
                    gfortran_fallback = candidate_path
                    break

            fc = ""
            for candidate in ordered_candidates:
                candidate_path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
                if not candidate_path:
                    continue
                probe = c.run(f"{shlex.quote(candidate_path)} --version", hide=True, warn=True)
                if not probe.failed:
                    fc = candidate_path
                    break

                if (
                    platform.system() == "Darwin"
                    and os.path.basename(candidate_path) in {"mpifort", "mpif90"}
                    and gfortran_fallback
                ):
                    probe_ompi_fc = c.run(
                        f"OMPI_FC={shlex.quote(gfortran_fallback)} {shlex.quote(candidate_path)} --version",
                        hide=True,
                        warn=True,
                    )
                    if not probe_ompi_fc.failed:
                        fc = candidate_path
                        ompi_fc_env = gfortran_fallback
                        break

            if not fc:
                # On macOS, if Fortran is unavailable and ScaLAPACK was requested,
                # disable ScaLAPACK and Fortran bindings as a fallback instead of failing.
                if platform.system() == "Darwin" and has_scalapack:
                    _task_screen_log(
                        "No Fortran compiler found on macOS; disabling ScaLAPACK and Fortran bindings.",
                        color="yellow",
                    )
                    # Rebuild cfg_flags: remove ScaLAPACK and Fortran bindings
                    cfg_flags = [
                        f
                        for f in cfg_flags
                        if not any(keyword in f for keyword in ["scalapack", "fortran-bindings"])
                    ]
                    # Explicitly disable Fortran bindings
                    cfg_flags.append("--with-fortran-bindings=0")
                    cfg_joined = " ".join(shlex.quote(flag) for flag in cfg_flags)
                else:
                    raise Exit(
                        "PETSc requested Fortran bindings, but no working Fortran compiler wrapper was found. "
                        "Tried: "
                        + ", ".join(ordered_candidates)
                        + ". Please ensure OpenMPI Fortran wrappers are installed (e.g. mpifort) "
                        "or that a working gfortran/gfortran-<version> is available on PATH."
                    )
            else:
                compiler_args.append(f"FC={fc}")
                if ompi_fc_env:
                    _task_screen_log(
                        f"Using OMPI_FC fallback with {ompi_fc_env} for OpenMPI Fortran wrappers.",
                        color="yellow",
                    )

        # Build configure command with compilers as arguments (not env vars)
        # IMPORTANT: PETSc's configure script ignores CC/CXX/FC environment variables and
        # requires them to be passed as command-line arguments to avoid warnings.
        # Ensure PETSC_DIR/PETSC_ARCH from the user's environment do not confuse configure.
        # We explicitly point PETSC_DIR at the cloned source dir and clear PETSC_ARCH here.
        # The configure options already include the desired PETSC_ARCH.
        compiler_joined = " ".join(shlex.quote(arg) for arg in compiler_args)
        ompi_fc_assignment = f" OMPI_FC={shlex.quote(ompi_fc_env)}" if ompi_fc_env else ""
        cmd = (
            f"{prefix_down} env -u PETSC_ARCH -u CC -u CXX -u FC{ompi_fc_assignment} PETSC_DIR=$PWD ./configure "
            f"{compiler_joined} {cfg_joined}"
        )
        _run(c, cmd, echo=True, pty=False)
        # Build PETSc
        print("Building PETSc (this may take a long time) …")
        make_prefix = f"OMPI_FC={shlex.quote(ompi_fc_env)} " if ompi_fc_env else ""
        c.run(f"{make_prefix}make PETSC_DIR={abs_petsc} PETSC_ARCH={arch} all", echo=True)
        # Check PETSc installation
        print("Checking PETSc installation…")
        try:
            c.run(f"{make_prefix}make PETSC_DIR={abs_petsc} PETSC_ARCH={arch} check", echo=True)
        except Exception as e:
            _task_screen_log("PETSc compiled, but failed in the checks!", color="red")
            _task_screen_log(f"PETSc check output: \n{e}", color="red")

    # Export PETSC_DIR and PETSC_ARCH so that subsequent steps see them:
    os.environ["PETSC_DIR"] = abs_petsc
    os.environ["PETSC_ARCH"] = arch

    # Create a symlink 'petsc' -> petsc_dir so that firedrake-configure --show-env works correctly
    # firedrake-configure assumes PETSC_DIR is os.getcwd()/petsc if not explicitly set,
    # but --show-env logic might depend on finding the dir.
    if os.path.lexists("petsc"):
        os.remove("petsc")
    os.symlink(petsc_dir, "petsc")
    _task_screen_log(f"✔ Created symlink 'petsc' -> '{petsc_dir}'", color="yellow")

    print(f"→ Exported PETSC_DIR={abs_petsc}")
    print(f"→ Exported PETSC_ARCH={arch}")
    _task_screen_log(
        f"✔ PETSc built.  PETSC_DIR: '{abs_petsc}', PETSC_ARCH: '{arch}'.", color="yellow"
    )


def _install_firedrake_python_package(c: Context, ref: str = "") -> None:
    """
    Install the Firedrake Python package (with [check]) inside the venv via pip.
    We use the environment variables from firedrake-configure to ensure it links against our custom PETSc.
    """
    prefix = _venv_activate_prefix()
    _task_screen_log("Installing Firedrake in the virtualenv …")

    # Pin build-time tooling in constraints.txt per Firedrake docs:
    # https://www.firedrakeproject.org/install.html
    # - setuptools<81 avoids a petsc4py build-time incompatibility:
    #   confpetsc.py calls distutils.util.execute(..., dry_run=...), which
    #   fails with setuptools 81+ where that kwarg is gone.
    # - Cython is needed for building Firedrake from source (not mentioned in docs
    #   but required for package metadata generation)
    Path("constraints.txt").write_text(
        "setuptools<81\nCython\n",
        encoding="utf-8",
    )
    c.run("cat constraints.txt", echo=True)
    # Set PIP_CONSTRAINT as the gentler default.
    # Do NOT set PIP_BUILD_CONSTRAINT as a global environment variable, since it conflicts
    # with --no-build-isolation. Instead, use --build-constraint via pip_constraint_opts
    # only when build isolation is active (checked below).
    os.environ["PIP_CONSTRAINT"] = "constraints.txt"

    requested_ref = (ref or os.environ.get("FIREDRAKE_REF", "")).strip()
    if requested_ref and requested_ref.lower() != "latest":
        firedrake_spec = f"firedrake[check]=={requested_ref}"
        _task_screen_log(f"Pinning firedrake to version {requested_ref}", color="green")
    else:
        firedrake_spec = "firedrake[check]"

    # Configure shell environment exactly as firedrake-configure expects.
    # Use eval with quoting so values containing special chars are handled safely.
    setup_env = f'eval "$({prefix}python firedrake-configure --show-env)"'
    # Explicitly set compiler environment variables for build isolation
    os.environ["CC"] = "mpicc"
    os.environ["CXX"] = "mpicxx"
    extra_build_env: list[str] = []
    if platform.system() == "Darwin":
        # Force arm64 wheel/ext build to avoid universal2 host/target flag clashes.
        extra_build_env.append("ARCHFLAGS='-arch arm64'")
        fc, ompi_fc = _find_working_fortran_compiler(c)
        extra_build_env.append(f"FC={shlex.quote(fc)}")
        if ompi_fc:
            extra_build_env.append(f"OMPI_FC={shlex.quote(ompi_fc)}")
    build_env_prefix = " ".join(extra_build_env + [_pip_cache_exports()]).strip()
    pip_constraint_opts = "-c constraints.txt"
    # --build-constraint is only for build-time isolation; incompatible with --no-build-isolation
    pip_constraint_only = "-c constraints.txt"
    has_build_constraint = c.run(
        f"{prefix} python -m pip help install | grep -q -- '--build-constraint'",
        hide=True,
        warn=True,
    )
    if has_build_constraint is not None and not has_build_constraint.failed:
        pip_constraint_opts += " --build-constraint constraints.txt"

    # Standard install command from documentation:
    # pip install --no-binary h5py 'firedrake[check]'
    install_cmd = (
        f"{prefix}{setup_env} && {build_env_prefix} "
        f"python -m pip install {pip_constraint_opts} --no-binary h5py '{firedrake_spec}'"
    )

    _task_screen_log("Running pip install for Firedrake (standard PyPI install) …")
    res = _run(c, install_cmd, pty=True, echo=True, warn=True)

    # If build isolation selects an incompatible setuptools for petsc4py,
    # retry in the active venv with pinned build tooling.
    if res is None or getattr(res, "failed", False):
        _task_screen_log(
            "Standard install failed. Retrying with pinned build tooling and --no-build-isolation …",
            color="yellow",
        )

        # Constraint file already enforces setuptools<81 for petsc4py compatibility.
        _run(
            c,
            f"{prefix} {_pip_cache_exports()} python -m pip install {pip_constraint_only} wheel",
            pty=True,
            echo=True,
            warn=True,
        )

        # Pre-install key native build dependencies and petsc4py in the active venv.
        # Cython is required for firedrake's setup.py to generate package metadata.
        _run(
            c,
            f"{prefix}{setup_env} && {build_env_prefix} "
            f"python -m pip install {pip_constraint_only} "
            "Cython pkgconfig pybind11 numpy mpi4py petsctools 'rtree>=1.2' libsupermesh",
            pty=True,
            echo=True,
            warn=True,
        )
        _run(
            c,
            f"{prefix}{setup_env} && {build_env_prefix} "
            f"python -m pip install {pip_constraint_only} --no-build-isolation 'petsc4py==3.24.0'",
            pty=True,
            echo=True,
            warn=True,
        )

        retry_cmd = (
            f"{prefix}{setup_env} && {build_env_prefix} "
            f"python -m pip install {pip_constraint_opts} --no-binary h5py '{firedrake_spec}'"
        )
        res = _run(c, retry_cmd, pty=True, echo=True, warn=True)

    # If PyPI failed (e.g. version mismatch with PETSc), try fallbacks
    if res is None or getattr(res, "failed", False):
        _task_screen_log(
            "PyPI install failed. Attempting fallback installation methods…", color="yellow"
        )

        if requested_ref and requested_ref.lower() != "latest":
            # Fallback for pinned version: try GitHub tarball
            tarball_url = (
                "https://github.com/firedrakeproject/firedrake/archive/refs/tags/"
                f"{requested_ref}.tar.gz"
            )
            _task_screen_log(f"Trying GitHub tarball for tag {requested_ref}…", color="yellow")
            cmd_url = (
                f"{prefix}{setup_env} && {build_env_prefix} "
                f"python -m pip install {pip_constraint_opts} --no-binary h5py 'firedrake[check]@{tarball_url}'"
            )
            _run(c, cmd_url, pty=True, echo=True)
        else:
            # Fallback for latest: try GitHub default branch
            _task_screen_log("Trying Firedrake from GitHub default branch…", color="yellow")
            cmd_git = (
                f"{prefix}{setup_env} && {build_env_prefix} "
                f"python -m pip install {pip_constraint_opts} --no-binary h5py "
                "'firedrake[check] @ git+https://github.com/firedrakeproject/firedrake.git'"
            )
            _run(c, cmd_git, pty=True, echo=True)

    # Clean up constraint file
    os.environ.pop("PIP_CONSTRAINT", None)

    print("\nVerifying the installation …")
    # Ensure immutabledict is present (sometimes missed by dependencies)
    res_immut = c.run(f"{prefix} python -c 'import immutabledict'", hide=True, warn=True)
    if res_immut is None or getattr(res_immut, "failed", False):
        _task_screen_log("Installing missing dependency: immutabledict …", color="yellow")
        _run(
            c,
            f"{prefix} {_pip_cache_exports()} python -m pip install immutabledict",
            pty=True,
            echo=True,
        )

    try:
        # Run checks. We unset PETSC_DIR/ARCH to ensure the installed package finds them itself (or via baked-in paths)
        # but firedrake-check might actually need them if the install isn't fully relocatable.
        # However, the standard check command usually works if the venv is active.
        # We'll try running it with the env vars set by firedrake-configure just to be safe,
        # or just rely on the venv.

        # Note: The previous task unset them. Let's try running with the configured env to be safe,
        # as that's how we installed it.
        check_cmd = (
            f"{prefix}{setup_env} && {_runtime_cache_exports(disable_loopy_cache=True)} "
            "OMP_NUM_THREADS=1 firedrake-check"
        )

        _run(c, check_cmd, echo=True, pty=True)
        _task_screen_log("✔ Firedrake installed successfully.", color="green")
    except Exception as e:
        raise Exit(f"Failed to verify Firedrake installation: {e}")


@task(
    help={
        "ref": "Firedrake version to install via pip (e.g., 2025.10.0). Defaults to 'latest' or FIREDRAKE_REF env.",
    },
)
def install_firedrake(c: Context, ref: str = "") -> None:
    """
    End-to-end Firedrake setup honoring an optional pinned release ref.
    """
    # Use task bodies directly to avoid Invoke pre-task recursion with default args.
    create_venv.body(c)
    install_deps.body(c)
    download_firedrake_configure.body(c, ref=ref)
    install_system_packages.body(c)
    install_petsc.body(c)
    _install_firedrake_python_package(c, ref=ref)


@task(
    help={
        "ref": "Firedrake release tag to pin across all steps (e.g., 2025.10.0).",
        "python": "Python interpreter to use for the venv. Defaults to the active interpreter running Invoke.",
        "force": "Recreate venv and redownload firedrake-configure if present.",
    }
)
def setup_firedrake(
    c: Context, ref: str = "", python: str | None = None, force: bool = False
) -> None:
    """
    One-shot setup of Firedrake toolchain honoring a specific release tag.

    This runs, in order: create_venv, install_deps, download_firedrake_configure (pinned),
    install_system_packages, install_petsc, and install_firedrake (pinned).
    """
    # Choose interpreter: if not provided, default to the one running this task.
    chosen_python = python or sys.executable

    # If the chosen interpreter is >=3.13, try to auto-fallback to a known-good Python (3.11/3.12)
    # to avoid missing wheels (e.g., VTK) for Firedrake 2025.10.0.
    try:
        ver = c.run(
            f"{chosen_python} -c \"import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))\"",
            hide=True,
            warn=True,
        )
        # c.run may return None in some contexts; use getattr to safely access stdout
        major_minor = (getattr(ver, "stdout", "") or "").strip()
        maj, minr = [int(x) for x in major_minor.split(".")[:2]]
    except Exception:
        maj, minr = (0, 0)

    if (maj, minr) >= (3, 13):
        candidates: list[str] = []
        if _HOST_SYSTEM == "Darwin":
            # Common Homebrew locations first, then PATH fallbacks
            candidates.extend(
                [
                    "/opt/homebrew/bin/python3.11",
                    "/usr/local/bin/python3.11",
                    "/opt/homebrew/bin/python3.12",
                    "/usr/local/bin/python3.12",
                ]
            )
        # Generic fallbacks for any platform (resolved via PATH)
        candidates.extend(["python3.11", "python3.12"])

        for cand in candidates:
            # Accept either absolute path existing or an executable discoverable on PATH
            if os.path.isabs(cand) and os.path.exists(cand):
                chosen_python = cand
                _task_screen_log(
                    f"Detected Python {maj}.{minr}. Auto-selecting '{cand}' for the venv to ensure Firedrake wheels.",
                    color="yellow",
                )
                break
            if shutil.which(cand):
                chosen_python = cand
                _task_screen_log(
                    f"Detected Python {maj}.{minr}. Auto-selecting '{cand}' for the venv to ensure Firedrake wheels.",
                    color="yellow",
                )
                break

    # Ensure we use the chosen Python and a clean venv if forcing
    # Use task bodies directly to keep this orchestration deterministic.
    create_venv.body(c, python=chosen_python, force=force)
    install_deps.body(c)
    # Ensure we use the same ref for both the configure script and pip install
    download_firedrake_configure.body(c, ref=ref, force=force)
    install_system_packages.body(c)
    install_petsc.body(c)
    _install_firedrake_python_package(c, ref=ref)


@task
def clean(c):
    """
    Remove the virtualenv, PETSc build directory, and any downloaded scripts.
    """
    _task_screen_log("Cleaning installation artifacts")

    # Remove the virtualenv first. This avoids modifying any global Python installation.
    if os.path.isdir(VENV_DIR):
        _task_screen_log(f"Removing '{VENV_DIR}/' …", color="yellow")
        shutil.rmtree(VENV_DIR)

    # Remove any petsc-* directories.
    for petsc_dir in sorted(Path(".").glob("petsc-*")):
        if petsc_dir.is_dir():
            _task_screen_log(f"Removing '{petsc_dir}' …", color="yellow")
            shutil.rmtree(petsc_dir)

    # Remove petsc symlink
    if os.path.islink("petsc"):
        print("Removing 'petsc' symlink …")
        os.remove("petsc")

    # Remove generated helper files, if present.
    for fname in ["firedrake-configure", "constraints.txt"]:
        if os.path.isfile(fname):
            print(f"Removing '{fname}' …")
            os.remove(fname)

    # Remove local runtime cache used by tests/install checks.
    runtime_cache_dir = Path(".cache/perphil-runtime")
    if runtime_cache_dir.exists():
        _task_screen_log(f"Removing '{runtime_cache_dir}' …", color="yellow")
        shutil.rmtree(runtime_cache_dir)

    _task_screen_log("✔ Cleanup complete.", color="yellow")


@task(help={"overwrite": "Reinstall git hooks overwriting the previous installation."})
def hooks(ctx, overwrite=False):
    """
    Configure pre-commit in the local git.
    """
    task_output_message = "Installing pre-commit hooks"
    _task_screen_log(task_output_message)
    base_command = "pre-commit install"

    if overwrite:
        base_command += " --overwrite"

    _task_screen_log(f"Running: {base_command}", color="yellow", bold=False)
    ctx.run(base_command)


@task(
    pre=[hooks],
    help={
        "all_files": "Run git hooks in all files (may take some time)",
        "files": "Run git hooks in a given set of files",
        "verbose": "Run git hooks in verbose mode",
        "from_ref": "(for usage with `--to-ref`) -- this option represents the original ref in a `from_ref...to_ref` diff expression. For `pre-push` hooks, this represents the branch you are pushing to. For `post-checkout` hooks, this represents the branch that was previously checked out.",
        "to_ref": "(for usage with `--from-ref`) -- this option represents the destination ref in a `from_ref...to_ref` diff expression. For `pre-push` hooks, this represents the branch being pushed. For `post-checkout` hooks, this represents the branch that is now checked out.",
    },
)
def run_hooks(ctx, all_files=False, verbose=False, files="", from_ref="", to_ref=""):
    """
    Run all the installed git hooks in all.
    """
    task_output_message = "Run installed git hooks"
    _task_screen_log(task_output_message)
    base_command = "pre-commit run"
    if all_files:
        base_command += " --all-files"

    if verbose:
        base_command += " --verbose"

    if files != "":
        base_command += f" --files '{files}'"

    if from_ref != "":
        base_command += f" --from-ref {from_ref}"

    if to_ref != "":
        base_command += f" --to-ref {to_ref}"

    _task_screen_log(f"Running: {base_command}", color="yellow", bold=False)
    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
    pty_flag = True if host_system != "Windows" else False
    ctx.run(base_command, pty=pty_flag)


@task(
    help={
        "src": "Glob pattern(s) or list of .ipynb paths to pair (default: notebooks/*.ipynb)",
        "dry": "Preview only (no files will be changed)",
    }
)
def pair_ipynbs(ctx, src="notebooks/*.ipynb", dry=False):
    """
    Pair Jupyter notebooks with Python scripts (percent‐format).
    """
    _task_screen_log("Pairing notebooks with Python scripts")

    # Gather files
    if isinstance(src, str):
        raw = glob.glob(src, recursive=True)
    else:
        raw = list(src)
    notebooks = [Path(p) for p in raw if Path(p).suffix == ".ipynb"]
    if not notebooks:
        raise Exit(f"No notebooks found for {src}", 1)

    # Run pairing
    for nb in notebooks:
        print(f"{'Would pair:' if dry else 'Pairing:'} {nb}")
        if not dry:
            cmd = ["jupytext", "--set-formats", "ipynb,py:percent", str(nb)]
            ctx.run(" ".join(shlex.quote(x) for x in cmd), pty=(_HOST_SYSTEM != "Windows"))

    print(f"{len(notebooks)} notebook(s) {'would be paired' if dry else 'paired'}.")


@task
def dev_install(ctx):
    """
    Install perphil in the virtual environment.
    """
    _ensure_venv_exists()
    task_output_message = "Installing perphil in the virtual environment"
    _task_screen_log(task_output_message)
    base_command = (
        f"{_venv_activate_prefix()} {_pip_cache_exports()} "
        'python -m pip install --no-build-isolation -e ".[dev]"'
    )
    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
    pty_flag = True if host_system != "Windows" else False
    ctx.run(base_command, pty=pty_flag)


@task(
    help={
        "numprocess": "Num of processes to run pytest in parallel",
        "verbose": "Run pytest in verbose mode",
        "color": "Colorize pytest output",
        "check_coverage": "Display coverage summary after running the tests",
        "generate_report": "Generate pytest report and save it as a xml file (named pytest.xml)",
        "generate_cov_xml": "Generate coverage report and save it as a xml file (named coverage.xml)",
        "record_output": "Record all the pytest CLI output to pytest-coverage.txt file",
    },
    optional=["numprocess"],
)
def tests(
    ctx,
    numprocess=-1,
    verbose=True,
    color=True,
    check_coverage=False,
    generate_cov_xml=False,
    generate_report=False,
    record_output=False,
):
    """
    Run tests with pytest.
    """
    task_output_message = "Running the tests"
    _task_screen_log(task_output_message)

    base_command = "pytest -ra -q"
    if verbose:
        base_command += " -v"

    if color:
        base_command += " --color=yes"

    if numprocess != 1:
        base_command += " -n"
        if numprocess == -1:
            base_command += " auto"
        elif numprocess > 1:
            base_command += f" {int(numprocess)}"
        else:
            _task_screen_log(
                "Warning: there is no negative number of processes. Setting to 1 (serial).",
                color="yellow",
            )
            base_command += " 1"

    if check_coverage:
        base_command += " --cov=src/perphil"

    if generate_report:
        base_command += " --junitxml=pytest.xml"
        if not check_coverage:
            base_command += " --cov=src/perphil"

    if generate_cov_xml:
        base_command += " --cov-report xml:coverage.xml"

    if generate_report or generate_cov_xml:
        base_command += " --cov-report=term-missing:skip-covered"

    if record_output:
        base_command += " | tee pytest-coverage.txt"

    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
    pty_flag = True if host_system != "Windows" else False
    # Avoid inheriting PETSc vars and force deterministic cache behavior for Firedrake/Loopy.
    full_cmd = (
        f"env -u PETSC_DIR -u PETSC_ARCH {_runtime_cache_exports(disable_loopy_cache=True)} "
        f"{base_command}"
    )
    _task_screen_log(f"Running: {full_cmd}", color="yellow", bold=False)
    ctx.run(full_cmd, pty=pty_flag)


@task(
    help={
        "verbose": "Run pytest in verbose mode",
        "color": "Colorize pytest output",
        "check_coverage": "Display coverage summary table after running the tests",
        "generate_report": "Generate pytest report and save it as a xml file (named pytest.xml)",
        "generate_cov_xml": "Generate coverage report and save it as a xml file (named coverage.xml)",
        "cov_append": "Append coverage output to existing coverage file",
    },
)
def tests_ipynb(
    ctx,
    verbose=True,
    color=True,
    check_coverage=False,
    generate_cov_xml=False,
    generate_report=False,
    cov_append=False,
):
    """
    Placeholder for notebook tests. No notebooks are marked as supported in this repo; runs a no-op unless enabled.
    """
    task_output_message = "Running notebooks as tests (noop)"
    _task_screen_log(task_output_message)

    base_command = "pytest -ra -q"
    if verbose:
        base_command += " -v"
    if color:
        base_command += " --color=yes"
    if check_coverage:
        base_command += " --cov=src/perphil"
    if generate_report:
        base_command += " --junitxml=pytest-ipynb.xml"
        if not check_coverage:
            base_command += " --cov=src/perphil"
    if generate_cov_xml:
        # append to main coverage if requested
        base_command += " --cov-report xml:coverage.xml"
    if generate_report or generate_cov_xml:
        base_command += " --cov-report=term-missing:skip-covered"
    if cov_append:
        base_command += " --cov-append"

    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
    pty_flag = True if host_system != "Windows" else False
    # Avoid inheriting PETSc vars and force deterministic cache behavior for Firedrake/Loopy.
    full_cmd = (
        f"env -u PETSC_DIR -u PETSC_ARCH {_runtime_cache_exports(disable_loopy_cache=True)} "
        f"{base_command}"
    )
    _task_screen_log(f"Running: {full_cmd}", color="yellow", bold=False)
    ctx.run(full_cmd, pty=pty_flag)


@task
def diff_coverage(ctx):
    """
    Run diff-cover to verify if all new/changed lines are covered. Needs coverage.xml present.
    """
    task_output_message = "Check if diff code is covered"
    _task_screen_log(task_output_message)

    base_command = "diff-cover coverage.xml --config-file pyproject.toml"
    _task_screen_log(f"Running: {base_command}", color="yellow", bold=False)

    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
    pty_flag = True if host_system != "Windows" else False
    ctx.run(base_command, pty=pty_flag)


@task(
    help={
        "color": "Display output with colors",
        "pretty": "Enable better and colorful mypy output",
        "verbose": "Run mypy in verbose mode",
        "files": "Files to be checked with mypy",
    }
)
def type_check(ctx, pretty=False, verbose=False, color=True, files=""):
    """
    Run mypy on perphil to check for typing issues.
    """
    task_output_message = "Running typing check on perphil"
    _task_screen_log(task_output_message)

    base_command = "mypy"
    if pretty:
        base_command += " --pretty"
    if verbose:
        base_command += " --verbose"
    if color:
        base_command += " --color-output"
    if files != "":
        base_command += f" {files}"

    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
    pty_flag = True if host_system != "Windows" else False
    _task_screen_log(f"Running: {base_command}", color="yellow", bold=False)
    ctx.run(base_command, pty=pty_flag)


@task(
    help={
        "dry": "Show what would be removed without actually deleting",
    }
)
def dev_clean(ctx, dry=False):
    """
    Remove perphil build/config dirs:
      - perphil.egg-info
      - dist
      - build
      - *_cache
      - site
    """
    patterns = [
        "*.egg-info",
        "dist",
        "build",
        "*_cache",
        "site",
    ]

    # Collect all matching dirs
    exclude_root = ".venv"  # we need to skip changes in .venv
    to_remove = []
    for pat in patterns:
        for d in Path(".").rglob(pat):
            if not d.is_dir():
                continue
            # skip .venv
            if exclude_root in d.parts:
                continue
            to_remove.append(d)

    if not to_remove:
        _task_screen_log("Nothing to clean.", color="yellow")
        return

    for d in to_remove:
        _task_screen_log(f"{'Would remove:' if dry else 'Removing:  '}{d}", color="yellow")
        if not dry:
            shutil.rmtree(d)

    _task_screen_log(
        f"\n{len(to_remove)} director{'y' if len(to_remove) == 1 else 'ies'} {'would be removed' if dry else 'removed'}.",
        color="yellow",
    )
