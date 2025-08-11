import glob
import os
from pathlib import Path
import shlex
import shutil
import sys
import platform
from invoke import task, exceptions, Exit
from rich import print

_PACKAGE_NAME = "dpp-studies"
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


def _platform_sanity_check() -> None:
    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise exceptions.Exit(
            f"The code is running on unsupported operating system: {host_system}", code=1
        )


def _venv_activate_prefix() -> str:
    """
    Return the shell prefix to activate our venv.
    Any c.run(...) should be prefixed by this (except for install_system_packages).
    """
    return f"source {VENV_DIR}/bin/activate && "


@task
def create_venv(c):
    """
    Create a Python 3.10+ virtualenv in ./.venv if it does not already exist.
    """
    _platform_sanity_check()

    if os.path.isdir(VENV_DIR):
        print(f"Virtualenv already exists at '{VENV_DIR}/'")
        return

    ver = sys.version_info
    if ver < (3, 10):
        raise Exit(f"Python 3.10+ is required (found {ver[0]}.{ver[1]}).")

    _task_screen_log(f"Creating virtualenv in '{VENV_DIR}/' …")
    c.run(f"python3 -m venv {VENV_DIR}", pty=True)
    c.run(f"{_venv_activate_prefix()} pip install --upgrade pip setuptools wheel", pty=True)
    _task_screen_log("✔ Virtualenv created.", color="yellow")


@task(pre=[create_venv])
def install_deps(c):
    """
    Install all Python dependencies listed in requirements.txt into the venv.
    """
    _task_screen_log("Installing Python dependencies for perphil …")
    c.run(f"{_venv_activate_prefix()} pip install -e '.[dev]'", pty=True)
    _task_screen_log("✔ Python-level dependencies installed.", color="yellow")


@task(pre=[install_deps])
def download_firedrake_configure(c):
    """
    Ensure a usable 'firedrake-configure' script is present in the repo root.
    Prefer the latest Firedrake release, but fall back gracefully to:
      1) existing local file (if present), or
      2) the 'main' branch script from Firedrake (best-effort).
    """
    # 0) If file already exists, keep it (avoid network flakiness in CI)
    if os.path.isfile("firedrake-configure"):
        _task_screen_log("Found existing firedrake-configure; skipping download.", color="green")
        # Ensure it's executable
        c.run("chmod +x firedrake-configure", warn=True)
        return

    _task_screen_log("Looking up the latest Firedrake release…")

    # Build curl headers (GitHub API can be picky/rate-limited)
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = "-H 'User-Agent: perphil-ci' -H 'Accept: application/vnd.github+json'"
    if gh_token:
        headers += f" -H 'Authorization: Bearer {gh_token}' -H 'X-GitHub-Api-Version: 2022-11-28'"

    # 1) Try GitHub API for the latest tag
    cmd = (
        f"curl -s {headers} https://api.github.com/repos/firedrakeproject/firedrake/releases/latest "
        "| grep -E '\"tag_name\"' | cut -d '\"' -f 4"
    )
    result = c.run(cmd, hide=True, warn=True)

    latest_tag = (result.stdout or "").strip()
    if latest_tag:
        _task_screen_log(f"Latest Firedrake release is '{latest_tag}'", color="green")
        raw_url = (
            "https://raw.githubusercontent.com/"
            f"firedrakeproject/firedrake/{latest_tag}/scripts/firedrake-configure"
        )
        dl = c.run(f"curl -fsSL {raw_url} -o firedrake-configure", warn=True, echo=True)
        if not dl.failed and os.path.isfile("firedrake-configure"):
            c.run("chmod +x firedrake-configure", warn=True)
            _task_screen_log(f"✔ downloaded firedrake-configure@{latest_tag}", color="yellow")
            return

    # 2) Fallback: try 'main' branch if latest release lookup/download failed
    _task_screen_log("Falling back to firedrake 'main' branch script …", color="yellow")
    fallback_url = (
        "https://raw.githubusercontent.com/"
        "firedrakeproject/firedrake/main/scripts/firedrake-configure"
    )
    dl_fb = c.run(f"curl -fsSL {fallback_url} -o firedrake-configure", warn=True, echo=True)
    if not dl_fb.failed and os.path.isfile("firedrake-configure"):
        c.run("chmod +x firedrake-configure", warn=True)
        _task_screen_log("✔ downloaded firedrake-configure@main", color="yellow")
        return

    # 3) If still nothing, give a clear error with next-step guidance
    raise Exit(
        "Failed to obtain firedrake-configure (GitHub API and fallback both failed). "
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
    system = platform.system()
    # 1) ask firedrake-configure which packages it wants
    result = c.run("python3 firedrake-configure --show-system-packages", hide=True, warn=True)
    if result.failed:
        raise Exit("Failed to query `firedrake-configure --show-system-packages`")

    base_pkgs = result.stdout.strip().split()
    if system == "Linux":
        all_pkgs = base_pkgs + ["libopenmpi-dev", "openmpi-bin"]

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
            c.run(
                f'sudo sh -c "apt update && apt install -y {pkgs_str}"',
                pty=True,
            )
            _task_screen_log(f"✔ Installed: {pkgs_str}", color="yellow")
        else:
            _task_screen_log("✔ All system packages are already installed.", color="green")

    elif system == "Darwin":
        all_pkgs = base_pkgs  # macOS doesn’t need explicit OpenMPI lines here (brew will pull it if required)

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
            c.run(
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
        f"{prefix} python3 firedrake-configure --show-petsc-version",
        hide=True,
        warn=True,
        echo=True,
        pty=True,
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
            print(f"→ Exported PETSC_DIR={abs_petsc}")
            print(f"→ Exported PETSC_ARCH={arch}")
            return

    print("Gathering PETSc configure flags …")
    cfg_flags = (
        c.run(
            f"{prefix} python3 firedrake-configure --show-petsc-configure-options",
            hide=True,
            warn=True,
            echo=True,
            pty=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if not cfg_flags:
        raise Exit("No PETSc configure options found.")

    print("Configuring PETSc …")
    with c.cd(petsc_dir):
        # Join configure flags into a single invocation and pass CC/CXX explicitly
        cfg_joined = " ".join(shlex.quote(f) for f in cfg_flags)
        cc = os.environ.get("CC", "").strip()
        cxx = os.environ.get("CXX", "").strip()
        cc_arg = f" CC={shlex.quote(cc)}" if cc else ""
        cxx_arg = f" CXX={shlex.quote(cxx)}" if cxx else ""
        cmd = f"{prefix_down} ./configure {cfg_joined}{cc_arg}{cxx_arg}"
        c.run(cmd, echo=True, pty=True)

        print("Building PETSc (this may take a long time) …")
        c.run(f"make PETSC_DIR={abs_petsc} PETSC_ARCH={arch} all", echo=True)

        print("Checking PETSc installation…")
        try:
            c.run(f"make PETSC_DIR={abs_petsc} PETSC_ARCH={arch} check", echo=True)
        except Exception as e:
            _task_screen_log("PETSc compiled, but failed in the checks!", color="red")
            _task_screen_log(f"PETSc check output: \n{e}", color="red")

    # Export PETSC_DIR and PETSC_ARCH so that subsequent steps see them:
    os.environ["PETSC_DIR"] = abs_petsc
    os.environ["PETSC_ARCH"] = arch

    print(f"→ Exported PETSC_DIR={abs_petsc}")
    print(f"→ Exported PETSC_ARCH={arch}")
    _task_screen_log(
        f"✔ PETSc built.  PETSC_DIR: '{abs_petsc}', PETSC_ARCH: '{arch}'.", color="yellow"
    )


@task(pre=[install_petsc])
def install_firedrake(c):
    """
    Install the Firedrake Python package (with [check]) inside the venv via pip.
    We explicitly pass PETSC_DIR and PETSC_ARCH (from install_petsc) so that petsc4py
    can find the correct build.
    """
    prefix = _venv_activate_prefix()

    # Grab exactly what install_petsc set earlier in os.environ:
    petsc_dir = os.environ.get("PETSC_DIR", "").strip()
    petsc_arch = os.environ.get("PETSC_ARCH", "").strip()
    if not petsc_dir or not petsc_arch:
        raise Exit(
            "PETSC_DIR and PETSC_ARCH must be set by install_petsc before installing Firedrake."
        )

    _task_screen_log("Installing Firedrake in the virtualenv …")

    # Pin Cython < 3.1 in constraints.txt:
    c.run("echo 'Cython<3.1' > constraints.txt", echo=True)
    os.environ["PIP_CONSTRAINT"] = "constraints.txt"

    # Now call pip install, making sure PETSC_DIR/PETSC_ARCH/CC/CXX/HDF5_MPI are exported.
    # NOTE: PETSC_DIR must be absolute, otherwise petsc4py's build script will look in /tmp/…
    cmd = (
        f"{prefix}"
        f"PETSC_DIR={petsc_dir} PETSC_ARCH={petsc_arch} "
        f"CC=mpicc CXX=mpicxx HDF5_MPI=ON "
        f"pip install --no-binary h5py 'firedrake[check]'"
    )

    c.run(cmd, pty=True, echo=True)

    # Clean up constraint file:
    os.environ.pop("PIP_CONSTRAINT", None)

    print("\nVerifying the installation …")
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        c.run(f"{prefix} firedrake-check", echo=True, pty=True)
        _task_screen_log("✔ Firedrake installed successfully.", color="green")
        os.environ.pop("OMP_NUM_THREADS", None)
    except Exception as e:
        raise Exit(f"Failed to import Firedrake: {e}")


@task
def clean(c):
    """
    Remove the virtualenv, PETSc build directory, and any downloaded scripts.
    """
    _task_screen_log("Cleaning installation artifacts")
    # Remove any petsc-* directories:
    if any(name.startswith("petsc-") and os.path.isdir(name) for name in os.listdir(".")):
        print("Removing any 'petsc-*' directories …")
        c.run("rm -rf petsc-*", echo=True)
        c.run("pip uninstall -y petsc4py", echo=True)
        c.run("pip cache remove petsc4py", echo=True)

    # Remove firedrake-configure script if present:
    for fname in ["firedrake-configure"]:
        if os.path.isfile(fname):
            print(f"Removing '{fname}' …")
            c.run(f"rm -f {fname}", echo=True)

    # Uninstall any leftover Python packages:
    c.run("pip cache remove firedrake", echo=True)
    c.run("pip uninstall -y h5py mpi4py", echo=True)
    c.run("pip cache purge", echo=True)
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
        raise exceptions.Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
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
        raise exceptions.Exit(f"No notebooks found for {src}", 1)

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
    task_output_message = "Installing perphil in the active environment"
    _task_screen_log(task_output_message)
    base_command = 'pip install -e ".[dev]"'
    host_system = _HOST_SYSTEM
    if host_system not in _SUPPORTED_SYSTEMS:
        raise exceptions.Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
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
        raise exceptions.Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
    pty_flag = True if host_system != "Windows" else False
    _task_screen_log(f"Running: {base_command}", color="yellow", bold=False)
    ctx.run(base_command, pty=pty_flag)


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
        raise exceptions.Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
    pty_flag = True if host_system != "Windows" else False
    _task_screen_log(f"Running: {base_command}", color="yellow", bold=False)
    ctx.run(base_command, pty=pty_flag)


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
        raise exceptions.Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
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
        raise exceptions.Exit(f"{_PACKAGE_NAME} is running on unsupported operating system", code=1)
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
