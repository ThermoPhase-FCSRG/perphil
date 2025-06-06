import glob
import os
from pathlib import Path
import shlex
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
    requirements_file = "requirements.txt"
    if not os.path.isfile(requirements_file):
        raise Exit(f"Could not find {requirements_file!r} in the project root.")
    _task_screen_log(f"Installing Python dependencies from {requirements_file} …")
    c.run(f"{_venv_activate_prefix()} pip install -r {requirements_file}", pty=True)
    _task_screen_log("✔ Python-level dependencies installed.", color="yellow")


@task(pre=[install_deps])
def download_firedrake_configure(c):
    """
    Download 'firedrake-configure' from the latest Firedrake release.
    """
    _task_screen_log("Looking up the latest Firedrake release…")

    # 1) Get the JSON for "latest release", then pull out tag_name.
    #    e.g. curl -s https://api.github.com/repos/firedrakeproject/firedrake/releases/latest
    result = c.run(
        "curl -s https://api.github.com/repos/firedrakeproject/firedrake/releases/latest "
        "| grep -E '\"tag_name\"' | cut -d '\"' -f 4",
        hide=True,
        warn=True,
    )

    if result.failed or not result.stdout.strip():
        raise Exit("Could not retrieve the latest Firedrake tag via GitHub API.")

    latest_tag = result.stdout.strip()
    _task_screen_log(f"Latest Firedrake release is '{latest_tag}'", color="green")

    # 2) Construct the raw URL for that tag
    raw_url = (
        "https://raw.githubusercontent.com/"
        f"firedrakeproject/firedrake/{latest_tag}/scripts/firedrake-configure"
    )

    _task_screen_log("Downloading firedrake-configure from the latest release …")
    c.run(f"curl -fsSL {raw_url} -o firedrake-configure", echo=True)
    c.run("chmod +x firedrake-configure")
    _task_screen_log(f"✔ downloaded firedrake-configure@{latest_tag}", color="yellow")


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
    if not os.path.isdir(petsc_dir):
        print(f"Cloning PETSc {petsc_version} …")
        c.run(f"git clone --branch {petsc_version} {petsc_repo} {petsc_dir}", echo=True)

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
        c.run(
            f"{prefix_down} python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure",
            echo=True,
            pty=True,
        )

        arch = "arch-firedrake-default"
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
        c.run(f"{prefix} firedrake-check", echo=True, pty=True)
        _task_screen_log("✔ Firedrake installed successfully.", color="green")
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
