import os
import sys
from invoke import task, exceptions, Exit
from rich import print
import platform


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
    if os.path.isdir(VENV_DIR):
        print(f"Virtualenv already exists at '{VENV_DIR}/'")
        return

    ver = sys.version_info
    if ver < (3, 10):
        raise Exit(f"Python 3.10+ is required (found {ver[0]}.{ver[1]}).")

    print(f"Creating virtualenv in '{VENV_DIR}/' …")
    c.run(f"python3 -m venv {VENV_DIR}", pty=True)
    c.run(f"{_venv_activate_prefix()} pip install --upgrade pip setuptools wheel", pty=True)
    print("✔ Virtualenv created.")


@task(pre=[create_venv])
def install_deps(c):
    """
    Install all Python dependencies listed in requirements.txt into the venv.
    """
    requirements_file = "requirements.txt"
    if not os.path.isfile(requirements_file):
        raise Exit(f"Could not find {requirements_file!r} in the project root.")
    print(f"Installing Python dependencies from {requirements_file} …")
    c.run(f"{_venv_activate_prefix()} pip install -r {requirements_file}", pty=True)
    print("✔ Python-level dependencies installed.")


@task(pre=[install_deps])
def download_firedrake_configure(c):
    """
    Download the 'firedrake-configure' helper script into the repo root.
    """
    url = (
        "https://raw.githubusercontent.com/"
        "firedrakeproject/firedrake/refs/tags/2025.4.0.post0/scripts/firedrake-configure"
    )
    print("Downloading firedrake-configure …")
    c.run(f"curl -fsSL {url} -o firedrake-configure")
    c.run("chmod +x firedrake-configure")
    print("✔ firedrake-configure downloaded.")


@task(pre=[download_firedrake_configure])
def install_system_packages(c):
    system = platform.system()
    if system == "Linux":
        print("Detected Linux. Installing system packages via apt …")
        # Use c.run with pty=True so sudo can read your password properly
        c.run("sudo apt update", pty=True)
        c.run(
            "sudo apt install -y $(python3 firedrake-configure --show-system-packages)",
            pty=True,
        )
        print("✔ System packages installed on Ubuntu.")
    elif system == "Darwin":
        print("Detected macOS. Installing system packages via brew …")
        c.run("brew update", echo=True)  # brew never needs sudo
        c.run(
            "brew install $(python3 firedrake-configure --show-system-packages)",
            pty=True,
        )
        print("✔ System packages installed on macOS.")
    else:
        raise Exit(f"Unsupported OS: {system}. Please install system packages manually.")


@task(pre=[install_system_packages])
def install_petsc(c):
    """
    Clone and build PETSc using exactly the configure options that Firedrake wants.
    """
    prefix = _venv_activate_prefix()
    prefix_down = f"source ../{VENV_DIR}/bin/activate && "
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
        c.run(f"make PETSC_DIR=$(pwd) PETSC_ARCH={arch} all", echo=True)

        print("Checking PETSc installation…")
        c.run(f"make PETSC_DIR=$(pwd) PETSC_ARCH={arch} check", echo=True)

    # Export PETSC_DIR and PETSC_ARCH env vars
    os.environ["PETSC_DIR"] = petsc_dir
    os.environ["PETSC_ARCH"] = arch
    print(f"→ Exported PETSC_DIR={petsc_dir}")
    print(f"→ Exported PETSC_ARCH={arch}")

    print(f"✔ PETSc built.  PETSC_DIR: './{petsc_dir}', PETSC_ARCH: '{arch}'.")


@task(pre=[install_petsc])
def install_firedrake(c):
    """
    Install the Firedrake Python package inside the venv via pip.
    """
    prefix = _venv_activate_prefix()
    print("Installing Firedrake in the virtualenv …")
    c.run(f"{prefix} export $(python3 firedrake-configure --show-env)", pty=True, echo=True)
    c.run(f"{prefix} pip cache remove petsc4py", pty=True, echo=True)
    c.run(f"{prefix} pip cache remove firedrake", pty=True, echo=True)
    c.run(f"{prefix} echo 'Cython<3.1' > constraints.txt", pty=True, echo=True)
    c.run(f"{prefix} export PIP_CONSTRAINT=constraints.txt", pty=True, echo=True)
    c.run(f"{prefix} pip install --no-binary h5py 'firedrake[check]'", pty=True, echo=True)
    c.run("export PIP_CONSTRAINT=", pty=True, echo=True)
    print("\nVerifying the installation …")
    try:
        c.run(f"{prefix} firedrake-check", echo=True, pty=True)
        print("✔ Firedrake installed successfully.")
    except Exception as e:
        raise Exit(f"Failed to import Firedrake: {e}")


@task
def clean(c):
    """
    Remove the virtualenv, PETSc build directory, and any downloaded scripts.
    """
    if os.path.isdir("petsc-*"):
        print("Removing any 'petsc-*' directories …")
        c.run("rm -rf petsc-*", echo=True)
        c.run("pip uninstall -y petsc4py", echo=True)
        c.run("pip cache remove petsc4py", echo=True)
    for fname in ["firedrake-configure"]:
        if os.path.isfile(fname):
            print(f"Removing '{fname}' …")
            c.run(f"rm -f {fname}", echo=True)

    c.run("pip cache remove firedrake", echo=True)
    c.run("pip uninstall -y h5py mpi4py", echo=True)
    c.run("pip cache purge", echo=True)
    print("✔ Cleanup complete.")
