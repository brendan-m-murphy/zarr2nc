from __future__ import annotations

import os
import shutil
from importlib.metadata import PackageNotFoundError, version

PYTHON_PACKAGES = ("zarr2nc", "xarray", "zarr", "dask", "h5netcdf", "h5py", "numpy")
OPTIONAL_TOOLS = ("ncrcat", "nccopy", "mpiexec")


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def detect_environment() -> str:
    if os.environ.get("PIXI_ENVIRONMENT_NAME") or os.environ.get("PIXI_PROJECT_ROOT"):
        return "pixi"
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        if os.environ.get("UV_PROJECT_ENVIRONMENT") or virtual_env.endswith(".venv"):
            return "uv/venv"
        return "venv"
    if os.environ.get("LOADEDMODULES"):
        return "system/HPC modules"
    return "system"


def collect_diagnostics() -> dict[str, dict[str, str] | str]:
    return {
        "environment": detect_environment(),
        "python_packages": {package: _package_version(package) for package in PYTHON_PACKAGES},
        "tools": {tool: shutil.which(tool) or "not found" for tool in OPTIONAL_TOOLS},
    }


def format_diagnostics(diagnostics: dict[str, dict[str, str] | str]) -> str:
    lines = [f"environment: {diagnostics['environment']}", "", "python packages:"]
    packages = diagnostics["python_packages"]
    assert isinstance(packages, dict)
    lines.extend(f"  {name}: {value}" for name, value in packages.items())
    lines.extend(("", "optional tools:"))
    tools = diagnostics["tools"]
    assert isinstance(tools, dict)
    lines.extend(f"  {name}: {value}" for name, value in tools.items())
    return "\n".join(lines)
