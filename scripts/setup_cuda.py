#!/usr/bin/env python3
"""Install a minimal CUDA 13.2 toolkit in this study's ignored .cuda directory.

Uses NVIDIA's redistributable manifest and verifies each archive's SHA-256.
Requires Python 3.12+ (tarfile's data extraction filter), Linux x86_64, and an
existing compatible NVIDIA driver. Does not install or change the driver.
"""

import hashlib
import io
import json
import os
import pathlib
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
DEST = pathlib.Path(__file__).resolve().parents[1] / ".cuda"

# NVIDIA's archives use these compatibility symlinks.  Keep this list pinned to
# the CUDA release above: each public name becomes its own regular-file copy so
# the repository-local toolkit does not add links to the surrounding tree.
CUDA_ALIASES = (
    ("lib/libcudart.so", "lib/libcudart.so.13.2.51"),
    ("lib/libcudart.so.13", "lib/libcudart.so.13.2.51"),
    ("lib/libcurand.so", "lib/libcurand.so.10.4.2.51"),
    ("lib/libcurand.so.10", "lib/libcurand.so.10.4.2.51"),
    ("nvvm/lib64/libnvvm.so", "nvvm/lib64/libnvvm.so.4.0.0"),
    ("nvvm/lib64/libnvvm.so.4", "nvvm/lib64/libnvvm.so.4.0.0"),
)


def _relative_parts(relative: str) -> tuple[str, ...]:
    path = pathlib.PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"unsafe CUDA path: {relative!r}")
    return path.parts


def _secure_root(root: pathlib.Path) -> None:
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError(f"CUDA toolkit root does not exist: {root}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(f"CUDA toolkit root must be a real directory: {root}")


def _path_with_real_parents(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = root
    parts = _relative_parts(relative)
    for part in parts[:-1]:
        path /= part
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as error:
            raise RuntimeError(f"missing CUDA directory: {path}") from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise RuntimeError(f"CUDA path parent must be a real directory: {path}")
    return path / parts[-1]


def _regular_source(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = _path_with_real_parents(root, relative)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError(f"missing CUDA alias source: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError(f"CUDA alias source must be a real regular file: {path}")
    return path


def _symlinks(root: pathlib.Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    }


def _same_independent_file(alias: pathlib.Path, source: pathlib.Path) -> bool:
    try:
        alias_stat = alias.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(alias_stat.st_mode):
        return False
    source_stat = source.stat()
    if (alias_stat.st_dev, alias_stat.st_ino) == (source_stat.st_dev, source_stat.st_ino):
        return False
    if alias_stat.st_size != source_stat.st_size:
        return False
    with alias.open("rb") as left, source.open("rb") as right:
        while left_chunk := left.read(1024 * 1024):
            if left_chunk != right.read(len(left_chunk)):
                return False
        return not right.read(1)


def _atomic_copy(source: pathlib.Path, alias: pathlib.Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{alias.name}.", dir=alias.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode))
        os.replace(temporary, alias)
        directory_fd = os.open(alias.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_cuda_aliases(root: pathlib.Path) -> None:
    """Replace the release's six known aliases with independent regular copies."""
    _secure_root(root)
    expected_aliases = {alias for alias, _ in CUDA_ALIASES}
    unexpected = sorted(_symlinks(root) - expected_aliases)
    if unexpected:
        raise RuntimeError(f"unexpected CUDA symlink(s): {', '.join(unexpected)}")

    for alias_relative, source_relative in CUDA_ALIASES:
        source = _regular_source(root, source_relative)
        alias = _path_with_real_parents(root, alias_relative)
        if alias.exists() and not alias.is_symlink() and not stat.S_ISREG(alias.lstat().st_mode):
            raise RuntimeError(f"CUDA alias path must be a file or symlink: {alias}")
        if not _same_independent_file(alias, source):
            _atomic_copy(source, alias)

    residual = sorted(_symlinks(root))
    if residual:
        raise RuntimeError(f"residual CUDA symlink(s): {', '.join(residual)}")


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This helper supports Linux x86_64 only; use a system CUDA toolkit.")
    with urllib.request.urlopen(BASE + "redistrib_13.2.0.json", timeout=60) as response:
        manifest = json.load(response)
    DEST.mkdir(exist_ok=True)
    for name in ("cuda_cudart", "cuda_crt", "cuda_cccl", "cuda_nvcc", "cuda_tileiras", "libnvvm", "libcurand"):
        package = manifest[name]["linux-x86_64"]
        marker = DEST / ("." + name + ".sha256")
        if marker.is_file() and marker.read_text().strip() == package["sha256"]:
            print(f"Already installed: {name}", flush=True)
            continue
        print(f"Downloading {name}: {int(package['size']) / 1024**2:.2f} MiB", flush=True)
        with urllib.request.urlopen(BASE + package["relative_path"], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != package["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as archive:
            for member in archive.getmembers():
                parts = pathlib.PurePosixPath(member.name).parts
                if len(parts) > 1:
                    member.name = str(pathlib.PurePosixPath(*parts[1:]))
                    archive.extract(member, DEST, filter="data")
        marker.write_text(package["sha256"] + "\n")
    materialize_cuda_aliases(DEST)
    print(f"CUDA_TOOLKIT_PATH={DEST}")


if __name__ == "__main__":
    main()
