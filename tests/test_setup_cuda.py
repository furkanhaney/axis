import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "setup_cuda.py"
SPEC = importlib.util.spec_from_file_location("setup_cuda", SCRIPT)
assert SPEC and SPEC.loader
setup_cuda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_cuda)


class CudaAliasTests(unittest.TestCase):
    def make_toolkit(self, root: pathlib.Path, *, symlink_aliases: bool = True) -> None:
        for index, (alias_relative, source_relative) in enumerate(setup_cuda.CUDA_ALIASES):
            source = root / source_relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"versioned CUDA library {index}\n".encode())
            alias = root / alias_relative
            alias.parent.mkdir(parents=True, exist_ok=True)
            if symlink_aliases:
                alias.symlink_to(os.path.relpath(source, alias.parent))

    def test_materializes_six_independent_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)

            setup_cuda.materialize_cuda_aliases(root)

            self.assertEqual([], list(path for path in root.rglob("*") if path.is_symlink()))
            for alias_relative, source_relative in setup_cuda.CUDA_ALIASES:
                alias = root / alias_relative
                source = root / source_relative
                self.assertTrue(alias.is_file())
                self.assertFalse(alias.is_symlink())
                self.assertEqual(source.read_bytes(), alias.read_bytes())
                self.assertNotEqual(source.stat().st_ino, alias.stat().st_ino)

    def test_repairs_corruption_and_leaves_valid_copy_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)
            setup_cuda.materialize_cuda_aliases(root)
            stable = root / setup_cuda.CUDA_ALIASES[0][0]
            broken = root / setup_cuda.CUDA_ALIASES[1][0]
            stable_inode = stable.stat().st_ino
            broken.write_bytes(b"corrupt")

            setup_cuda.materialize_cuda_aliases(root)

            self.assertEqual(stable_inode, stable.stat().st_ino)
            self.assertEqual(
                (root / setup_cuda.CUDA_ALIASES[1][1]).read_bytes(), broken.read_bytes()
            )

    def test_rejects_unexpected_and_parent_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)
            (root / "unexpected").symlink_to("lib")
            with self.assertRaisesRegex(RuntimeError, "unexpected CUDA symlink"):
                setup_cuda.materialize_cuda_aliases(root)

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = pathlib.Path(directory)
            (root / "lib").symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "unexpected CUDA symlink"):
                setup_cuda.materialize_cuda_aliases(root)

    def test_main_repairs_aliases_after_all_package_markers_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)
            packages = {}
            for name in (
                "cuda_cudart", "cuda_crt", "cuda_cccl", "cuda_nvcc",
                "cuda_tileiras", "libnvvm", "libcurand",
            ):
                digest = f"digest-{name}"
                packages[name] = {"linux-x86_64": {"sha256": digest}}
                (root / f".{name}.sha256").write_text(digest + "\n")
            response = io.BytesIO(json.dumps(packages).encode())

            with mock.patch.object(setup_cuda, "DEST", root), mock.patch.object(
                setup_cuda.urllib.request, "urlopen", return_value=response
            ) as urlopen:
                setup_cuda.main()

            self.assertEqual(1, urlopen.call_count)
            self.assertFalse(any(path.is_symlink() for path in root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
