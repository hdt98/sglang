from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from supercompiler.dsv41 import verify_overlay


class VerifyOverlayTest(unittest.TestCase):
    def test_snapshot_paths_cover_in_tree_and_external_mounts(self) -> None:
        root = Path("/repo")
        inside = {
            "index": 3,
            "inside_sglang": True,
            "source": "/remote/models/model.py",
            "target": "/sgl-workspace/sglang/python/model.py",
        }
        external = {
            "index": 8,
            "inside_sglang": False,
            "source": "/remote/aiter/module.so",
            "target": "/sgl-workspace/aiter/module.so",
        }

        self.assertEqual(
            verify_overlay.snapshot_path(root, inside), root / "python" / "model.py"
        )
        self.assertEqual(
            verify_overlay.snapshot_path(root, external),
            root
            / "supercompiler"
            / "dsv41"
            / "external"
            / "08"
            / "module.so",
        )

    def test_file_and_tree_hashes_match_export_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            single = directory / "single.txt"
            single.write_bytes(b"one")
            nested = directory / "tree"
            nested.mkdir()
            (nested / "a.txt").write_bytes(b"alpha")
            (nested / "b.txt").write_bytes(b"beta")

            self.assertEqual(
                verify_overlay.sha256_file(single),
                hashlib.sha256(b"one").hexdigest(),
            )
            digest = hashlib.sha256()
            for name, payload in (("a.txt", b"alpha"), ("b.txt", b"beta")):
                digest.update(name.encode())
                digest.update(b"\0")
                digest.update(payload)
                digest.update(b"\0")
            self.assertEqual(verify_overlay.tree_sha256(nested), digest.hexdigest())


if __name__ == "__main__":
    unittest.main()
