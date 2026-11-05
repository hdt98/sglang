#!/usr/bin/env python3
"""Reconstruct and verify the DSv4.1 banked serving overlay on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SGLANG_TARGET_PREFIX = "/sgl-workspace/sglang/"
EXPECTED_MOUNTS = 26


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_sha256(path: Path, kind: str) -> str:
    if kind == "file":
        return sha256_file(path)
    if kind == "directory":
        return tree_sha256(path)
    raise ValueError(f"unsupported mount type: {kind}")


def artifact_size(path: Path, kind: str) -> int:
    if kind == "file":
        return path.stat().st_size
    if kind == "directory":
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise ValueError(f"unsupported mount type: {kind}")


def snapshot_path(root: Path, mount: dict[str, Any]) -> Path:
    target = mount["target"]
    if mount["inside_sglang"]:
        if not target.startswith(SGLANG_TARGET_PREFIX):
            raise ValueError(f"invalid SGLang target: {target}")
        return root / target.removeprefix(SGLANG_TARGET_PREFIX)
    source_name = Path(mount["source"]).name
    return root / "supercompiler" / "dsv41" / "external" / f"{mount['index']:02d}" / source_name


def changed_from_image(
    root: Path, image_commit: str, relative_path: str, new_file: bool
) -> tuple[bool, str | None]:
    if not relative_path.startswith("python/"):
        return True, None
    if new_file:
        process = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{image_commit}:{relative_path}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if process.returncode != 0:
            return True, None
    process = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", image_commit, "--", relative_path],
        check=False,
    )
    if process.returncode == 0:
        return False, "snapshot is unchanged from the image commit"
    if process.returncode == 1:
        return True, None
    raise RuntimeError(f"git diff failed for {relative_path}: {process.returncode}")


def copy_artifact(source: Path, destination: Path, kind: str, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "file":
        shutil.copyfile(source, destination)
        destination.chmod(mode & 0o777)
    elif kind == "directory":
        shutil.copytree(source, destination)
        destination.chmod(mode & 0o777)
        for child in destination.rglob("*"):
            if child.is_file():
                child.chmod(mode & 0o777)
    else:
        raise ValueError(f"unsupported mount type: {kind}")


def load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    mounts = document.get("mounts")
    if document.get("mount_count") != EXPECTED_MOUNTS or not isinstance(mounts, list):
        raise ValueError(f"manifest must declare {EXPECTED_MOUNTS} mounts")
    if len(mounts) != EXPECTED_MOUNTS:
        raise ValueError(f"manifest declares {len(mounts)} mounts")
    if [mount["index"] for mount in mounts] != list(range(EXPECTED_MOUNTS)):
        raise ValueError("manifest mount indices are not contiguous")
    return document


def verify(root: Path, reference: Path, output: Path) -> dict[str, Any]:
    manifest_path = root / "supercompiler" / "dsv41" / "MANIFEST.json"
    manifest = load_manifest(manifest_path)
    rows = []
    with tempfile.TemporaryDirectory(prefix="dsv41-overlay-") as temporary:
        reconstructed = Path(temporary) / "mounts"
        for mount in manifest["mounts"]:
            index = mount["index"]
            snapshot = snapshot_path(root, mount)
            reference_artifact = reference / f"{index:02d}" / Path(mount["source"]).name
            if not snapshot.exists():
                raise FileNotFoundError(f"missing branch snapshot: {snapshot}")
            if not reference_artifact.exists():
                raise FileNotFoundError(f"missing reference artifact: {reference_artifact}")

            observed_hash = artifact_sha256(snapshot, mount["type"])
            observed_size = artifact_size(snapshot, mount["type"])
            reference_hash = artifact_sha256(reference_artifact, mount["type"])
            reference_size = artifact_size(reference_artifact, mount["type"])
            changed, reason = changed_from_image(
                root,
                manifest["image_git_commit"],
                snapshot.relative_to(root).as_posix(),
                bool(mount["new_file"]),
            )
            if observed_hash != mount["sha256"] or reference_hash != mount["sha256"]:
                raise ValueError(f"hash mismatch at mount {index} ({mount['target']})")
            if observed_size != mount["size"] or reference_size != mount["size"]:
                raise ValueError(f"size mismatch at mount {index} ({mount['target']})")
            if not changed:
                raise ValueError(f"mount {index}: {reason}")

            destination = reconstructed / f"{index:02d}" / Path(mount["source"]).name
            copy_artifact(snapshot, destination, mount["type"], mount["mode"])
            reconstructed_hash = artifact_sha256(destination, mount["type"])
            reconstructed_size = artifact_size(destination, mount["type"])
            if reconstructed_hash != mount["sha256"] or reconstructed_size != mount["size"]:
                raise ValueError(f"reconstruction mismatch at mount {index}")
            if stat.S_IMODE(destination.stat().st_mode) != (mount["mode"] & 0o777):
                raise ValueError(f"mode mismatch at mount {index}")

            rows.append(
                {
                    "index": index,
                    "target": mount["target"],
                    "snapshot": snapshot.relative_to(root).as_posix(),
                    "type": mount["type"],
                    "size": observed_size,
                    "sha256": observed_hash,
                    "reference_equal": True,
                    "reconstruction_equal": True,
                    "changed_from_image": changed,
                }
            )

    result = {
        "schema_version": "dsv41_overlay_equivalence.v1",
        "image": manifest["image"],
        "image_id": manifest["image_id"],
        "image_git_commit": manifest["image_git_commit"],
        "manifest_sha256": sha256_file(manifest_path),
        "mount_count": len(rows),
        "all_equal": True,
        "mounts": rows,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.chmod(0o644)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.reference.resolve(), args.output.resolve())
    print(f"{result['mount_count']}/{result['mount_count']} mounts byte-equal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
