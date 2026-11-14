"""Unpack and verify the released AppWorld-derived evaluation datasets."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from appworld.common.constants import PASSWORD, SALT
from appworld.common.crypto import unpack_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / "data"
    destination = ROOT / ".appworld"
    bundle = data / "clarifysafe-v2.bundle"
    manifest = json.loads((data / "manifest.json").read_text())
    if sha256(bundle.read_bytes()).hexdigest() != manifest["bundle_sha256"]:
        raise SystemExit("The dataset bundle does not match its published checksum.")
    destination.mkdir(exist_ok=True)
    files = manifest["file_sha256"]
    folders = sorted({Path(name).parts[0] for name in files})
    with TemporaryDirectory(prefix="clarifysafe-unpack-", dir=destination) as stage:
        temporary = Path(stage)
        unpack_bundle(str(bundle), stage, PASSWORD, SALT)
        for name, digest in files.items():
            path = temporary / name
            if sha256(path.read_bytes()).hexdigest() != digest:
                raise SystemExit(f"Checksum mismatch after unpacking {name}.")
            existing = destination / name
            if (destination / Path(name).parts[0]).exists() and (
                not existing.is_file() or sha256(existing.read_bytes()).hexdigest() != digest
            ):
                raise SystemExit(f"Existing dataset differs at {existing}. Move the local copy first.")
        for folder in folders:
            target = destination / folder
            if not target.exists():
                (temporary / folder).rename(target)
    print("Verified and ready:", ", ".join(folders))


if __name__ == "__main__":
    main()
