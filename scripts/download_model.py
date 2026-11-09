#!/usr/bin/env python3
"""Download only reviewed public native assets; verify every SHA256 before use."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="native-model.lock.json")
    parser.add_argument("--output", default="artifacts/native-model")
    args = parser.parse_args()
    lock = json.loads(Path(args.lock).read_text())
    root = Path(args.output)
    for name, expected in lock["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid asset path")
        target = root / name
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f"existing asset differs; preserved: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        if temporary.exists():
            raise FileExistsError(temporary)
        url = f"https://huggingface.co/{lock['model_id']}/resolve/{lock['revision']}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as stream:
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
        if digest(temporary) != expected:
            raise ValueError(f"download hash differs; partial retained: {temporary}")
        temporary.rename(target)
        print(name, flush=True)


if __name__ == "__main__":
    main()
