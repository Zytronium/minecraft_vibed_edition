#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path

ROOT = Path(".")
FILES_TO_PRINT = [
    Path("Cargo.toml"),
    Path(".claude"),
]

def run_tree():
    try:
        result = subprocess.run(
            ["tree", ".", "-I", "target/"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
    except FileNotFoundError:
        print("tree command not found. Please install it.")
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)

def print_file(path: Path):
    if not path.exists() or not path.is_file():
        return

    print(f"=== {path} ===")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            print(f.read())
    except Exception as e:
        print(f"[Error reading file: {e}]")
    print("\n----------------------------\n")

def print_src_files():
    src_dir = ROOT / "src"
    if not src_dir.exists():
        return

    for root, _, files in os.walk(src_dir):
        for name in sorted(files):
            print_file(Path(root) / name)

def main():
    run_tree()
    print_src_files()

    for file_path in FILES_TO_PRINT:
        print_file(file_path)

if __name__ == "__main__":
    main()
