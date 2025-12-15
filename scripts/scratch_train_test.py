#!/usr/bin/env python3
"""
Usage:
    python scratch_train_test.py /path/to/list.txt NEW_PREFIX

This will produce /path/to/list_scratch.txt with updated paths.
If a line does not contain 'embeddings_0_1' it will be left unchanged.
"""

import argparse
from pathlib import Path

def transform_path(path_str: str, replacement: str, marker: str = "embeddings_0_1") -> str:
    """
    Remove everything up to and including the first occurrence of `marker` (as a path component)
    and replace it with `replacement`. If marker not found, return original path (stripped).
    """
    if not path_str:
        return path_str

    # Normalize and split into components without touching separators semantics too roughly:
    p = Path(path_str.strip())
    parts = p.parts  # tuple of components, preserves root (e.g. '/' on Unix)

    # Find first index of marker
    try:
        idx = parts.index(marker)
    except ValueError:
        # marker not present — return original line (stripped)
        return path_str.strip()

    # Remainder after marker
    remainder_parts = parts[idx+1:]  # may be empty tuple

    # Build new path: replacement followed by remainder parts
    if remainder_parts:
        new_path = Path(replacement, *remainder_parts)
    else:
        new_path = Path(replacement)

    # Return as POSIX path string on all platforms to keep paths consistent in text files,
    # unless original had a drive/UNC root (Windows). We'll return the OS-native string.
    return str(new_path)

def process_file(input_file: Path, replacement: str):
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Build output filename: same folder, same stem + '_scratch' + original suffixes (e.g. .txt)
    out_name = input_file.stem + "_scratch" + "".join(input_file.suffixes)
    out_file = input_file.with_name(out_name)

    with input_file.open("r", encoding="utf-8") as rf, out_file.open("w", encoding="utf-8") as wf:
        for line in rf:
            # keep empty lines as-is (preserve)
            if line.strip() == "":
                wf.write("\n")
                continue

            transformed = transform_path(line, replacement)
            wf.write(transformed + "\n")

    print(f"Processed {input_file} -> {out_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Replace prefix up to and including 'embeddings_0_1' with a provided variable."
    )
    parser.add_argument("--input_txt", type=Path, help="Path to the .txt file containing file paths (one per line).")
    parser.add_argument("--replacement", help="Replacement string (prefix) that will replace the folders up to and including 'embeddings_0_1'.")
    args = parser.parse_args()

    process_file(args.input_txt, args.replacement)

if __name__ == "__main__":
    main()
