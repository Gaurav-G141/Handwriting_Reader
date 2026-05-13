"""
Test harness for splitter.py's split() function.

Usage:
    python test_harness.py <input_folder> <output_folder>

For each image in <input_folder>, calls split() and saves the resulting
images into a dedicated sub-folder inside <output_folder>.
"""

import sys
import os
from pathlib import Path
from PIL import Image

# Make sure splitter.py is importable (assumes it lives alongside this script)
sys.path.insert(0, str(Path(__file__).parent))
from splitter import split


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}


def process_image(image_path: Path, output_root: Path) -> tuple[int, str | None]:
    """
    Run split() on a single image and write results to output_root/<stem>/.

    Returns (number_of_splits, error_message_or_None).
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return 0, f"Could not open image: {e}"

    try:
        results = split(img)
    except Exception as e:
        return 0, f"split() raised an exception: {e}"

    if not isinstance(results, list):
        return 0, f"split() returned {type(results).__name__!r} instead of a list"

    if len(results) == 0:
        return 0, "split() returned an empty list"

    # Create a sub-folder named after the source image (without extension)
    sub_folder = output_root / image_path.stem
    sub_folder.mkdir(parents=True, exist_ok=True)

    # Derive a safe output extension from the source image
    ext = image_path.suffix.lower() or ".png"
    if ext in (".jpg", ".jpeg"):
        save_ext = ".jpg"
    else:
        save_ext = ".png"          # fall back to PNG for lossless safety

    errors = []
    for idx, piece in enumerate(results):
        if not isinstance(piece, Image.Image):
            errors.append(f"  item[{idx}] is {type(piece).__name__!r}, not a PIL Image — skipped")
            continue
        out_path = sub_folder / f"{image_path.stem}_{idx:03d}{save_ext}"
        try:
            piece.save(out_path)
        except Exception as e:
            errors.append(f"  Could not save item[{idx}]: {e}")

    if errors:
        return len(results), "Partial errors:\n" + "\n".join(errors)

    return len(results), None


def run_harness(input_folder: str, output_folder: str) -> None:
    input_root = Path(input_folder)
    output_root = Path(output_folder)

    if not input_root.is_dir():
        print(f"ERROR: Input folder does not exist: {input_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p for p in input_root.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not candidates:
        print(f"No supported images found in {input_root}")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(0)

    print(f"Found {len(candidates)} image(s) in '{input_root}'")
    print(f"Output root: '{output_root}'\n")
    print(f"{'Image':<40} {'Splits':>6}  {'Status'}")
    print("-" * 70)

    passed = failed = 0

    for image_path in candidates:
        n_splits, error = process_image(image_path, output_root)

        if error:
            status = f"FAIL — {error}"
            failed += 1
        else:
            status = "OK"
            passed += 1

        print(f"{image_path.name:<40} {n_splits:>6}  {status}")

    print("-" * 70)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(candidates)} image(s)")


run_harness("./colorized_images","./split_images")