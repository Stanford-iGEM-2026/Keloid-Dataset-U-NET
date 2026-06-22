#!/usr/bin/env python3
"""Verify generated keloid segmentation masks against source images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ALLOWED_VALUES = {0, 255}


def verify_masks(dataset_root: Path) -> int:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    missing_masks: list[str] = []
    dimension_mismatches: list[str] = []
    invalid_values: list[str] = []
    verified = 0

    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}.png"

        if not mask_path.is_file():
            missing_masks.append(image_path.name)
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: could not read image {image_path.name}")
            continue
        if mask is None:
            print(f"Warning: could not read mask {mask_path.name}")
            continue

        image_height, image_width = image.shape[:2]
        mask_height, mask_width = mask.shape[:2]

        if image_width != mask_width or image_height != mask_height:
            dimension_mismatches.append(
                f"{image_path.name}: image={image_width}x{image_height}, "
                f"mask={mask_width}x{mask_height}"
            )
            continue

        unique_values = set(np.unique(mask).tolist())
        if not unique_values.issubset(ALLOWED_VALUES):
            invalid_values.append(
                f"{mask_path.name}: found pixel values {sorted(unique_values)}"
            )
            continue

        verified += 1

    extra_masks = sorted(
        path.name
        for path in masks_dir.glob("*.png")
        if not (images_dir / f"{path.stem}.jpg").exists()
        and not (images_dir / f"{path.stem}.jpeg").exists()
        and not (images_dir / f"{path.stem}.png").exists()
    )

    print(f"Images checked: {len(image_paths)}")
    print(f"Verified image/mask pairs: {verified}")
    print(f"Missing masks: {len(missing_masks)}")
    if missing_masks:
        print("Missing:")
        for name in missing_masks:
            print(f"  - {name}")

    print(f"Dimension mismatches: {len(dimension_mismatches)}")
    if dimension_mismatches:
        print("Dimension mismatches:")
        for message in dimension_mismatches:
            print(f"  - {message}")

    print(f"Invalid mask values: {len(invalid_values)}")
    if invalid_values:
        print("Invalid values:")
        for message in invalid_values:
            print(f"  - {message}")

    if extra_masks:
        print(f"Extra masks without matching images: {len(extra_masks)}")
        for name in extra_masks:
            print(f"  - {name}")

    passed = (
        not missing_masks
        and not dimension_mismatches
        and not invalid_values
        and verified == len(image_paths)
    )

    print()
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify keloid masks match images in size and pixel values"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the keloid dataset root (default: script directory)",
    )
    args = parser.parse_args()

    try:
        return verify_masks(args.dataset_root)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
