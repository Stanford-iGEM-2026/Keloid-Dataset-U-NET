#!/usr/bin/env python3
"""Generate binary segmentation masks from CVAT Images 1.1 polygon annotations."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

TARGET_LABEL = "keloid"
BACKGROUND_VALUE = 0
FOREGROUND_VALUE = 255


def find_annotations_file(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "annotations.xml",
        dataset_root / "masks" / "annotations.xml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find annotations.xml in dataset root or masks/ subdirectory."
    )


def parse_polygon_points(points_str: str) -> np.ndarray:
    """Parse CVAT polygon points string into an (N, 2) float array."""
    vertices: list[list[float]] = []
    for pair in points_str.strip().split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",", 1)
        vertices.append([float(x_str), float(y_str)])
    if not vertices:
        raise ValueError("Polygon has no points.")
    return np.array(vertices, dtype=np.float32)


def load_image_size(image_path: Path) -> tuple[int, int]:
    """Return (width, height) from the source image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def create_mask(
    width: int,
    height: int,
    polygons: list[np.ndarray],
) -> np.ndarray:
    """Create a single-channel binary mask by filling all polygons."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        # OpenCV requires integer pixel coordinates; round without scaling.
        pts = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], FOREGROUND_VALUE)
    return mask


def generate_masks(
    dataset_root: Path,
    annotations_path: Path | None = None,
) -> int:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    annotations_file = annotations_path or find_annotations_file(dataset_root)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    masks_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(annotations_file)
    root = tree.getroot()

    generated = 0
    skipped: list[str] = []
    errors: list[str] = []

    for image_elem in root.findall("image"):
        image_name = image_elem.get("name")
        if not image_name:
            errors.append("Found <image> element without a name attribute.")
            continue

        image_path = images_dir / image_name
        stem = Path(image_name).stem
        mask_path = masks_dir / f"{stem}.png"

        keloid_polygons = [
            polygon
            for polygon in image_elem.findall("polygon")
            if polygon.get("label") == TARGET_LABEL
        ]

        if not keloid_polygons:
            skipped.append(image_name)
            continue

        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Source image not found: {image_path}")

            width, height = load_image_size(image_path)

            xml_width = image_elem.get("width")
            xml_height = image_elem.get("height")
            if xml_width is not None and xml_height is not None:
                if int(xml_width) != width or int(xml_height) != height:
                    raise ValueError(
                        f"XML dimensions ({xml_width}x{xml_height}) do not match "
                        f"image dimensions ({width}x{height}) for {image_name}"
                    )

            polygons: list[np.ndarray] = []
            for polygon_elem in keloid_polygons:
                points_str = polygon_elem.get("points")
                if not points_str:
                    raise ValueError(f"Empty polygon points for {image_name}")
                polygons.append(parse_polygon_points(points_str))

            mask = create_mask(width, height, polygons)

            if mask.shape[1] != width or mask.shape[0] != height:
                raise ValueError(
                    f"Mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match "
                    f"image dimensions {width}x{height} for {image_name}"
                )

            if not cv2.imwrite(str(mask_path), mask):
                raise ValueError(f"Failed to write mask: {mask_path}")

            generated += 1
            print(f"Generated: {mask_path.name} ({width}x{height})")

        except Exception as exc:
            errors.append(f"{image_name}: {exc}")

    print()
    print(f"Masks generated: {generated}")
    print(f"Skipped images (no '{TARGET_LABEL}' annotations): {len(skipped)}")
    if skipped:
        print("Skipped:")
        for name in skipped:
            print(f"  - {name}")

    print(f"Errors: {len(errors)}")
    if errors:
        print("Errors:")
        for message in errors:
            print(f"  - {message}")

    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate binary keloid masks from CVAT annotations.xml"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the keloid dataset root (default: script directory)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Path to annotations.xml (default: dataset_root/annotations.xml)",
    )
    args = parser.parse_args()

    try:
        return generate_masks(args.dataset_root, args.annotations)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
