import argparse
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np

import cv2

import matplotlib.pyplot as plt

def import_image(path: Path) -> Image:
    """Import an image from a file."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    print("Imported image from", path)
    return im

def im_scale(im: Image, scale: float) -> np.ndarray:
    """Resize an image by a scale factor."""
    out_im = im.resize((int(im.size[0]*scale), int(im.size[1]*scale)))
    print("Resized image to", out_im.size)
    out_im = np.array(out_im)
    return out_im

def distance_transform(im: np.ndarray) -> np.ndarray:
    """Compute the distance transform of an image."""

    return cv2.distanceTransform(im, cv2.DIST_L2, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Create distance transforms for two images.")
    parser.add_argument("im1", help="Path to image 1")
    parser.add_argument("im2", help="Path to image 2")

    args = parser.parse_args()

    im1 = import_image(Path(args.im1)).convert("L")
    im1 = im_scale(im1, 1)


    im2 = import_image(Path(args.im2)).convert("L")
    im2 = im_scale(im2, 1)

    im1_blurred = cv2.GaussianBlur(im1, (3, 3), 0)
    im2_blurred = cv2.GaussianBlur(im2, (3, 3), 0)

    im1_dist = distance_transform(im1_blurred)
    im2_dist = distance_transform(im2_blurred)

    im1_dist = Image.fromarray(im1_dist).convert("L")
    im2_dist = Image.fromarray(im2_dist).convert("L")

    im1_dist.save(Path(args.im1).parent / f"{Path(args.im1).stem}_dist.png")

    im2_dist.save(Path(args.im2).parent / f"{Path(args.im2).stem}_dist.png")

