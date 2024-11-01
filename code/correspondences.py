import argparse
from pathlib import Path

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import cv2

import lib

def plot_correspondences(im1: Image, im2: Image, num_pts: int = 10):
    """Plot correspondences between two images to get two (n x 2) point arrays."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(im1)
    ax[1].imshow(im2)

    pts = plt.ginput(num_pts * 2, timeout=0)
    plt.close(fig)

    im1_pts, im2_pts = [], []

    for i in range(0, len(pts), 2):
        im1_pts.append(list(pts[i]))
        im2_pts.append(list(pts[i+1]))

    return np.array(im1_pts), np.array(im2_pts)

def create_correspondences(im1_path: Path, im2_path: Path, num_pts: int = 10):
    """Create correspondences between two images and save them to files."""
    im1 = lib.import_image(im1_path)
    im1 = lib.im_scale(im1, args.scale)

    im2 = lib.import_image(im2_path)
    im2 = lib.im_scale(im2, args.scale)

    im1_pts, im2_pts = plot_correspondences(im1, im2, num_pts=num_pts)

    print("Image 1 points:\n", im1_pts)
    print("Image 2 points:\n", im2_pts)

    print("Saving points to file...")

    root = im1_path.parent

    np.save(root / f"{im1_path.stem}_{im2_path.stem}_pts.npy", im1_pts)
    np.save(root / f"{im2_path.stem}_{im1_path.stem}_pts.npy", im2_pts)

    print("im1 points saved to", f"{im1_path.stem}_{im2_path.stem}_pts.npy")
    print("im2 points saved to", f"{im2_path.stem}_{im1_path.stem}_pts.npy")

    # Show matches

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.imshow(np.hstack((im1, im2)))

    for i in range(len(im1_pts)):
        ax.plot([im1_pts[i][0], im2_pts[i][0] + im1.shape[1]], [im1_pts[i][1], im2_pts[i][1]], "r.-")

    fig.savefig(root / f"{im1_path.stem}_{im2_path.stem}_matches.jpg")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Create correspondences between two images, and save scaled images.")
    parser.add_argument("im1", help="Path to image 1")
    parser.add_argument("im2", help="Path to image 2")
    parser.add_argument("--scale", type=float, default=0.2, required=False, help="Scale factor")
    parser.add_argument("--num_pts", type=int, default=10, required=False, help="Number of points to select")

    args = parser.parse_args()

    im1_path = Path(args.im1)
    im2_path = Path(args.im2)

    create_correspondences(im1_path, im2_path, num_pts=args.num_pts)

    root = im1_path.parent

    im1_scaled = lib.import_image(im1_path)
    im1_scaled = lib.im_scale(im1_scaled, args.scale)
    im1_scaled = Image.fromarray(im1_scaled)
    im1_scaled.save(root / f"{im1_path.stem}_scaled.jpg")

    im2_scaled = lib.import_image(im2_path)
    im2_scaled = lib.im_scale(im2_scaled, args.scale)
    im2_scaled = Image.fromarray(im2_scaled)
    im2_scaled.save(root / f"{im2_path.stem}_scaled.jpg")

