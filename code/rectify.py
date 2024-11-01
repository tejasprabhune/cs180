import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import lib
import homography

def main():
    parser = argparse.ArgumentParser(description="Find and apply a homography between two images.")

    parser.add_argument("im", help="Path to image")
    parser.add_argument("pts", help="Path to image points")

    args = parser.parse_args()

    im_path = Path(args.im)

    im = lib.import_image(im_path)
    im = lib.im_scale(im, 1)

    pts = np.load(args.pts)

    print(im.shape)
    print(pts)

    square_pts = np.array([[0, 0], [0, 200], [200, 0], [200, 200]])

    H = homography.compute_H(pts, square_pts)
    print(H)

    warped_im1, min_x, min_y = homography.warp_image(im, H)
    warped_im1 = Image.fromarray(warped_im1)

    warped_im1.save(f"rectified/{im_path.stem}_rectified.jpg")
    print("Rectified image saved to", f"rectified/{im_path.stem}_rectified.jpg")


if __name__ == "__main__":
    main()
