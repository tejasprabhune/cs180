import argparse
from pathlib import Path

from PIL import Image, ImageOps
import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

import matplotlib.pyplot as plt

import lib

def compute_H(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    """Compute the homography matrix between two sets of points."""
    
    A = []
    b = []
    for i in range(len(im1_pts)):
        x, y = im1_pts[i]
        xp, yp = im2_pts[i]
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp])
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp])
        b.append(xp)
        b.append(yp)

    A = np.array(A)
    b = np.array(b)

    H = np.linalg.lstsq(A, b, rcond=-1)[0]

    H = np.append(H, 1)

    return H.reshape(3, 3)

def warp_image(im1: np.ndarray, H: np.ndarray) -> np.ndarray:
    h1, w1 = im1.shape[:2]

    im1_pts = np.array([[0, 0], [im1.shape[1], 0], [0, im1.shape[0]], [im1.shape[1], im1.shape[0]]])
    im1_pts = np.append(im1_pts, np.ones((4, 1)), axis=1).T

    print("Image 1 boundary points:\n", im1_pts)

    im2_pts = np.dot(H, im1_pts)
    im2_pts = im2_pts / im2_pts[2]

    print("Image 2 boundary points:\n", im2_pts)
    
    min_x = int(min(im2_pts[0]))
    max_x = int(max(im2_pts[0]))
    min_y = int(min(im2_pts[1]))
    max_y = int(max(im2_pts[1]))

    print("Bounding box:", min_x, min_y, max_x, max_y)

    dst_y, dst_x = np.mgrid[min_y:max_y, min_x:max_x]
    dst_coords = np.vstack([dst_x.ravel(), dst_y.ravel()])

    dst_coords_homog = np.vstack([dst_coords, np.ones(dst_coords.shape[1])])

    src_coords_homog = np.dot(np.linalg.inv(H), dst_coords_homog)
    src_coords = src_coords_homog[:2] / src_coords_homog[2]

    valid_mask = (
        (src_coords[0] >= 0) & (src_coords[0] < w1) &
        (src_coords[1] >= 0) & (src_coords[1] < h1)
    )

    valid_dst_coords = dst_coords[:, valid_mask]
    valid_src_coords = src_coords[:, valid_mask]

    src_y, src_x = np.mgrid[0:h1, 0:w1]
    src_points = np.vstack((src_x.ravel(), src_y.ravel())).T
    src_values = im1.reshape(-1, im1.shape[-1])

    interpolator = LinearNDInterpolator(src_points, src_values)

    warped_image = np.zeros((max_y - min_y, max_x - min_x, im1.shape[-1]), dtype=im1.dtype)
    print("Warped image shape:", warped_image.shape)
    warped_image[valid_dst_coords[1] - min_y, valid_dst_coords[0] - min_x] = interpolator(valid_src_coords.T)

    return warped_image, min_x, min_y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and apply a homography between two images.")

    parser.add_argument("im1", help="Path to image 1")
    parser.add_argument("pts1", help="Path to image 1 points")
    parser.add_argument("im2", help="Path to image 2")
    parser.add_argument("pts2", help="Path to image 2 points")

    args = parser.parse_args()
    
    im1_path = Path(args.im1)
    im2_path = Path(args.im2)

    im1 = lib.import_image(im1_path)
    im1 = lib.im_scale(im1, 1)

    im2 = lib.import_image(im2_path)
    im2 = lib.im_scale(im2, 1)

    # Testing homography
    im1_pts = np.load(args.pts1)

    im2_pts = np.load(args.pts2)

    H = compute_H(im1_pts, im2_pts)

    print("Homography matrix:\n", H)

    root = im1_path.parent

    warped_im1, min_x, min_y = warp_image(im1, H)

    warped_im1 = Image.fromarray(warped_im1)
    warped_im1.save(root / f"{im1_path.stem}_warped_to_{im2_path.stem}.jpg")
    print("Warped image saved to", f"{root / im1_path.stem}_warped_to_{im2_path.stem}.jpg")

    im2_pts = im2_pts - np.array([min_x, min_y])

    np.save(root / f"{im1_path.stem}_{im2_path.stem}_wpts.npy", im2_pts)
