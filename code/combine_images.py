import argparse
from pathlib import Path

from PIL import Image, ImageOps
import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

import matplotlib.pyplot as plt

import lib
import homography

def blend_dt_images(im1, im2):
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1_blurred = cv2.GaussianBlur(im1_gray, (3, 3), 0)
    im2_blurred = cv2.GaussianBlur(im2_gray, (3, 3), 0)

    dt1 = lib.distance_transform(im1_blurred)
    dt2 = lib.distance_transform(im2_blurred)

    im1_low_pass = cv2.GaussianBlur(im1, (5, 5), 0)
    im2_low_pass = cv2.GaussianBlur(im2, (5, 5), 0)

    im1_high_pass = im1 - im1_low_pass
    im2_high_pass = im2 - im2_low_pass

    high_pass_blend = np.where(dt1[..., np.newaxis] > dt2[..., np.newaxis], im1_high_pass, im2_high_pass)

    print(high_pass_blend.dtype, im1_low_pass.dtype)

    # Blend low-pass images with weighted combination using distance transforms
    dt_sum = dt1 + dt2
    dt1_weight = np.divide(dt1, dt_sum, where=dt_sum != 0)[..., np.newaxis]
    dt2_weight = np.divide(dt2, dt_sum, where=dt_sum != 0)[..., np.newaxis]
    low_pass_blend = im1_low_pass * dt1_weight + im2_low_pass * dt2_weight

    low_pass_blend = np.clip(low_pass_blend, 0, 255)
    low_pass_blend = low_pass_blend.astype(np.uint8)
    # high_pass_blend = high_pass_blend.astype(np.float32)

    print(low_pass_blend.dtype, high_pass_blend.dtype)

    final_image = low_pass_blend + high_pass_blend

    # final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    return final_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("im1", type=Path, help="Path to the middle image.")
    parser.add_argument("im2", type=Path, help="Path to the other image.")
    parser.add_argument("im1_pts", type=Path, help="Path to the middle image points.")
    parser.add_argument("im2_pts", type=Path, help="Path to the other image points.")
    parser.add_argument("--output", type=str, help="Name of the output image.", default="output")

    args = parser.parse_args()
    
    im1 = lib.import_image(args.im1)
    im2 = lib.import_image(args.im2)

    im1 = lib.im_scale(im1, 1)
    im2 = lib.im_scale(im2, 1)

    # Align images

    im1_pts = np.load(args.im1_pts)
    im2_pts = np.load(args.im2_pts)

    plt.imshow(im1)
    plt.plot(im1_pts[:, 0], im1_pts[:, 1], 'r.', markersize=1)
    plt.show()

    H = homography.compute_H(im1_pts, im2_pts)

    warped_image, min_x, min_y = homography.warp_image(im1, H)
    print("Warped image shape:", warped_image.shape)
    print("Min x:", min_x, "Min y:", min_y)
    
    # Put the images together

    im1_canvas = np.zeros((warped_image.shape[0], warped_image.shape[1] + im2.shape[1], 3), dtype=np.uint8)

    im2_canvas = im1_canvas.copy()

    print("Canvas shape:", im1_canvas.shape)

    im1_canvas[:warped_image.shape[0], :warped_image.shape[1]] = warped_image

    plt.imshow(im1_canvas)
    plt.show()


    if min_x < 0 and min_y < 0:
        print(im2.shape)
        im2_canvas[-min_y:im2.shape[0]-min_y, -min_x:im2.shape[1]-min_x] = im2
    elif min_x < 0:
        im2_canvas[:im2.shape[0], -min_x:im2.shape[1]-min_x] = im2
    elif min_y < 0:
        im2_canvas[-min_y:im2.shape[0]-min_y, :im2.shape[1]] = im2
    else:
        im2_canvas[:im2.shape[0], :im2.shape[1]] = im2

    output = blend_dt_images(im1_canvas, im2_canvas)

    plt.imshow(output)
    plt.show()

    # Save the output image
    output_path = args.im1.with_name(args.output + ".png")
    Image.fromarray(output).save(output_path)
