import argparse

from pathlib import Path

from PIL import Image
import numpy as np
from skimage.feature import corner_harris, peak_local_max
from skimage.transform import resize
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

import lib
import homography

def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1, threshold_rel=0.1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

def anms(coords, h, n_points=500, c_robust=0.9):

    coords = coords.T

    print(coords.shape)
    print(h.shape)
    strengths = h[coords[:, 0], coords[:, 1]]
    
    sorted_indices = np.argsort(strengths)[::-1]
    sorted_coords = coords[sorted_indices]
    sorted_strengths = strengths[sorted_indices]
    
    radii = np.full(len(sorted_coords), np.inf)
    
    for i in range(1, len(sorted_coords)):
        for j in range(i):
            dist = dist2(sorted_coords[i].reshape(1, 2), sorted_coords[j].reshape(1, 2))
            
            if sorted_strengths[j] > c_robust * sorted_strengths[i]:
                radii[i] = min(radii[i], dist)
    
    anms_indices = np.argsort(radii)[-n_points:]
    anms_points = sorted_coords[anms_indices]
    
    return anms_points

def feature_extractor(im, coords):
    features = []

    for i, (y, x) in enumerate(coords):
        patch_40 = im[y-20:y+20, x-20:x+20]

        patch_8 = resize(patch_40, (8, 8), anti_aliasing=True)

        patch_8 -= np.mean(patch_8)
        if np.std(patch_8) > 0:
            patch_8 /= np.std(patch_8)

        features.append(patch_8.flatten())

    return np.array(features)

def feature_matching(descriptors1, descriptors2, ratio_threshold=0.8):
    matches = []

    distances = cdist(descriptors1, descriptors2, 'euclidean')
    
    for i in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[i])
        closest_idx = sorted_indices[0]
        second_closest_idx = sorted_indices[1]

        ratio = distances[i, closest_idx] / distances[i, second_closest_idx]
        
        if ratio < ratio_threshold:
            matches.append((i, closest_idx))

    return matches

def ransac(matches, coords1, coords2, n_iters=1000, inlier_threshold=100):
    best_inliers = []
    best_H = None

    for _ in range(n_iters):
        sample = np.random.choice(len(matches), 4, replace=False)
        sample_matches = np.array(matches)[sample]

        sample_coords1 = coords1[sample_matches[:, 0]]
        sample_coords2 = coords2[sample_matches[:, 1]]

        H = homography.compute_H(sample_coords1, sample_coords2)

        inliers = []

        for i, j in matches:
            p1 = np.array([coords1[i, 1], coords1[i, 0], 1])
            p2 = np.array([coords2[j, 1], coords2[j, 0], 1])

            p2_pred = H @ p1

            p2_pred /= p2_pred[2]

            if np.linalg.norm(p2 - p2_pred) < inlier_threshold:
                inliers.append((i, j))

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    return best_H, best_inliers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Create correspondences between two images, and save scaled images.")
    parser.add_argument("im1", help="Path to image 1")
    parser.add_argument("im2", help="Path to image 2")

    args = parser.parse_args()

    im1_path = Path(args.im1)
    im2_path = Path(args.im2)

    im1_color = lib.import_image(im1_path)
    im1_color = lib.im_scale(im1_color, 1)

    im2_color = lib.import_image(im2_path)
    im2_color = lib.im_scale(im2_color, 1)
    
    im1 = lib.import_image(im1_path).convert('L')
    im1 = lib.im_scale(im1, 1)

    im2 = lib.import_image(im2_path).convert('L')
    im2 = lib.im_scale(im2, 1)

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    h1, coords1 = get_harris_corners(im1)
    plt.imshow(h1)
    plt.axis('off')

    plt.savefig(im1_path.parent / f"{im1_path.stem}_harris.jpg", dpi=600, bbox_inches='tight')

    h2, coords2 = get_harris_corners(im2)
    plt.imshow(h2)
    plt.axis('off')

    plt.savefig(im2_path.parent / f"{im2_path.stem}_harris.jpg", dpi=600, bbox_inches='tight')

    plt.imshow(im1, cmap='gray')
    plt.plot(coords1[1], coords1[0], 'r.', markersize=1)
    plt.axis('off')

    plt.savefig(im1_path.parent / f"{im1_path.stem}_corners.jpg", dpi=600, bbox_inches='tight')

    plt.clf()

    plt.imshow(im2, cmap='gray')
    plt.plot(coords2[1], coords2[0], 'r.', markersize=1)
    plt.axis('off')

    plt.savefig(im2_path.parent / f"{im2_path.stem}_corners.jpg", dpi=600, bbox_inches='tight')

    plt.clf()

    anms_points1 = anms(coords1, h1)

    plt.imshow(im1, cmap='gray')
    plt.plot(anms_points1[:, 1], anms_points1[:, 0], 'r.', markersize=1)
    plt.axis('off')

    plt.savefig(im1_path.parent / f"{im1_path.stem}_anms.jpg", dpi=600, bbox_inches='tight')

    plt.clf()

    anms_points2 = anms(coords2, h2)
    
    plt.imshow(im2, cmap='gray')

    plt.plot(anms_points2[:, 1], anms_points2[:, 0], 'r.', markersize=1)
    plt.axis('off')

    plt.savefig(im2_path.parent / f"{im2_path.stem}_anms.jpg", dpi=600, bbox_inches='tight')

    plt.clf()

    features1 = feature_extractor(im1, anms_points1)
    
    features2 = feature_extractor(im2, anms_points2)

    matches = feature_matching(features1, features2, ratio_threshold=0.5)

    plt.imshow(np.hstack((im1_color, im2_color)), cmap='gray')

    for i, j in matches:
        plt.plot([anms_points1[i, 1], anms_points2[j, 1] + im1.shape[1]], [anms_points1[i, 0], anms_points2[j, 0]], 'r-', lw=0.5)

    plt.axis('off')

    plt.savefig(im1_path.parent / f"{im1_path.stem}_{im2_path.stem}_feat_matches.jpg", dpi=600, bbox_inches='tight')

    plt.clf()

    H, inliers = ransac(matches, anms_points1, anms_points2, n_iters=20000, inlier_threshold=140)
    print(f"Number of inliers: {len(inliers)}")
    print(f"H:\n{H}")

    kp1 = np.array([anms_points1[i][::-1] for i, j in inliers])
    kp2 = np.array([anms_points2[j][::-1] for i, j in inliers])

    np.save(im1_path.parent / f"{im1_path.stem}_{im2_path.stem}_kp1.npy", kp1)
    np.save(im1_path.parent / f"{im1_path.stem}_{im2_path.stem}_kp2.npy", kp2)

    plt.imshow(np.hstack((im1_color, im2_color)), cmap='gray')

    for i, j in inliers:
        plt.plot([anms_points1[i, 1], anms_points2[j, 1] + im1.shape[1]], [anms_points1[i, 0], anms_points2[j, 0]], 'r-', lw=0.5)

    plt.axis('off')

    plt.savefig(im1_path.parent / f"{im1_path.stem}_{im2_path.stem}_ransac_matches.jpg", dpi=600, bbox_inches='tight')
