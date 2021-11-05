import math
import numpy as np
from PIL import Image


def compute_h(p1, p2):
    """
    Args:
        p1: corresponded (x,y)^T coordinates from igs_ref (N x 2 numpy array)
        p2: corresponded (x,y)^T coordinates from igs_in (N x 2 numpy array)
        p1 = H * p2

    Returns:
        H: homography matrix (3 x 3 numpy array)
    """
    N = p1.shape[0]  # number of correspondences
    # Least squares form: Ah = 0
    # Construct matrix A, size 2N x 9
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x1, y1 = p1[i, 0], p1[i, 1]
        x2, y2 = p2[i, 0], p2[i, 1]
        A[2 * i] = np.array([x2, y2, 1, 0, 0, 0, -x1 * x2, -x1 * y2, -x1])
        A[2 * i + 1] = np.array([0, 0, 0, x2, y2, 1, -y1 * x2, -y1 * y2, -y1])

    # Do the singular value decomposition of matrix A
    U, S, Vt = np.linalg.svd(A)

    # Retrieve the last row of Vt as h, the least squares solution
    h = Vt[-1]
    H = h.reshape(3, 3)

    return H


def compute_h_norm(p1, p2):
    """
    Args:
        p1: corresponded (x,y)^T coordinates from igs_ref (N x 2 numpy array)
        p2: corresponded (x,y)^T coordinates from igs_in (N x 2 numpy array)
        p1 = H * p2

    Returns:
        H: homography matrix after normalizing p1 and p2 (3 x 3 numpy array)
    """
    N1, N2 = p1.shape[0], p2.shape[0]
    p1_norm, p2_norm = np.zeros(p1.shape), np.zeros(p2.shape)  # data type can be float!

    # Reference: https://core.ac.uk/download/pdf/132551944.pdf (Check writeup.pdf)
    # 1. Translation: bring the centroid of the set of points to the origin of coordinates
    # centroid
    p1_bar = np.mean(p1, axis=0)  # [x_bar, y_bar]
    p2_bar = np.mean(p2, axis=0)

    # 2. Scaling: make the average distance from a point to the origin sqrt(2)
    # initial average distance from every point to the origin of coordinates
    d1_bar = np.linalg.norm(p1 - p1_bar, axis=1).sum() / N1
    d2_bar = np.linalg.norm(p2 - p2_bar, axis=1).sum() / N2
    # scaling factor s
    s1 = np.sqrt(2) / d1_bar
    s2 = np.sqrt(2) / d2_bar

    # 3. Construct the transformation matrices T1 and T2
    T1 = s1 * np.array([[1, 0, -p1_bar[0]], [0, 1, -p1_bar[1]], [0, 0, 1 / s1]])
    T2 = s2 * np.array([[1, 0, -p2_bar[0]], [0, 1, -p2_bar[1]], [0, 0, 1 / s2]])

    # 4. Normalize coordinates
    for i in range(N1):  # need homogeneous coordinates for each point
        coord_ref = np.array([p1[i, 0], p1[i, 1], 1])
        coord_norm = np.matmul(T1, coord_ref.T)
        p1_norm[i, 0] = coord_norm[0]
        p1_norm[i, 1] = coord_norm[1]  # since the last element of T1 is 1, we don't need additional division
    for i in range(N2):  # need homogeneous coordinates for each point
        coord_in = np.array([p2[i, 0], p2[i, 1], 1])
        coord_norm = np.matmul(T2, coord_in.T)
        p2_norm[i, 0] = coord_norm[0]
        p2_norm[i, 1] = coord_norm[1]

    # 5. Calculate the homography matrix
    H = compute_h(p1_norm, p2_norm)
    H = np.linalg.inv(T1).dot(H).dot(T2)  # undo normalization to obtain the original homography matrix back

    return H


def warp_image(igs_in, igs_ref, H):
    """
    Args:
        igs_in: input image (numpy array; shape is [y, x, 3])
        igs_ref: reference image (numpy array; shape is [y, x, 3])
        H: homography matrix after normalizing p1 and p2 (3 x 3 numpy array)

    Returns:
        igs_warp: igs_in warped according to H to be in the frame of the reference image igs_ref
        igs_merge: single mosaic image with a larger field of view containing both igs_in and igs_ref
    """
    igs_in_height, igs_in_width, _ = igs_in.shape
    igs_ref_height, igs_ref_width, _ = igs_ref.shape
    pad_r, pad_c = 461, 1640  # for perfect fit

    H_inv = np.linalg.inv(H)
    igs_warp = np.zeros_like(igs_ref)
    igs_merge = np.pad(igs_ref, ((pad_r, pad_r), (pad_c, 0), (0, 0)))  # make huge image like a zero padding

    # (x_, y_) are image coordinates; the origin (0, 0) is set at the lower left corner of igs_ref
    # As it seems normal for an image, I will scan the whole canvas from the bottom left and reach upper right
    # We apply H_inv to the whole blank canvas
    for y_ in range(-pad_r, igs_ref_height + pad_r):
        for x_ in range(-pad_c, igs_ref_width):
            coord_ref = np.array([x_, y_, 1])  # homogeneous coordinates in igs_ref
            coord_warped = np.matmul(H_inv, coord_ref.T)
            x = coord_warped[0] / coord_warped[2]  # inverse-transformed coordinates in igs_in
            y = coord_warped[1] / coord_warped[2]

            # From now on I use matrix(numpy) coordinates that correspond to the image coordinates
            # pixel (x, y) in img_in is equivalent to igs_in[y, x]
            # This is because numpy array has an origin in the upper left corner
            if 0 <= y < igs_in_height and 0 <= x < igs_in_width:
                # Bilinear interpolation
                # (i, j) is a matrix coordinate; this pixel is located at the upper left corner of (y, x)
                i, j = np.floor(y).astype(int), np.floor(x).astype(int)  # np.floor function only returns float values
                a, b = x - j, y - i
                if i < igs_in_height - 1 and j < igs_in_width - 1:
                    pixel = (1 - a) * (1 - b) * igs_in[i, j] + a * (1 - b) * igs_in[i, j + 1] + a * b * igs_in[
                        i + 1, j + 1] + (1 - a) * b * igs_in[i + 1, j]
                    if 0 <= y_ < igs_ref_height and 0 <= x_:  # x_ is already smaller than igs_ref_width
                        igs_warp[y_, x_] = pixel
                    igs_merge[y_ + pad_r, x_ + pad_c] = pixel

    return igs_warp, igs_merge


def rectify(igs, p1, p2):
    """
    Makes the new image plane parallel to the wall as best as possible
    Args:
        igs: input image to rectify (2D numpy array)
        p1: (x,y)^T coordinates of the four corner points located at the intersections of the four boundary straight lines) (N x 2 numpy array)
        p2: (x,y)^T coordinates of the target corner locations to compute the homography matrix (N x 2 numpy array)

    Returns:
        igs_rec: rectified image of the igs, with the same size as igs
    """

    igs_height, igs_width, _ = igs.shape
    H = compute_h_norm(p2, p1)
    H_inv = np.linalg.inv(H)
    igs_rec = np.zeros_like(igs)

    for y_ in range(igs_height):
        for x_ in range(igs_width):
            coord_ref = np.array([x_, y_, 1])
            coord_rec = np.matmul(H_inv, coord_ref.T)
            x = coord_rec[0] / coord_rec[2]
            y = coord_rec[1] / coord_rec[2]

            # From now on I use matrix(numpy) coordinates that correspond to the image coordinates
            if 0 <= x < igs_width and 0 <= y < igs_height:
                # Bilinear interpolation
                # (i, j) is a matrix coordinate; this pixel is located at the upper left corner of (y, x)
                i, j = np.floor(y).astype(int), np.floor(x).astype(int)  # np.floor function only returns float values
                a, b = x - j, y - i
                if i < igs_height - 1 and j < igs_width - 1:
                    pixel = (1 - a) * (1 - b) * igs[i, j] + a * (1 - b) * igs[i, j + 1] + a * b * igs[i + 1, j + 1] + (
                                1 - a) * b * igs[i + 1, j]
                    igs_rec[y_, x_] = pixel

    return igs_rec


def set_cor_mosaic():
    """
    Returns:
        p_in: corresponded (x,y)^T coordinates from igs_in (N x 2 numpy array)
        p_ref: corresponded (x,y)^T coordinates from igs_ref (N x 2 numpy array)
    """
    # TV, sign, door, chair, table, air conditioning
    p_in = np.array(
        [[1282, 417], [1284, 503], [1241, 544], [1253, 958], [1067, 435], [1069, 815], [1117, 759], [1169, 790],
         [1273, 255]])
    p_ref = np.array(
        [[536, 424], [537, 511], [495, 544], [509, 948], [320, 429], [323, 825], [373, 762], [426, 789], [526, 268]])

    return p_in, p_ref


def set_cor_rec():
    """
    Returns:
        c_in: (x,y)^T coordinates of the four corner points located at the intersections of the four boundary straight lines) (N x 2 numpy array)
        c_ref: (x,y)^T coordinates of the target corner locations to compute the homography matrix (N x 2 numpy array)
    """
    c_in = np.array([[1060, 160], [1403, 123], [1048, 869],
                     [1400, 888]])  # upper left, upper right, lower left, lower right, shadow left, shadow right
    c_ref = np.array([[1054, 133], [1400, 133], [1054, 842], [1400, 842]])

    return c_in, c_ref


def main():
    ##############
    # step 1: mosaicing
    ##############
    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_merged.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')


if __name__ == '__main__':
    main()
