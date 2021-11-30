import datetime
import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    """
    Args:
        img1: image I(t), serves as the template to be tracked in image I(t+1)
        img2: image I(t+1), assumed to be approximately an affine warped version of I(t)
        p: 6-vector p = [p1 p2 p3 p4 p5 p6]^T of affine flow parameters
        Gx: Ix, gradient of the brightness in x direction
        Gy: Iy

    Returns:
        dp: delta(p), computed via a least squares method using the pseudo-inverse
    """
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.

    img1_height, img1_width = img1.shape
    img2_height, img2_width = img2.shape
    img2_X = np.arange(0, img2_width)
    img2_Y = np.arange(0, img2_height)

    # Template T_x
    T_x = img1

    # Warp I with W(x;p) to I(W(x;p))
    # generate beginning and ending x and y coordinates of W(x;p)
    M = np.array([[1 + p[0], p[2], p[4]],
                  [p[1], 1 + p[3], p[5]]])  # affine transformation matrix
    img1_begin = np.array([0, 0, 1])  # homogeneous coordinates
    img1_end = np.array([img1_width, img2_height, 1])  # [x, y, 1]
    W_x_p_begin = M @ img1_begin.T  # matrix multiplication
    W_x_p_end = M @ img1_end.T
    # given the range of coordinates, generate meshgrid of all coordinates within the range
    W_x_p_X = np.linspace(W_x_p_begin[0], W_x_p_end[0], img1_width)  # strictly ascending order!
    W_x_p_Y = np.linspace(W_x_p_begin[1], W_x_p_end[1], img1_height)
    W_x_p_Xmesh, W_x_p_Ymesh = np.meshgrid(W_x_p_X, W_x_p_Y)  # warp img1

    # interpolate img2
    img2_spline = RectBivariateSpline(img2_Y, img2_X, img2)
    # evaluate the spline at points, i.e., get corresponding points of the warped img1
    I_W_x_p = img2_spline.ev(W_x_p_Ymesh, W_x_p_Xmesh)
    # Compute error image T_x - I(W(x;p))
    img_error = (T_x - I_W_x_p).reshape(-1, 1)  # (img1_width * img1_height, 1)

    # Warp gradient of I to compute ∇I
    Gx_spline = RectBivariateSpline(img2_Y, img2_X, Gx)
    Gy_spline = RectBivariateSpline(img2_Y, img2_X, Gy)
    W_Gx = Gx_spline.ev(W_x_p_Ymesh, W_x_p_Xmesh)
    W_Gy = Gy_spline.ev(W_x_p_Ymesh, W_x_p_Xmesh)
    nabla_I = np.vstack((W_Gx.ravel(), W_Gy.ravel())).T  # flatten image gradients into 1D, reshape ∇I into (img2_width * img2_height, 2)
    # scale coordinates of image gradients ∇I to be between 0 and 1
    W_G_max = max(W_Gx.max(), W_Gy.max())
    nabla_I /= W_G_max

    # ∇I * Jacobian
    nabla_I_J = np.zeros((img1_width * img1_height, 6))
    for x in range(img1_width):
        for y in range(img1_height):
            J = np.array([[x, 0, y, 0, 1, 0],
                          [0, x, 0, y, 0, 1]])  # Jacobian of the warp (2, 6)
            nabla_I_J[img1_height * x + y] = nabla_I[img1_height * x + y] @ J  # (1, 2) @ (2, 6)
    nabla_I_J *= W_G_max  # undo normalization

    # Compute Hessian H (6, 6)
    H = nabla_I_J.T @ nabla_I_J

    # Compute dp
    # dp is (6, 6) @ (6, img1_width * img1_height) @ (img1_width * img1_height, 1) = (6, 1)
    dp = np.linalg.inv(H) @ nabla_I_J.T @ img_error
    dp = dp.ravel()  # flatten to (1, 6)

    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    """
    Warp the image I(t) using M so that it is registered to I(t + 1), and subtract it from I(t + 1).

    Args:
        img1, img2: input image pair

    Returns:
        mask: binary image of the same size that dictates which pixels are considered to be corresponding to moving objects
    """
    # Sobel filter, size 5
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)  # do not modify this
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)  # do not modify this
    
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.
    p = np.zeros(6)
    epsilon = 0.018  # min search window
    dp_norm = np.inf
    while dp_norm > epsilon:  # reduce error
        dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        p += dp  # update p
        dp_norm = np.linalg.norm(dp)

    img1_height, img1_width = img1.shape
    img2_height, img2_width = img2.shape
    img2_X = np.arange(0, img2_width)
    img2_Y = np.arange(0, img2_height)

    # Template T_x
    T_x = img1

    # Warp I with W(x;p) to I(W(x;p))
    # generate beginning and ending x and y coordinates of W(x;p)
    M = np.array([[1 + p[0], p[2], p[4]],
                  [p[1], 1 + p[3], p[5]]])  # affine transformation matrix
    img1_begin = np.array([0, 0, 1])  # homogeneous coordinates
    img1_end = np.array([img1_width, img2_height, 1])  # [x, y, 1]
    W_x_p_begin = M @ img1_begin.T  # matrix multiplication
    W_x_p_end = M @ img1_end.T
    # given the range of coordinates, generate meshgrid of all coordinates within the range
    W_x_p_X = np.linspace(W_x_p_begin[0], W_x_p_end[0], img1_width)  # strictly ascending order!
    W_x_p_Y = np.linspace(W_x_p_begin[1], W_x_p_end[1], img1_height)
    W_x_p_Xmesh, W_x_p_Ymesh = np.meshgrid(W_x_p_X, W_x_p_Y)  # warp img1

    # interpolate img2
    img2_RBS = RectBivariateSpline(img2_Y, img2_X, img2)
    # evaluate the spline at points, i.e., get corresponding points of the warped img1
    I_W_x_p = img2_RBS.ev(W_x_p_Ymesh, W_x_p_Xmesh)

    # Compute moving image T_x - I(W(x;p))
    moving_image = np.abs(T_x - I_W_x_p)

    th_hi = 0.2 * 256  # you can modify this
    th_lo = 0.15 * 256  # you can modify this

    ### END CODE HERE ###
    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

if __name__ == "__main__":
    # begin_time = datetime.datetime.now()
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        # print(i)
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
    # print("Execution time is:", datetime.datetime.now() - begin_time)
    