import os
import math
import glob
import numpy as np
from PIL import Image, ImageDraw

# parameters

datadir = './data'
resultdir ='./results'

# you can calibrate these parameters
sigma = 2
highThreshold = 0.12
lowThreshold = 0.02
rhoRes = 1
thetaRes = math.pi/180  # math.pi/180 radian = 1 degree
nLines = 20


def ConvFilter(Igs, G):
    # TODO ...
    """
    Args:
        Igs: grayscale image (2D numpy array)
        G: convolution filter (2D numpy array)

    Returns:
        Iconv: convolved image (Igs convolved with G) of the same size as Igs
    """
    Igs_height, Igs_width = Igs.shape
    G_height, G_width = G.shape

    # Replication Padding
    pad_r, pad_c = G_height // 2, G_width // 2  # top/bottom padding; left/right padding
    Ipadded = np.concatenate((np.tile(Igs[0], (pad_r, 1)), Igs), axis=0)  # padding_top
    Ipadded = np.concatenate((Ipadded, np.tile(Igs[-1], (pad_r, 1))), axis=0)  # padding_bottom
    sliced_col = np.expand_dims(Ipadded[:, 0], axis=1)  # expand sliced column from 1D to 2D
    Ipadded = np.concatenate((np.tile(sliced_col, (1, pad_c)), Ipadded), axis=1)  # padding_left
    sliced_col = np.expand_dims(Ipadded[:, -1], axis=1)  # expand sliced column from 1D to 2D
    Ipadded = np.concatenate((Ipadded, np.tile(sliced_col, (1, pad_c))), axis=1)  # padding_right

    # Convolution
    G = np.fliplr(np.flipud(G))  # flip the kernel in both horizontal and vertical direction
    Iconv = np.zeros_like(Igs)

    for i in range(Igs_height):
        for j in range(Igs_width):
            patch = Ipadded[i:i + G_height, j:j + G_width]
            Iconv[i, j] = np.sum(np.multiply(patch, G))

    return Iconv


def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...
    """
    Args:
        Igs: grayscale image
        sigma: standard deviation of the Gaussian smoothing kernel to be used before edge detection
        highThreshold: constant (if an edge pixel's gradient magnitude is higher than this value, it is marked as a strong edge pixel)
        lowThreshold: constant (if an edge pixel's gradient magnitude is lower than this value, it will be suppressed)
        * any edge pixels whose gradient magnitude is in between highThreshold and lowThreshold will be marked as a weak edge pixel
        * a weak edge pixel will have at least one strong edge pixel among its neighboring pixels

    Returns:
        Im: edge magnitude image
        Io: edge orientation image
        Ix, Iy: edge filter responses in the x and y directions, respectively
    """
    # 1. Smooth image with the specified Gaussian kernel
    size = int(sigma * 6) + 1 # Reference: https://en.wikipedia.org/wiki/Gaussian_blur#:~:text=Typically%2C%20an%20image,entire%20Gaussian%20distribution.
    G = np.fromfunction(lambda x, y: 1 / (2*np.pi*sigma**2) * np.exp(-1 * ((x-(size-1)/2-1)**2 + (y-(size-1)/2-1)**2) / (2*sigma**2)), (size, size))  # Reference: https://en.wikipedia.org/wiki/Canny_edge_detector#:~:text=The%20equation%20for%20a%20Gaussian%20filter%20kernel%20of%20size%20(2k%2B1)%C3%97(2k%2B1)%20is%20given%20by%3A
    G = G / np.sum(G)  # normalize Gaussian kernel
    Ig = ConvFilter(Igs, G)

    # 2. Find gradient magnitude and direction for each pixel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)  # Sobel (3 x 3) is the default edge operator
    Ix = ConvFilter(Ig, sobel_x)

    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    Iy = ConvFilter(Ig, sobel_y)

    Im = np.hypot(Ix, Iy)
    Io = np.arctan2(Iy, Ix)  # array of angles in radians, in the range [-pi, pi]

    '''
    Im = Im / Im.max() * 255
    image = Image.fromarray(Im).convert("L")
    image.show()
    image.save(img_name + "_magnitude_original.jpeg")
    '''

    # 3. Run non-maximum suppression: locate edges by finding zero-crossings along the edge normal directions
    # range of Im: [0, 1]
    Im_height, Im_width = Im.shape
    Im_nms = np.zeros_like(Im)
    Io_deg = np.rad2deg(Io)  # array of angles in degrees, in the range [-180., 180.]
    Io_deg[Io_deg < 0] += 180  # reduced the range to [0., 180.], to remove redundant comparison of angles

    for i in range(1, Im_height - 1):
        for j in range(1, Im_width - 1):  # point q is (i, j)
            if 0 <= Io_deg[i, j] < 22.5 or 157.5 <= Io_deg[i, j] <= 180:
                p_x, p_y = i, j - 1
                r_x, r_y = i, j + 1
            elif 22.5 <= Io_deg[i, j] < 67.5:
                p_x, p_y = i - 1, j + 1
                r_x, r_y = i + 1, j - 1
            elif 67.5 <= Io_deg[i, j] < 112.5:
                p_x, p_y = i - 1, j
                r_x, r_y = i + 1, j
            elif 112.5 <= Io_deg[i, j] < 157.5:
                p_x, p_y = i - 1, j - 1
                r_x, r_y = i + 1, j + 1

            if Im[i, j] >= max(Im[p_x, p_y], Im[r_x, r_y]):
                Im_nms[i, j] = Im[i, j]
            # else: Im_nms is already initialized with zeros
    '''
    Im_nms = Im_nms / Im_nms.max() * 255
    image = Image.fromarray(Im_nms).convert("L")
    image.show()
    image.save(img_name + "_magnitude_afterNMS.jpeg")
    '''
    Im = Im_nms  # non-maximum suppression complete; again normalize for double thresholding

    # 4. Double thresholding
    Im_dt = np.zeros_like(Im)
    strong_x, strong_y = np.where(Im >= highThreshold)
    weak_x, weak_y = np.where(np.logical_and(Im >= lowThreshold, Im < highThreshold))
    Im_dt[strong_x, strong_y] = highThreshold
    Im_dt[weak_x, weak_y] = lowThreshold

    # 5. Edge tracking by hysteresis - connect any weak edges neighboring strong edges
    dx = np.array([-1, -1, -1, 0, 1, 1, 1, 0])  # inspecting in clockwise direction, starting from the upper left neighboring pixel
    dy = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
    idx = 0
    while idx < len(strong_x):
        x, y = strong_x[idx], strong_y[idx]
        for direction in range(len(dx)):
            n_x = x + dx[direction]  # neighboring pixel
            n_y = y + dy[direction]
            if 0 <= n_x < Im_height and 0 <= n_y < Im_width and Im_dt[n_x, n_y] == lowThreshold:
                Im_dt[n_x, n_y] = highThreshold
                np.append(strong_x, n_x)  # add new indices of strong edges
                np.append(strong_y, n_y)
        idx += 1
    Im_dt[Im_dt != highThreshold] = 0  # clean up remaining weak edges by setting them to zero
    Im_dt[Im_dt > 0] = 1
    '''
    Im_dt = Im_dt / Im_dt.max() * 255
    image = Image.fromarray(Im_dt).convert("L")
    image.show()
    image.save(img_name + "_magnitude_afterDT.jpeg")
    Im_dt /= 255
    '''
    Im = Im_dt  # double hysteresis-based thresholding complete

    return Im, Io, Ix, Iy


def HoughTransform(Im, rhoRes, thetaRes):
    # TODO ...
    """
    Args:
        Im: edge magnitude image
        rhoRes: resolution of the Hough transform accumulator along the rho axis (rho: distance from origin to the line)
        thetaRes:  resolution of the Hough transform accumulator along the theta axis (theta: angle from origin to the line. [-90° to 90°])

    Returns:
        H: Hough transform accumulator that contains the number of 'votes' for all the possible lines passing through the image
    """

    Im_height, Im_width = Im.shape
    Im_diagonal = int(np.ceil(np.hypot(Im_height, Im_width)))  # max value of rho
    rhos = np.arange(-Im_diagonal, Im_diagonal + 1, rhoRes)
    thetas = np.deg2rad(np.arange(-90, 90, thetaRes * 180/math.pi))  # in radian

    cos_theta, sin_theta = np.cos(thetas), np.sin(thetas)

    H = np.zeros((len(rhos), len(thetas)))
    for x in range(Im_width):  # changing notation to match the dimension of the image space
        for y in range(Im_height):
            if Im[y, x] == 1:  # point (y, x) is an edge
                for t in range(len(thetas)):
                    rho = int((x * cos_theta[t] + y * sin_theta[t]) + Im_diagonal) // rhoRes  # max value of rho is added for a positive index
                    H[rho, t] += 1  # accumulate in edge pixels

    '''
    H = H / H.max() * 255
    image = Image.fromarray(H).convert("L")
    image.show()
    image.save(img_name + "_accumulator.jpeg")
    H /= 255
    '''
    return H


def HoughLines(H, rhoRes, thetaRes, nLines):
    # TODO ...
    """
    Args:
        H: Hough transform accumulator
        rhoRes: accumulator resolution parameter
        thetaRes: accumulator resolution parameter
        nLines: number of lines to return

    Returns:
        lRho: nLines arrays that contain the rho parameter of the lines found in the image
        lTheta: nLines arrays that contain the theta parameter of the lines found in the image
        (rho and theta values for the n highest scoring cells in the Hough accumulator)
    """
    lRho, lTheta = [], []
    H_height, H_width = H.shape
    d_rho = np.array([-1, -1, -1, 0, 1, 1, 1, 0])  # inspecting in clockwise direction, starting from the upper left neighboring pixel
    d_theta = np.array([-1, 0, 1, 1, 1, 0, -1, -1])

    # cnt = 0
    for i in range(nLines):
        max_idx = np.argmax(H)  # find index of the highest vote, note that this is a linear index assuming H_copy is a flattened array
        max_vote = np.unravel_index(max_idx, H.shape)  # get the 2D index (coordinate) of the where the computed linear index points to
        rho, theta = max_vote

        for direction in range(len(d_rho)):
            n_rho = rho + d_rho[direction]  # neighbor
            n_theta = theta + d_theta[direction]
            if 0 <= n_rho < H_height and 0 <= n_theta < H_width:  # boundary check
                H[rho, theta] = 0  # suppress
        lRho.append(rho)
        lTheta.append(theta)

    lRho = np.array(lRho)
    lTheta = np.array(lTheta)  # these denote plain coordinates in the Hough space
    lTheta = np.deg2rad(lTheta * (thetaRes * 180/math.pi) - 90)  # center theta and convert into radian
    # Note: thetas = np.deg2rad(np.arange(-90, 90, thetaRes * 180/math.pi))

    return lRho, lTheta


def HoughLineSegments(lRho, lTheta, Im):
    # TODO ...
    """
    Args:
        lRho: nLines arrays that contain the rho parameter of the lines found in the image
        lTheta: nLines arrays that contain the theta parameter of the lines found in the image
        Im: edge magnitude image

    Returns:
        l: dictionary structure containing the pixel locations of the start and end points of each line segment in the image
        { index of line segment: { 'start': (x_s, y_s), 'end': (x_e, x_y) } }
    """
    Im_height, Im_width = Im.shape
    Im_diagonal = int(np.ceil(np.hypot(Im_height, Im_width)))  # max value of rho
    l = {}

    for k in range(nLines):
        # Plot one Hough line
        line = np.zeros_like(Im)
        img_line = Image.fromarray(line)
        draw = ImageDraw.Draw(img_line)
        rho, theta = lRho[k], lTheta[k]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        if theta != 0:
            x1 = 0
            y1 = int((rho * rhoRes - Im_diagonal - x1 * cos_theta) / sin_theta)  # reverting in the Hough line equation
            x2 = Im_width - 1
            y2 = int((rho * rhoRes - Im_diagonal - x2 * cos_theta) / sin_theta)
            draw.line([(x1, y1), (x2, y2)], fill=255, width=1)
        else:  # when sin_theta = 0, which prompts ZeroDivisionError
            x = int((rho * rhoRes - Im_diagonal) / cos_theta)  # use the other coordinate instead
            draw.line([(0, x), (Im_width - 1, x)], fill=255, width=1)

        line_x, line_y = np.nonzero(np.array(img_line))
        start_candidates, end_candidates = [], []
        end_segment = False
        for i in range(len(line_x)):
            if Im[line_x[i], line_y[i]] == 1:  # magnitude defines the object border
                if not end_segment:
                    start_candidates.append([line_y[i], line_x[i]])
                    end_segment = True
            else:  # some gap in between
                if end_segment:
                    end_candidates.append([line_y[i], line_x[i]])
                    end_segment = False
        if end_segment:  # dangling start_candidate. possibly reached the border of image!
            end_candidates.append([line_y[-1], line_x[-1]])

        # Merge short segments
        idx = 0
        max_line_gap = 11
        while idx < len(end_candidates) - 1:
            distance = math.dist(end_candidates[idx], start_candidates[idx + 1])
            if distance <= max_line_gap:
                del end_candidates[idx]
                del start_candidates[idx + 1]
            else:
                idx += 1

        # Find longest segment
        distances = []
        for i in range(len(start_candidates)):
            distance = math.dist(start_candidates[i], end_candidates[i])
            distances.append(distance)

        idx_max_length = np.argmax(distances)
        segment_dict = dict(start=tuple(start_candidates[idx_max_length]), end=tuple(end_candidates[idx_max_length]))
        l[k] = segment_dict

    return l


def main():
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")
        img_original = Image.open(img_path).convert("RGB")
        # print(img_path)
        global img_name
        img_name = img_path[-9:-4]

        Igs = np.array(img)
        Igs = Igs / 255.  # float range in [0, 1]

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)
        H = HoughTransform(Im, rhoRes, thetaRes)
        lRho, lTheta = HoughLines(H, rhoRes, thetaRes, nLines)

        # Plot HoughLines
        img_lines = img_original.copy()
        draw = ImageDraw.Draw(img_lines)
        Im_height, Im_width = Im.shape
        Im_diagonal = int(np.ceil(np.hypot(Im_height, Im_width)))  # max value of rho

        for k in range(nLines):
            rho, theta = lRho[k], lTheta[k]
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)

            if theta != 0:
                x1 = 0
                y1 = int((rho * rhoRes - Im_diagonal - x1 * cos_theta) / sin_theta)  # reverting in the Hough line equation
                x2 = Im_width - 1
                y2 = int((rho * rhoRes - Im_diagonal - x2 * cos_theta) / sin_theta)
                draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=1)
            else:  # when sin_theta = 0, which prompts ZeroDivisionError
                x = int((rho * rhoRes - Im_diagonal) / cos_theta)  # use the other coordinate instead
                draw.line([(0, x), (Im_width - 1, x)], fill=(0, 255, 0), width=1)

        # img_lines.show()
        # img_lines.save(img_name + "_HoughLines.jpeg")

        l = HoughLineSegments(lRho, lTheta, Im)
        # Plot Hough line segments
        img_segments = img_original.copy()
        draw = ImageDraw.Draw(img_segments)

        for k in range(nLines):
            draw.line([l[k]['start'], l[k]['end']], fill=(0, 255, 0), width=2)

        # img_segments.show()
        # img_segments.save(img_name + "_HoughLineSegments.jpeg")

        # Saves the outputs to files
        # Im, H, img_original + hough line , img_original + hough line segments
        logdir = os.path.join('results', img_name)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        Im = Im / Im.max() * 255
        image = Image.fromarray(Im).convert("L")
        image.save(os.path.join(logdir, img_name + "_Im.jpeg"))

        H = H / H.max() * 255
        image = Image.fromarray(H).convert("L")
        image.save(os.path.join(logdir, img_name + "_H.jpeg"))

        img_lines.save(os.path.join(logdir, img_name + "_HoughLines.jpeg"))

        img_segments.save(os.path.join(logdir, img_name + "_HoughLineSegments.jpeg"))


if __name__ == '__main__':
    main()