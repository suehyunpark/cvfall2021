import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """

    # Your code
    pad_r, pad_c = size[0] // 2, size[1] // 2  # upper/lower margin; left/right margin

    output = np.concatenate((input_image[pad_r:0:-1, ...], input_image), axis=0)  # upper padding
    output = np.concatenate((output, output[-2:-2-pad_r:-1, ...]), axis=0)  # lower padding
    output = np.concatenate((output[:, pad_c:0:-1, :], output), axis=1)  # left padding
    output = np.concatenate((output, output[:, -2:-2-pad_c:-1, :]), axis=1)  # right padding

    return output


def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """

    # Your code
    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image
    padded_image = reflect_padding(input_image, Kernel.shape)  # source pixel
    kernel = np.fliplr(np.flipud(Kernel))  # flip the kernel in both horizontal and vertical direction

    img_height, img_width, _ = input_image.shape
    ker_height, ker_width = kernel.shape
    output = np.zeros_like(input_image)

    for i in range(img_height):
        for j in range(img_width):
            patch = padded_image[i:i+ker_height, j:j+ker_width]
            for k in range(3):  # RGB channel
                rgb_patch = patch[..., k]
                output[i, j, k] = np.sum(np.multiply(rgb_patch, kernel))

    return output


def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")

    # Your code
    padded_image = reflect_padding(input_image, size)
    img_height, img_width, _ = input_image.shape
    output = np.zeros_like(input_image)

    for i in range(img_height):
        for j in range(img_width):
            patch = padded_image[i:i+size[0], j:j+size[1]]
            for k in range(3):
                rgb_patch = patch[..., k]
                output[i, j, k] = np.median(rgb_patch)  # find median of sorted numpy array (rgb_patch)

    return output


def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X direction
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """
    # Your code
    padded_image = reflect_padding(input_image, size)
    img_height, img_width, _ = input_image.shape
    filter_height, filter_width = size
    pad_r, pad_c = filter_height // 2, filter_width // 2

    gaussian_1d_x = np.fromfunction(lambda x: 1 / (np.sqrt(2*np.pi) * sigmax) * np.exp(-x**2 / (2*sigmax**2)), (size[1],))
    gaussian_1d_y = np.fromfunction(lambda x: 1 / (np.sqrt(2*np.pi) * sigmay) * np.exp(-x**2 / (2*sigmay**2)), (size[0],))  # using math module creates errors
    x_norm, y_norm = np.sum(gaussian_1d_x), np.sum(gaussian_1d_y)
    gaussian_1d_x, gaussian_1d_y = gaussian_1d_x / x_norm, gaussian_1d_y / y_norm  # normalize the kernel to maintain adequate intensity
    # reference: https://stackoverflow.com/questions/61354389/when-applying-gaussian-filter-the-image-becomes-dark

    # convolve in x-direction
    output_mid = np.zeros_like(input_image)
    for i in range(img_height):
        for j in range(img_width):
            patch = padded_image[i+pad_r, j:j+filter_width]  # 1D array with filter_width at each row
            for k in range(3):
                rgb_patch = patch[..., k]
                output_mid[i, j, k] = np.sum(np.multiply(rgb_patch, gaussian_1d_x))
    output_mid = output_mid.astype(np.uint8)

    # convolve in y-direction
    output_mid = reflect_padding(output_mid, size)  # get padded values again
    output_fin = np.zeros_like(input_image)
    for i in range(img_height):
        for j in range(img_width):
            patch = output_mid[i:i+filter_height, j+pad_c]  # 1D array with filter_height at each column
            for k in range(3):
                rgb_patch = patch[..., k]
                output_fin[i, j, k] = np.sum(np.multiply(rgb_patch, gaussian_1d_y))
    output_fin = output_fin.astype(np.uint8)

    return output_fin


if __name__ == '__main__':
    image = np.asarray(Image.open(os.path.join('images', 'baboon.jpeg')).convert('RGB'))
    # image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    # image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_1')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((5, 5)) / 25.
    sigmax, sigmay = 5, 5
    ret = reflect_padding(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'reflect.jpeg'))
        plt.show()

    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'convolve.jpeg'))
        plt.show()

    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'median.jpeg'))
        plt.show()

    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian.jpeg'))
        plt.show()