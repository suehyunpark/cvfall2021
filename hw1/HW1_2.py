import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils


def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array): length of level + 1
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    G = input_image.copy()
    output = [G]

    for l in range(level):
        G = utils.down_sampling(G)
        output.append(G)

    return output


def laplacian_pyramid(gaussian_pyramid):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    output = []

    for l in range(1, len(gaussian_pyramid)):
        G_expanded = utils.up_sampling(gaussian_pyramid[l])
        DoG = utils.safe_subtract(gaussian_pyramid[l - 1], G_expanded)
        output.append(DoG)

    output.append(gaussian_pyramid[-1])

    return output


def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """
    # Your code
    gaussian1, gaussian2 = gaussian_pyramid(image1, level), gaussian_pyramid(image2, level)
    laplacian1, laplacian2 = laplacian_pyramid(gaussian1), laplacian_pyramid(gaussian2)

    gaussian_mask = gaussian_pyramid(mask, level)

    combined_laplacian = []
    for G_mask, L1, L2 in zip(gaussian_mask, laplacian1, laplacian2):
        L_combined = utils.safe_add(G_mask/255 * L2, (1 - G_mask/255) * L1)  # for weighted sum, divide the mask values by 255
        combined_laplacian.append(L_combined.astype(np.uint8))

    output = combined_laplacian[-1]
    for l in range(len(combined_laplacian)-1, 0, -1):
        output = utils.up_sampling(output)
        output = utils.safe_add(output, combined_laplacian[l - 1])

    return output


if __name__ == '__main__':
    hand = np.asarray(Image.open(os.path.join('images', 'hand.jpeg')).convert('RGB'))
    flame = np.asarray(Image.open(os.path.join('images', 'flame.jpeg')).convert('RGB'))
    mask = np.asarray(Image.open(os.path.join('images', 'mask.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3

    plt.figure()
    plt.imshow(Image.open(os.path.join('images', 'direct_concat.jpeg')))
    plt.axis('off')
    plt.savefig(os.path.join(logdir, 'direct.jpeg'))
    plt.show()

    ret = gaussian_pyramid(hand, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian_pyramid.jpeg'))
        plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis('off')
            plt.savefig(os.path.join(logdir, 'laplacian_pyramid.jpeg'))
            plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'blended.jpeg'))
        plt.show()
