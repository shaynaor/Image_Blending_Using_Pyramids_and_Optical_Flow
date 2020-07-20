import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def convDerivative(img: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate derivative in x and y direction of an image
    :param img: image
    :return: (x_der,y_der)
    """
    ker_x = np.array([[-1, 0, 1]])
    ker_y = ker_x.reshape((3, 1))
    return cv2.filter2D(img, -1, ker_x), cv2.filter2D(img, -1, ker_y)


def is_solvable(A: np.ndarray) -> bool:
    """
    :param A: matrix to calculate eigvals on (A*A^T)
    :return: true iff lambda_1, lambda_2 > 1 , and  (lamda_2 / lamda_1) < 100
    """
    M = np.dot(np.transpose(A), A)
    lamda_1, lamda_2 = np.sort(np.linalg.eigvals(M))
    if lamda_1 > 1 and (lamda_2 / lamda_1) < 100:
        return True
    return False


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    I_x, I_y = convDerivative(blurImage2(im1, 5))
    I_t = np.subtract(im2, im1)
    height, width = im2.shape[:2]
    pad_len, patch_height = win_size // 2, win_size ** 2
    u_v_list, y_x_list = [], []
    for i in range(step_size, height, step_size):
        for j in range(step_size, width, step_size):
            patch_x = I_x[i - pad_len: i + pad_len + 1, j - pad_len: j + pad_len + 1]
            patch_y = I_y[i - pad_len: i + pad_len + 1, j - pad_len: j + pad_len + 1]
            patch_t = I_t[i - pad_len: i + pad_len + 1, j - pad_len: j + pad_len + 1]
            b = (-1) * patch_t.reshape(patch_height, 1)
            A = np.hstack((patch_x.reshape(patch_height, 1), patch_y.reshape(patch_height, 1)))
            if is_solvable(A):
                y_x_list.append((j, i))
                d = np.dot(np.linalg.pinv(A), b)
                u_v_list.append(d)

    return np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2)


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    return cv2.filter2D(in_image, -1, get_gaussian2D(kernel_size, get_gaus_sigma(kernel_size)))


def get_gaus_sigma(kernel_size: int):
    """
    :param kernel_size: size of kernel
    :return: value of sigma factor in gaussian filter
    """
    return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8


def get_gaussian2D(size: int, sigma: float) -> np.ndarray:
    """
    :param size: kernel size, It should be odd.
    :param sigma: gaussian standard deviation.
    :return: gaussian 2D filter in shape ksizexksize
    """
    gaus_kernel = cv2.getGaussianKernel(size, sigma)
    return np.dot(gaus_kernel, np.transpose(gaus_kernel))


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaus_pyr = gaussianPyr(img, levels)
    gaus_ker = get_gaussian2D(5, get_gaus_sigma(5))
    lap_pyr = [gaus_pyr[i] - gaussExpand(gaus_pyr[i + 1], gaus_ker) for i in range(len(gaus_pyr) - 1)]
    lap_pyr.append(gaus_pyr[-1])
    return lap_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gaus_ker = get_gaussian2D(5, get_gaus_sigma(5))
    gaus_img = gaussExpand(lap_pyr[len(lap_pyr) - 1], gaus_ker) + lap_pyr[len(lap_pyr) - 2]
    for i in range(len(lap_pyr) - 2, 0, -1):
        gaus_img = gaussExpand(gaus_img, gaus_ker) + lap_pyr[i - 1]
    return gaus_img


def gaussReduce(img: np.ndarray) -> np.ndarray:
    """
    define reduce operation:
    1. blurring the image.
    2. reduce image width and height by 2.
    :param img: input image.
    :return: reduced image.
    """
    return blurImage2(img, 5)[::2, ::2]


def get_proper_size(height: int, width: int, levels: int) -> (int, int):
    """
    Each level in the pyramids, the image shape is cut in half, so for x levels, crop the initial image to
    2^x · floor(img size /2^x )
    :param height: image height
    :param width: image width
    :param levels: Pyramid levels
    :return: 2^levels · floor(img size /2^levels )
    """
    size = np.power(2, levels)
    return int(size * int(height / size)), int(size * int(width / size))


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    height, width = get_proper_size(img.shape[0], img.shape[1], levels)
    pyramid_list = [img[:height, :width]]
    for i in range(1, levels + 1):
        pyramid_list.append(gaussReduce(pyramid_list[i - 1]))
    return pyramid_list


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    if len(img.shape) == 3:
        height, width, channels = img.shape
        expand_img = np.zeros((height * 2, width * 2, channels))
        expand_img[::2, ::2, :] = img
    else:
        height, width = img.shape
        expand_img = np.zeros((height * 2, width * 2))
        expand_img[::2, ::2] = img

    return cv2.filter2D(expand_img, -1, gs_k) * 4


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    lap_pyr_img_1, lap_pyr_img_2 = laplaceianReduce(img_1, levels), laplaceianReduce(img_2, levels)
    gaus_pyr_mask = gaussianPyr(mask, levels)
    blended_pyr_list = [gaus_pyr_mask[i] * lap_pyr_img_1[i] + (1 - gaus_pyr_mask[i]) * lap_pyr_img_2[i]
                        for i in range(levels + 1)]

    height, width = get_proper_size(img_1.shape[0], img_1.shape[1], levels)
    naive_blend = mask[:height, :width] * img_1[:height, :width] + (1 - mask[:height, :width]) * img_2[:height, :width]
    return naive_blend, laplaceianExpand(blended_pyr_list)
