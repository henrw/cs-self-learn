import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import imag
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    H = image.shape[0]
    W = image.shape[1]
    if u < 0:
        image = image[::-1, :]
    if v < 0:
        image = image[:, ::-1]
    image = np.vstack((np.hstack((image, np.zeros((H, abs(v))))), np.zeros(
        (abs(u), W+abs(v)))))
    difference2 = (
        image[abs(u):, abs(v):]-image[:image.shape[0]-abs(u), :image.shape[1]-abs(v)])**2
    if u < 0:
        difference2 = difference2[::-1, :]
    if v < 0:
        difference2 = difference2[:, ::-1]
    output = np.zeros((H, W))
    difference2 = np.vstack((np.hstack((difference2, np.zeros((difference2.shape[0], window_size[1])))), np.zeros(
        (window_size[0], difference2.shape[1]+window_size[1]))))
    for i in range(H):
        for j in range(W):
            output[i, j] = np.sum(
                difference2[i:i+window_size[0], j:j+window_size[1]])
    return output


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    kx = np.array([[1, 0, -1]])
    ky = np.array([[1], [0], [-1]])
    Ix = scipy.ndimage.convolve(image, kx, mode="constant")/2
    Iy = scipy.ndimage.convolve(image, ky, mode="constant")/2

    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy

    d = window_size[0]//2
    # For each image location, construct the structure tensor and calculate
    # the Harris response

    sum_filter = np.ones((window_size[0], window_size[1]))
    # Ixx=np.pad(Ixx,((d, d), (d, d)),constant_values=0)
    # Iyy=np.pad(Iyy,((d, d), (d, d)),constant_values=0)
    # Ixy=np.pad(Ixy,((d, d), (d, d)),constant_values=0)
    # [d:d+image.shape[0],d:d+image.shape[1]]
    M00 = scipy.ndimage.convolve(Ixx, sum_filter, mode="constant")
    # [d:d+image.shape[0],d:d+image.shape[1]]
    M11 = scipy.ndimage.convolve(Iyy, sum_filter, mode="constant")
    # [d:d+image.shape[0],d:d+image.shape[1]]
    M01 = M10 = scipy.ndimage.convolve(Ixy, sum_filter, mode="constant")
    response = np.zeros((image.shape[0], image.shape[1]))
    alpha = 0.05
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            M = np.array([[M00[i, j], M01[i, j]], [M10[i, j], M11[i, j]]])
            response[i, j] = np.linalg.det(M)-alpha*(M[0, 0]+M[1, 1])**2
    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 5: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    # u, v, W = 5, 5, (5, 5)
    # for u, v in [(0, 5), (0, -5), (5, 0), (-5, 0)]:
    #     score = corner_score(img, u, v, W)
    #     save_img(score, "./feature_detection/corner_score_" +
    #              str(u)+"_"+str(v)+".png")

    # (c): No Code

    # -- TODO Task 6: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    plt.imsave('./feature_detection/harris_response_heat.png',
               harris_corners, cmap='autumn')
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
