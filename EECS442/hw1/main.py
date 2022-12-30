"""
Starter code for EECS 442 W21 HW1
"""
from numpy.lib.twodim_base import tri
from util import generate_gif, renderCube
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib
from numpy.ma.core import cos
matplotlib.use('agg')


def rotX(theta):
    """
    Generate 3D rotation matrix about X-axis
    Input:  theta: rotation angle about X-axis
    Output: Rotation matrix (3 x 3 array)
    """
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def rotY(theta):
    """
    Generate 3D rotation matrix about Y-axis
    Input:  theta: rotation angle along y-axis
    Output: Rotation matrix (3 x 3 array)
    """
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def part1():
    # TODO: Solution for Q1
    # Task 1: Use rotY() to generate cube.gif
    # generate_gif([rotX(0)], file_name='cube1.gif')
    # Task 2:  Use rotX() and rotY() sequentially to check
    # the commutative property of Rotation Matrices
    # generate_gif([rotY(np.pi/2)], file_name='cube2.gif')
    # Task 3: Combine rotX() and rotY() to render a cube
    # projection such that end points of diagonal overlap
    # Hint: Try rendering the cube with multiple configrations
    # to narrow down the search region
    pass


def split_triptych(trip):
    """
    Split a triptych into thirds
    Input:  trip: a triptych (H x W matrix)
    Output: R, G, B martices
    """
    # TODO: Split a triptych into thirds and
    # return three channels as numpy arrays
    # mid_j, mid_i = int(trip.shape[0]/2), int(trip.shape[1]/2)
    # i_left = i_right = mid_i
    # while trip[mid_j, i_left] != 255:
    #     i_left -= 1
    # while trip[mid_j, i_right] != 255:
    #     i_right += 1
    # j_up = j_down = mid_j
    # while trip[j_up, mid_i] != 255:
    #     j_up -= 1
    # while trip[j_down, mid_i] != 255:
    #     j_down += 1
    # i_left += 20
    # i_right -= 20
    i_left = j_up = 0
    i_right = trip.shape[1]
    j_down = trip.shape[0]
    M = int((j_down-j_up-1-80)/3)
    B = trip[j_up+1+20:j_up+1+M+20, i_left+1:i_right]
    G = trip[j_up+1+M+40:j_up+1+2*M+40, i_left+1:i_right]
    R = trip[j_up+1+2*M+60:j_up+1+3*M+60, i_left+1:i_right]
    return R, G, B


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    """
    return np.sum(((ch1-ch1.mean())/ch1.std())*((ch2-ch2.mean())/ch2.std()))


def best_offset(ch1, ch2, metric, Xrange=np.arange(-10, 10),
                Yrange=np.arange(-10, 10)):
    """
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
            metric: similarity measure between two channels
            Xrange: range to search for optimal offset in vertical direction
            Yrange: range to search for optimal offset in horizontal direction
    Output: optimal offset for X axis and optimal offset for Y axis

    Note: Searching in Xrange would mean moving in the vertical 
    axis of the image/matrix, Yrange is the horizontal axis 
    """
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets
    best_x = 0
    best_y = 0
    best_ncc = normalized_cross_correlation(ch1, ch2)
    for i in Xrange:
        for j in Yrange:
            if normalized_cross_correlation(ch1, np.roll(np.roll(ch2, j, axis=1), i, axis=0)) > best_ncc:
                best_x, best_y = i, j
    return best_x, best_y


def align_and_combine(R, G, B, metric):
    """
    Input:  R: red channel
            G: green channel
            B: blue channel
            metric: similarity measure between two channels
    Output: aligned RGB image 
    """
    # TODO: Use metric to align the three channels
    # Hint: Use one channel as the anchor to align other two
    return np.dstack((R, np.roll(np.roll(G, best_j_G, axis=1), best_i_G, axis=0),
                      np.roll(np.roll(B, best_j_B, axis=1), best_i_B, axis=0)))


def pyramid_align():
    # TODO: Reuse the functions from task 2 to perform the
    # image pyramid alignment iteratively or recursively
    pass


def part2():
    # TODO: Solution for Q2
    # Task 1: Generate a colour image by splitting
    # the triptych image and save it

    # img = plt.imread('00153v.jpg')
    # R, G, B = split_triptych(img)
    # plt.imsave('00153v_R.jpg', R)
    # plt.imsave('00153v_G.jpg', G)
    # plt.imsave('00153v_B.jpg', B)
    # img_out = np.dstack((R, G, B))
    # plt.imsave('00153v_trivial.jpg', img_out)
    # # Task 2: Remove misalignment in the colour channels
    # # by calculating best offset
    # best_i_G = 0
    # best_j_G = 0
    # best_i_B = 0
    # best_j_B = 0
    # best_ncc = normalized_cross_correlation(R, G)
    # for i in range(-15, 15):
    #     for j in range(-15, 15):
    #         if normalized_cross_correlation(R, np.roll(np.roll(G, j, axis=1), i, axis=0)) > best_ncc:
    #             best_i_G, best_j_G = i, j
    #             best_ncc = normalized_cross_correlation(
    #                 R, np.roll(np.roll(G, j, axis=1), i, axis=0))
    # best_ncc = normalized_cross_correlation(R, B)
    # for i in range(-15, 15):
    #     for j in range(-15, 15):
    #         if normalized_cross_correlation(R, np.roll(np.roll(B, j, axis=1), i, axis=0)) > best_ncc:
    #             best_i_B, best_j_B = i, j
    #             best_ncc = normalized_cross_correlation(
    #                 R, np.roll(np.roll(B, j, axis=1), i, axis=0))
    # best_ncc_img = np.dstack((R, np.roll(np.roll(G, best_j_G, axis=1), best_i_G, axis=0), np.roll(
    #     np.roll(B, best_j_B, axis=1), best_i_B, axis=0)))
    # plt.imsave('00153v_ncc.jpg', best_ncc_img)
    # print(best_i_B, best_j_B)
    # print(best_i_G, best_j_G)
    # Task 3: Pyramid alignment
    pass


def part3():
    # TODO: Solution for Q3
    img1_rgb = cv2.imread('indoor.png')
    R, G, B = cv2.split(img1_rgb)
    plt.imsave('indoor_R.png', R, cmap='gray')
    plt.imsave('indoor_G.png', G, cmap='gray')
    plt.imsave('indoor_B.png', B, cmap='gray')

    img1_lab = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2Lab)
    L, a, b = cv2.split(img1_lab)
    plt.imsave('indoor_lab_L.png', L, cmap='gray')
    plt.imsave('indoor_lab_a.png', a, cmap='gray')
    plt.imsave('indoor_lab_b.png', b, cmap='gray')

    img2_rgb = cv2.imread('outdoor.png')
    R, G, B = cv2.split(img2_rgb)
    plt.imsave('outdoor_R.png', R, cmap='gray')
    plt.imsave('outdoor_G.png', G, cmap='gray')
    plt.imsave('outdoor_B.png', B, cmap='gray')

    img2_lab = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2Lab)
    L, a, b = cv2.split(img2_lab)
    plt.imsave('outdoor_lab_L.png', L, cmap='gray')
    plt.imsave('outdoor_lab_a.png', a, cmap='gray')
    plt.imsave('outdoor_lab_b.png', b, cmap='gray')
    pass


def main():
    part1()
    part2()
    part3()


if __name__ == "__main__":
    main()
