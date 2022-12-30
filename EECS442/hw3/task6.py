"""
Task6 Code
"""
import numpy as np
import common
from common import save_img, read_img
from homography import fit_homography, homography_transform, RANSAC_fit_homography
import os
import cv2


def compute_distance(desc1, desc2):
    '''
    Calculates L2 distance between 2 binary descriptor vectors.

    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)

    Output - dist: a (N,M) L2 distance matrix where dist(i,j)
             is the squared Euclidean distance between row i of
             desc1 and desc2. You may want to use the distance
             calculation trick
             ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''
    dist = (np.sum(desc1**2, axis=1, keepdims=True) +
            np.sum(desc2**2, axis=1, keepdims=True).T)-2*np.dot(desc1, desc2.T)
    dist[dist < 0] = 0
    return np.sqrt(dist)


def find_matches(desc1, desc2, ratioThreshold):
    '''
    Calculates the matches between the two sets of keypoint
    descriptors based on distance and ratio test.

    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
            ratioThreshhold : maximum acceptable distance ratio between 2
                              nearest matches

    Output - matches: a list of indices (i,j) 1 <= i <= N, 1 <= j <= M giving
             the matches between desc1 and desc2.

             This should be of size (K,2) where K is the number of
             matches and the row [ii,jj] should appear if desc1[ii,:] and
             desc2[jj,:] match.
    '''
    matches = []
    dist = compute_distance(desc1, desc2)
    sorted_index_0 = np.argsort(dist, axis=0)
    sorted_index_1 = np.argsort(dist, axis=1)

    sorted_dist_0 = np.take_along_axis(dist, sorted_index_0, axis=0)
    for j in range(dist.shape[1]):
        if sorted_dist_0[0][j]/sorted_dist_0[0][j] < ratioThreshold and (sorted_index_0[0][j], j) not in matches:
            matches.append([sorted_index_0[0][j], j])
    sorted_dist_1 = np.take_along_axis(dist, sorted_index_1, axis=1)
    for i in range(dist.shape[0]):
        if sorted_dist_1[i][0]/sorted_dist_1[i][1] < ratioThreshold and (i, sorted_index_1[i][0]) not in matches:
            matches.append([i, sorted_index_1[i][0]])

    return matches


def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    Creates an output image where the two source images stacked vertically
    connecting matching keypoints with a line. 

    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            kp1: Keypoint matrix for image 1 of shape (N,4)
            kp2: Keypoint matrix for image 2 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of 
                     keypoints (K,2)

    Output - Image where 2 input images stacked vertically with lines joining 
             the matched keypoints

    Hint: see cv2.line
    '''
    # Hint:
    # Use common.get_match_points() to extract keypoint locations
    H1 = img1.shape[0]
    H2 = img2.shape[0]
    W = max(img1.shape[1], img2.shape[2])
    output = np.zeros((H1+H2, W, 3))
    output[0:H1, 0:img1.shape[1], :] = img1
    output[H1:, 0:img2.shape[1], :] = img2
    key_location = common.get_match_points(kp1, kp2, matches)
    print("key_location", key_location)
    for XYpair in key_location:
        output = cv2.line(output, (int(XYpair[0]), int(XYpair[1])), (int(
            XYpair[2]), int(XYpair[3])+H1), color=(255, 0, 0), thickness=10)
    return output


def warp_and_combine(img1, img2, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.

    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.

    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them

    Output - V: stitched image of size (?,?,3); unknown since it depends on H
    '''
    transformed_pix_X = []
    transformed_pix_Y = []
    transformed_pix_intensity = []
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            this_XY = np.array([[j, i]])
            transformed_XY = homography_transform(this_XY, H)
            transformed_pix_Y.append(transformed_XY[0][1])
            transformed_pix_X.append(transformed_XY[0][0])
            transformed_pix_intensity.append(img2[i, j])
    offsetY = -min(0, int(min(transformed_pix_Y)))
    offsetX = -min(0, int(min(transformed_pix_X)))
    newH = max(img1.shape[0], int(max(transformed_pix_Y)))+offsetY
    newW = max(img1.shape[1], int(max(transformed_pix_X)))+offsetX
    V = np.zeros((int(newH), int(newW), 3))
    V[offsetY:offsetY+img1.shape[0], offsetX:offsetX+img1.shape[1]] = img1
    for i in range(img2.shape[0]*img2.shape[1]):
        V[offsetY+int(transformed_pix_Y[i]-1), offsetX +
          int(transformed_pix_X[i])-1] = transformed_pix_intensity[i]
    return V


def make_warped(img1, img2):
    '''
    Take two images and return an image, putting together the full pipeline.
    You should return an image of the panorama put together.


    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 1 of shape (H2,W2,3)

    Output - Final stitched image

    Be careful about:
    a) The final image size 
    b) Writing code so that you first estimate H and then merge images with H.
    The system can fail to work due to either failing to find the homography or
    failing to merge things correctly.
    '''
    kps1 = common.get_AKAZE(img1)[0]
    desc1 = common.get_AKAZE(img1)[1]
    kps2 = common.get_AKAZE(img2)[0]
    desc2 = common.get_AKAZE(img2)[1]

    matches = find_matches(desc1, desc2, 0.6)
    save_img(draw_matches(I1, I2, kps1, kps2, matches), 'tst.png')
    # print(matches)
    XY = np.zeros((len(matches), 4))
    for i in range(len(matches)):
        XY[i] = np.hstack((kps2[matches[i][1]][:2], kps1[matches[i][0]][:2]))
    H = RANSAC_fit_homography(XY)
    stitched = warp_and_combine(img1, img2, H)
    return stitched


if __name__ == "__main__":

    # Possible starter code; you might want to loop over the task 6 images
    to_stitch = 'vgg'
    I1 = read_img(os.path.join('task6', to_stitch, 'p1.jpg'))
    I2 = read_img(os.path.join('task6', to_stitch, 'p2.jpg'))
    res = make_warped(I1, I2)
    save_img(res, "result_"+to_stitch+".jpg")
