from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import os


def find_fundamental_matrix(shape1,shape2, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """
    # print(shape)
    T=np.array([[1/shape1[0],0,-1/(2*shape1[0])],[0,1/shape1[1],-1/(2*shape1[1])],[0,0,1]])
    T_=np.array([[1/shape2[0],0,-1/(2*shape2[0])],[0,1/shape2[1],-1/(2*shape2[1])],[0,0,1]])
    pts1=np.dot(T,homogenize(pts1).T).T
    pts2=np.dot(T_,homogenize(pts2).T).T
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    U=np.empty((0,9))
    for line1,line2 in zip(pts1,pts2):
        U=np.vstack((U,[line1[0]*line2[0],line1[1]*line2[0],line2[0],line2[1]*line1[0],line2[1]*line1[1],line2[1],line1[0],line1[1],1]))
    eigval,eigvec=np.linalg.eig(np.dot(U.T,U))
    F=eigvec[:,np.argmin(eigval)].reshape((3,3))
    u,s,vh=np.linalg.svd(F)
    s[np.argmin(s)]=0
    F_new=np.dot(u*s,vh)
    # print(F,F_new)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    u,s,vh=np.linalg.svd(F)
    e1=vh[2,:]
    e2=u[:,2]
    # print(np.dot(F,e1.T))
    # print(np.dot(e2,F))
    # print(u,s,vh)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera instrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera instrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    pcd = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pcd


if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape
        F=find_fundamental_matrix(img1.shape,img2.shape,pts1,pts2)
        print(F)
        # you can check against this
        # FCheck, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
        compute_epipoles(F)
        # print(F/FCheck)
        #######################################################################
        # TODO: Your code here                                                #
        #######################################################################
