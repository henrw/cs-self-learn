"""
Homography fitting functions
You should write these
"""
import numpy as np
from common import homography_transform
import matplotlib.pyplot as plt


def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].

    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''
    x = XY[:, 0]
    y = XY[:, 1]
    x_ = XY[:, 2]
    y_ = XY[:, 3]
    A = np.zeros((XY.shape[0]*2, 9))
    for i in range(XY.shape[0]):
        A[2*i, :] = [0, 0, 0, -x[i], -y[i], -1, y_[i]*x[i], y_[i]*y[i], y_[i]]
        A[2*i+1, :] = [x[i], y[i], 1, 0, 0, 0,  -x_[i]*x[i], -x_[i]*y[i], -x_[i]]
    H=np.linalg.eig(np.dot(A.T, A))[1][:,np.argmin(np.linalg.eig(np.dot(A.T, A))[0])].reshape((3,3))
    return H


def RANSAC_fit_homography(XY, eps=0.8, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation
    matrix which has the most inliers

    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the
                    inliers from the best model.

    Hints:
    a) Sample without replacement. Otherwise you risk picking a set of points
       that have a duplicate.
    b) *Re-fit* the homography after you have found the best inliers
    '''
    N=XY.shape[0]
    bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
    bestRefit = np.eye(3)
    for iter in range(nIters):
        inlier_index=np.random.randint(N,size=4)
        inlier_XY=XY[inlier_index,:]
        this_H=fit_homography(inlier_XY)
        diff=np.sum((homography_transform(XY[:,:2],this_H)-XY[:,2:])**2,axis=1,keepdims=True)
        thisInliers=diff<eps**2
        if np.sum(thisInliers)>bestCount:
            bestH=this_H
            bestInliers=thisInliers
            bestCount=np.sum(thisInliers)
    for i in range(2):
        bestInliers=np.hstack((bestInliers,bestInliers))
    bestRefit=fit_homography(XY[bestInliers].reshape(-1,4))
    return bestRefit


if __name__ == "__main__":
    # If you want to test your homography, you may want write any code here, safely
    # enclosed by a if __name__ == "__main__": . This will ensure that if you import
    # the code, you don't run your test code too
    for i in range(1, 10):
        data = np.load('./task4/points_case_'+str(i)+'.npy')
        H = fit_homography(data)
        XY = homography_transform(data[:, 0:2], H)
        plt.scatter(data[:, 0], data[:, 1], c='r', s=2)
        plt.scatter(data[:, 2], data[:, 3], c='g', s=2)
        plt.scatter(XY[:, 0], XY[:, 1], c='b', s=2)
        plt.savefig('./task4/points_case_'+str(i)+'.png')
        plt.clf()
    pass
