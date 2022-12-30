import numpy as np
from numpy.linalg import eig
import utils


def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - M: Numpy array of shape (3,4) giving the camera projection matrix P

    """
    M = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    N = pts2d.shape[0]
    A = np.empty((0, 12))
    for line2d, line3d in zip(pts2d, pts3d):
        line3d_homo = np.hstack((line3d, 1))
        zeros = [0, 0, 0, 0]
        A = np.vstack((A, [np.hstack((zeros, line3d_homo, -line2d[1]*line3d_homo)),
                           np.hstack((line3d_homo, zeros, -line2d[0]*line3d_homo)),
                           np.hstack((-line2d[1]*line3d_homo, line2d[0]*line3d_homo, zeros))]))
    eigval,eigvec=np.linalg.eig(np.dot(A.T,A))
    M=eigvec[:,np.argmin(eigval)].reshape((3,4))
    # print(M)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return M


if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")
    M=find_projection(pts2d, pts3d)
    print(pts3d[0].shape[0])
    proj=np.dot(M,utils.homogenize(pts3d[0].reshape((1,3))).T)
    proj/=proj[2][0]
    print(proj)
    print(pts2d[0])
    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """
