import numpy as np
import matplotlib.pyplot as plt

for i in range(1, 2):
    data = np.load('./task4/points_case_'+str(i)+'.npy')
    x = data[:, 0]
    y = data[:, 1]
    x_ = data[:, 2]
    y_ = data[:, 3]
    A = np.zeros((data.shape[0]*2, 9))
    for i in range(data.shape[0]):
        A[2*i, :] = [x[i], y[i], 1, 0, 0, 0, -x_[i]*x[i], -x_[i]*y[i], -x_[i]]
        A[2*i+1, :] = [0, 0, 0, x[i], y[i], 1,  -y_[i]*x[i], -y_[i]*y[i], -y_[i]]
    print(np.linalg.eig(np.dot(A.T,A))[1][np.argmin(np.linalg.eig(np.dot(A.T,A))[0])].reshape((3,3)))

    # plt.scatter(x, y, c='r', s=2)
    # plt.scatter(x_, y_, c='g', s=2)
    # xy=A.dot(v[0])
    # plt.scatter(xy[::2],xy[1::2],c='b', s=2)
    # plt.savefig('./task3/data.png')
    # plt.clf()
