b = data1[:, 2:].flatten()
# A = np.zeros((data1.shape[0]*2, 6))
# for i in range(data1.shape[0]):
#     A[2*i, :] = [data1[i, 0], data1[i, 1], 0, 0, 1, 0]
#     A[2*i+1, :] = [0, 0, data1[i, 0], data1[i, 1], 0, 1]
# v = np.linalg.lstsq(A, b)
# print(v[0])

# x = data1[:, 0]
# y = data1[:, 1]
# x_ = data1[:, 2]
# y_ = data1[:, 3]

# plt.scatter(x, y, c='r', s=2)
# plt.scatter(x_, y_, c='g', s=2)
# xy=A.dot(v[0])
# plt.scatter(xy[::2],xy[1::2],c='b', s=2)
# plt.savefig('./task3/data1.png')
# plt.clf()