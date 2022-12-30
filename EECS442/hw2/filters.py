import os

import numpy as np

from common import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    output = []
    for i in range(image.shape[0]//patch_size[0]):
        for j in range(image.shape[1]//patch_size[1]):
            this_patch = image[i*patch_size[0]
                :(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]]
            output.append((this_patch-this_patch.mean())/this_patch.std())
    return output


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    if kernel.ndim == 1:
        kernel.resize((kernel.shape[0], 1))
    kernel = kernel[::-1, ::-1]
    H = image.shape[0]
    W = image.shape[1]
    h = kernel.shape[0]
    w = kernel.shape[1]
    image = np.vstack((np.hstack((image, np.zeros((H, w-1)))), np.zeros(
        (h-1, W+w-1))))
    output = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            output[i, j] = np.sum(
                kernel*image[i:i+h, j:j+w])
    return output


def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([[1, 0, -1]])
    ky = kx.T  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix**2+Iy**2)

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Sy = Sx.T
    Gx, Gy, grad_magnitude = convolve(image, Sx), convolve(
        image, Sy), np.sqrt(convolve(image, Sx)**2+convolve(image, Sy)**2)

    return Gx, Gy, grad_magnitude


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Task 1: Image Patches --
    # (a)
    # First complete image_patches()
    # patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    # for i in range(3):
    #     chosen_patches = patches[i]
    #     save_img(chosen_patches, "./image_patches/q1_patch"+str(i)+".png")
    # (b), (c): No code

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Task 2: Convolution and Gaussian Filter --
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    sigma = 0.572
    kernel_gaussian = np.array(
        [[1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2)) for x in [-1, 0, 1]] for y in [-1, 0, 1]])
    print(kernel_gaussian.shape)
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # (d), (e): No code

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    # _, _, edge_detect = edge_detection(img)
    # save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    # _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    # save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    # print("Gaussian Filter is done. ")

    # -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    # Gx, Gy, edge_sobel = sobel_operator(img)
    # save_img(Gx, "./sobel_operator/q2_Gx.png")
    # save_img(Gy, "./sobel_operator/q2_Gy.png")
    # save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img,kernel_LoG1)
    filtered_LoG2 = convolve(img,kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    data = np.load('log1d.npz')
    dfiltered_approx=convolve(img,data['gauss53'])-convolve(img,data['gauss50'])
    save_img(dfiltered_approx, "./log_filter/dfilter_approx.png")
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
