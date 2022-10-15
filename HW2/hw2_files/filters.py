import numpy as np
import os
from common import *
import math
from math import cos
from cmath import sin

## Image Patches ##
def image_patches(image, patch_size=(16,16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N 
    # Output- results: a list of images of size M x N
    
    h,w = image.shape
    output = []
    for j in range( 0, h, patch_size[0]):
        for i in range( 0, w, patch_size[1]):
            # split up a patch
            patch = image[ j : j + patch_size[0], i : i + patch_size[1]]
            # normalize each patch
            norm = (patch - patch.mean()) / patch.var()
            output.append(patch)
            
    return output


## Gaussian Filter ##
def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Reminder to implement convolution and not cross-correlation!
    # Input- image: H x W
    #        kernel: h x w
    # Output- convolve: H x W

    # Get number of rows and columns
    row, col = image.shape

    #kernel radius
    #kernel radius is the same as pad width
    k_radius = np.max(kernel.shape) // 2

    output = np.zeros(image.shape)

    # Make a copy and pad the edges
    padded = np.pad(image, k_radius, 'edge')

    # Traverse the image, ignoring the padded edges
    for y in range(k_radius, row + k_radius):
        for x in range(k_radius, col + k_radius):
            # Grab window
            window = padded[y - k_radius : y + k_radius + 1, x - k_radius : x + k_radius + 1]
            #convolve the window
            output[y - k_radius, x - k_radius] = (kernel * window).sum()
    return output


## Edge Detection ##
def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    kx = np.array([
        [-1/2, 0, 1/2],
    ])  
    ky = np.array([
        [-1/2], 
        [0], 
        [1/2]
    ])  

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    grad_magnitude = np.sqrt((Ix ** 2) + (Iy ** 2))

    return grad_magnitude, Ix, Iy


## Sobel Operator ##
def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    sobelX = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    Gx, Gy = convolve(image, sobelX), convolve(image, sobelY)
    grad_magnitude = np.sqrt((Gx**2) + (Gy**2))

    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=[0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W

    output = []
    for a in angles:
        steerable_kernel = np.array([
            [cos(a) + sin(a), 2 * sin(a), sin(a) - cos(a)],
            [2 * cos(a), 0, -2 * cos(a)],
            [cos(a) - sin(a), -2 * sin(a), -1*(sin(a) + cos(a))]
        ])
        output.append(convolve(image, steerable_kernel))
    return output

def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Image Patches #####
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # Q1
    patches = image_patches(img)
  
    p1 = patches[260]
    p2 = patches[528]
    p3 = patches[0]

    save_img(p1, "./image_patches/q1_p1_patch.png")
    save_img(p2, "./image_patches/q1_p2_patch.png")
    save_img(p3, "./image_patches/q1_p_3patch.png")

    # Q2: No code

    ##### Gaussian Filter #####
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # Q1: No code

    # Q2
    kernel_gaussian = np.array([
    [math.log(2)/(4*math.pi),math.log(2)/(2*math.pi),math.log(2)/(4*math.pi)],
    [math.log(2)/(2*math.pi),math.log(2)/(math.pi),math.log(2)/(2*math.pi)],
    [math.log(2)/(4*math.pi),math.log(2)/(2*math.pi),math.log(2)/(4*math.pi)]
    ])

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # Q3
    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    ########################

    ##### Sobel Operator #####
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Q1: No code

    # Q2
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    # Q3
    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    ########################

    #####LoG Filter#####
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # Q1
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # Q2: No code

    print("LoG Filter is done. ")
    ########################


if __name__ == "__main__":
    main()
