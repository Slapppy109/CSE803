import os
from common import read_img, save_img 
import matplotlib.pyplot as plt
import numpy as np 
from filters import convolve

def corner_score(image, u=5, v=5, window_size=(5,5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return the function E(u,v) for window size W
    # corner detector score for that pixel.
    # Input- image: H x W
    #        u: a scalar for x offset
    #        v: a scalar for y offset
    #        window_size: a tuple for window size
    #
    # Output- results: a image of size H x W
    # Use zero-padding to handle window values outside of the image. 

    h, w = image.shape
    # Get window radius
    window_radius = window_size[0] // 2
    # Find padding size
    pad = window_radius + max(abs(v), abs(u))
    # Pad image
    padded = np.pad(image, pad, 'edge')
    output = np.zeros(image.shape)
    
    #Traverse the image, ignoring the padded edges
    for y in range(pad, h + pad):
        for x in range(pad, w + pad):
            # Grab window
            window = padded[y - window_radius : y + window_radius, x - window_radius: x + window_radius]
            offset_window = padded[y - window_radius + v : y + window_radius + v, x - window_radius + u : x + window_radius + u]
            output[y - pad, x - pad] = ((offset_window - window) ** 2).sum()
    return output

def harris_detector(image, window_size=(5,5)):
    # Given an input image, calculate the Harris Detector score for all pixels
    # Input- image: H x W
    # Output- results: a image of size H x W
    # 
    # You can use same-padding for intensity (or zero-padding for derivatives) 
    # to handle window values outside of the image. 

    ## compute the derivatives 
    Ix = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    Iy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    Ixx = convolve(image, Ix)
    Iyy = convolve(image, Iy)
    Ixy = convolve(convolve(image, Ix),Iy)

    alpha = 0.05
    det = Ixx*Iyy -Ixy**2
    trace = Ixx + Iyy

    # For each location of the image, construct the structure tensor and calculate the Harris response
    response = det - alpha * trace

    return response

def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Feature Detection #####  
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # define offsets and window size and calulcate corner score
    u, v, W = 0, 2, (5,5)
    
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")

if __name__ == "__main__":
    main()
