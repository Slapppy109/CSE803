# This starter code requires functions in the Dolly Zoom Notebook to work
from re import T
from dolly_zoom import *
import math

import cv2 as cv
import os
import imageio
import time
import sys

# Call this function to generate gif. make sure you have rotY() implemented.
def generate_gif():
    n_frames = 30
    if not os.path.isdir("frames"):
        os.mkdir("frames")
    fstr = "frames/%d.png"
    for i,theta in enumerate(np.arange(0,2*np.pi,2*np.pi/n_frames)):
        fname = fstr % i
        renderCube(f=15, t=(0,0,3), R=rotY(theta), O=True)
        plt.savefig(fname)
        plt.close()

    with imageio.get_writer("cube.gif", mode='I') as writer:
        for i in range(n_frames):
            frame = plt.imread(fstr % i)
            writer.append_data(frame)
            os.remove(fstr%i)
            
    os.rmdir("frames")

def rotY(theta):
    return [[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [(-1 * math.sin(theta)), 0, math.cos(theta)]]

def rotX(theta):
    return [[1, 0, 0], [0, math.cos(theta), -1 * math.sin(theta)], [0, math.sin(theta), math.cos(theta)]]

def rotXthenY(theta):
    return np.matmul(rotY(theta), rotX(theta))

def rotYthenX(theta):
    return np.matmul(rotX(theta), rotY(theta))

def centerDiag():
    return np.matmul(rotX(np.pi/5.1), rotY(np.pi/4))

def orthogdraw():
    fname = "1d"
    renderCube(f=15, t=(0,0,3), R=centerDiag(), O=True)
    plt.savefig(fname)
    plt.close()

def parse(im):
    h, w = im.shape
    cut = h // 3
    
    #create blue picture
    b_im = im[: cut, : ]

    #create green picture
    g_im = im[cut :cut * 2, : ]

    #create red picture
    r_im = im[cut * 2 : cut * 3, : ]

    return b_im, g_im, r_im

def combine(im_name):
    #set up picture
    p = plt.imread("prokudin-gorskii/" + im_name + ".jpg")

    b_im, g_im, r_im = parse(p)

    combined = np.dstack((r_im, g_im, b_im))
    plt.imsave("Answers/Q2/combined.jpg", combined)

def combine_alignment(im, im_name, write = True, start = [(0,0), (0,0)]):
    b_im, g_im, r_im = parse(im)

    b_blur = cv.GaussianBlur(b_im, (3,3), 0)
    b_details =  b_im - b_blur

    g_blur = cv.GaussianBlur(g_im, (3,3), 0)
    g_details = g_im - g_blur

    r_blur = cv.GaussianBlur(r_im, (3,3), 0)
    r_details =  r_im - r_blur

    g_im_offset= align(b_details, g_details, start[0])
    aligned_g_im = roll_XY(g_im, g_im_offset[0], g_im_offset[1])

    r_im_offset= align(b_details, r_details, start[1])
    aligned_r_im = roll_XY(r_im, r_im_offset[0], r_im_offset[1])

    aligned_im = cv.merge([b_im, aligned_g_im, aligned_r_im])

    print(im_name)
    print(f"R Offset: {r_im_offset}")
    print(f"G Offset: {g_im_offset}")

    if write:
        cv.imwrite(f"Answers/Q2/aligned_"+ im_name +".jpg", aligned_im)
    return g_im_offset, r_im_offset

def roll_XY(img, x, y):
    #roll image via x-axis
    rolled = np.roll(img, x, axis = 1)
    #return rolled image via y-axis
    return np.roll(rolled, y, axis = 0)

def align(ref_color, color, start):
    min_val = sys.maxsize
    offset = start
    for i in range (start[0] - 15, start[0] + 16):
        for j in range(start[1] - 15, start[1] + 16):
            metric = np.sum((roll_XY(color, i, j) - ref_color) ** 2)
            if metric < min_val:
                min_val = metric
                offset = (i,j)

    return offset

def testCombine(im_name):
    p = cv.imread(im_name + ".jpg")
    p = cv.cvtColor(p, cv.COLOR_BGR2GRAY)
    combine_alignment(p, im_name, False)

def pyramid(im_name):
    p = cv.imread(im_name + ".jpg")

    # Convert for correct shape and color channels
    p = cv.cvtColor(p, cv.COLOR_BGR2GRAY)

    h, w = p.shape
    p_low = cv.resize(p, (w//2,h//2))
    g_offset_lower, r_offset_lower = combine_alignment(p_low, im_name + "_lower", False)
    print(f"Coarse G Channel Offset: {g_offset_lower}")
    print(f"Coarse R Channel Offset: {r_offset_lower}")

    g_offset_full, r_offset_full = combine_alignment(p, im_name, True, [g_offset_lower, r_offset_lower] )
    print(f"Full Resolution G Channel Offset: {g_offset_full}")
    print(f"Full Resolution R Channel Offset: {r_offset_full}")

def combine_all():
    im_name = ["00125v", "00149v", "00153v", "00351v", "00398v", "01112v"]
    for im in im_name:
        testCombine("prokudin-gorskii/" + im)
        
def split_plot():
    p = cv.imread("indoor.png")
    b_im, g_im, r_im = cv.split(p)
    cv.imshow("Red_Indoor", r_im)
    cv.imshow("Blue_Indoor", b_im)
    cv.imshow("Green_Indoor", b_im)

    p = cv.cvtColor(p, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(p)
    cv.imshow("Light_Indoor", l)
    cv.imshow("Red-Green_Indoor", a)
    cv.imshow("Blue-Yellow_Indoor", b)

    p = cv.imread("outdoor.png")
    b_im, g_im, r_im = cv.split(p)
    cv.imshow("Red_Outdoor", r_im)
    cv.imshow("Blue_Outdoor", b_im)
    cv.imshow("Green_Outdoor", b_im)

    p = cv.cvtColor(p, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(p)
    cv.imshow("Light_Outdoor", l)
    cv.imshow("Red-Green_Outdoor", a)
    cv.imshow("Blue-Yellow_Outdoor", b)

    cv.waitKey(0)

def compare():
    coord1 = (60,93)
    coord2 = (55,106)

    p1 = cv.imread("Answers/Q3/im1.jpg")
    p1_patch = p1[coord1[1] : coord1[1]+32, coord1[0] : coord1[0]+32]

    p2 = cv.imread("Answers/Q3/im2.jpg")
    p2_patch = p2[coord2[1] : coord2[1]+32, coord2[0] : coord2[0]+32]

    #Compare LAB colorspace
    p1_patch = cv.cvtColor(p1_patch, cv.COLOR_BGR2LAB)
    p1_l, p1_a, p1_b = cv.split(p1_patch)

    p1_concat = cv.hconcat([p1_l, p1_a, p1_b])
    cv.imshow("P1 Patch", p1_concat)

    p2_patch = cv.cvtColor(p2_patch, cv.COLOR_BGR2LAB)
    p2_l, p2_a, p2_b = cv.split(p2_patch)

    p2_concat = cv.hconcat([p2_l, p2_a, p2_b])
    cv.imshow("P2 Patch", p2_concat)

    cv.waitKey(0)


if __name__ == "__main__":
    #combine("efros_tableau")
    # combine_alignment("efros_tableau")
    # testCombine("prokudin-gorskii/00153v")
    #combine_all()
    #combine("00153v")
    # pyramid("seoul_tableau")
    #split_plot()
    #compare()
    # renderCube(f=15, t=(0,0,3), R=centerDiag())
    # plt.savefig("test")
    # plt.close()
    # orthogdraw()
    # testCombine("efros_tableau")