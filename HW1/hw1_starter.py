# This starter code requires functions in the Dolly Zoom Notebook to work
from re import T
from dolly_zoom import *
import math

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

def normalize(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def combine(im_name, parse = False):
    #set up picture
    p = plt.imread("prokudin-gorskii/" + im_name + ".jpg")
    # p = plt.imread(im_name + ".jpg")

    #n = normalize(p)
    #plt.imsave("norm.jpg", n)

    h, w = p.shape

    pixel_truncate = h % 3
    cut = h // 3
    
    #create blue picture
    b_im = p[: cut, : ]
    # bn_im = n[: cut, : ]
    #create green picture
    g_im = p[cut :cut * 2, : ]
    # gn_im = n[cut :cut * 2, : ]
    #create red picture
    r_im = p[cut * 2 : -pixel_truncate, : ]
    # rn_im = n[cut * 2 : -pixel_truncate, : ]

    combined = np.dstack((r_im, g_im, b_im))
    if parse:
        return b_im, g_im, r_im
    else:
        plt.imsave("Answers/Q2/combined.jpg", combined)

def combine_alignment(im_name):
    b_im, g_im, r_im = combine(im_name, True)
    aligned_g_im = align(b_im, g_im)
    aligned_r_im = align(b_im, r_im)
    aligned_im = np.dstack((aligned_r_im, aligned_g_im, b_im))

    plt.imsave(f"Answers/Q2/aligned_"+ im_name +".jpg", aligned_im)

def roll_XY(img, x, y):
    #roll image via x-axis
    rolled = np.roll(img, x, axis = 1)
    #return rolled image via y-axis
    return np.roll(rolled, y, axis = 0)

def align(ref_color, color, channel):
    min_val = sys.maxsize
    offset = [0, 0]
    for i in range (-15, 16):
        for j in range(-15, 16):
            metric = np.sum((roll_XY(color, i, j) - ref_color) ** 2)
            #metric = np.dot(roll_XY(color, i, j), ref_color)
            if metric < min_val:
                min_val = metric
                offset = [i,j]

    return roll_XY(channel, offset[0], offset[1])

def combine_all():
    im_name = ["00125v", "00149v", "00153v", "00351v", "00398v", "01112v"]
    for im in im_name:
        combine_alignment(im)
        
    
if __name__ == "__main__":
    # combine("efros_tableau")
    combine("00153v")