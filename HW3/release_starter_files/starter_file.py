from turtle import right
import numpy as np
from matplotlib import pyplot as plt
from common import *
# feel free to include libraries needed


def homography_transform(X, H):
    # Perform homography transformation on a set of points X
    # using homography matrix H
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix 
    # Output - a set of 2D points in an array with size (N,2)
    X = np.hstack((X, np.ones((X.shape[0],1))))
    Y = X @ H.T
    Y /= Y[:,2:3]
    return Y[:,:2]


def fit_homography(XY):
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix
    X = XY[:,0:2]
    Y = XY[:,2:4]
    rows = XY.shape[0]
    o = np.ones((rows,1))
    X = np.hstack((X, o)) 
    A = np.zeros((rows*2, 9))
    for i in range(rows):
        A = np.vstack((A, np.array([0,0,0, -X[i][0], -X[i][1], -X[i][2], Y[i][1] * X[i][0], Y[i][1] * X[i][1], Y[i][1] * X[i][2]])))
        A = np.vstack((A, np.array([X[i][0], X[i][1], X[i][2], 0,0,0, -Y[i][0] * X[i][0], -Y[i][0] * X[i][1], -Y[i][0] * X[i][2]])))
    val, vec = np.linalg.eig(A.T @ A)
    H = vec[:, np.argmin(val)].reshape((3, 3))
    return H


def p1():
    # 1.2.3 - 1.2.5
    # TODO
    # 1. load points X from p1/transform.npy

    # 2. fit a transformation y=Sx+t

    # 3. transform the points 

    # 4. plot the original points and transformed points


    # 1.2.6 - 1.2.8
    case = 8 # you will encounter 8 different transformations
    for i in range(case):
        XY = np.load('p1/points_case_'+str(i)+'.npy')
        # 1. generate your Homography matrix H using X and Y
        #
        #    specifically: fill function fit_homography() 
        #    such that H = fit_homography(XY)
        H = fit_homography(XY)
        # 2. Report H in your report
        print(H)
        # 3. Transform the points using H
        #
        #    specifically: fill function homography_transform
        #    such that Y_H = homography_transform(X, H)
        Y_H = homography_transform(XY[:,:2], H)
        # 4. Visualize points as three images in one figure
        # the following codes plot figure for you
        plt.scatter(XY[:,1],XY[:,0],c="red") #X
        plt.scatter(XY[:,3],XY[:,2],c="green") #Y
        plt.scatter(Y_H[:,1],Y_H[:,0],c="blue") #Y_hat
        plt.savefig('./case_'+str(i))
        plt.close()

def stitchimage(imgleft, name_L, imgright, name_R):
    name_T = name_L.split('_')[0]
    #imgleft, imgright = equalizeShape(imgleft, imgright)

    # 1. extract descriptors from images
    #    you may use SIFT/SURF of opencv
    gray_L = cv2.cvtColor(imgleft, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(imgright, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp_L, ds_L= sift.detectAndCompute(gray_L, None)
    kp_R, ds_R = sift.detectAndCompute(gray_R, None)

    left_draw_kp = cv2.drawKeypoints(gray_L, kp_L, imgleft, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    right_draw_kp= cv2.drawKeypoints(gray_R, kp_R, imgright, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    save_img(f"{name_L}_kp.jpg", left_draw_kp)
    save_img(f"{name_R}_kp.jpg", right_draw_kp)

    # 2. select paired descriptors
    kp_L_loc = np.array([[kp.pt[0],kp.pt[1]] for kp in kp_L])
    kp_R_loc = np.array([[kp.pt[0],kp.pt[1]] for kp in kp_R])
    distfunc= lambda p1, p2: np.sqrt(((p1-p2)**2).sum())
    dist = np.asarray([[distfunc(p1, p2) for p2 in ds_R] for p1 in ds_L])

    t = 0.5
    matches = []
    des_indL = []
    des_indR = []

    for i in range(len(dist)):
        part = dist[i]
        min_idx = np.argpartition(part, 2)
        ratio = part[min_idx[0]]/part[min_idx[1]]
        if ratio < t:
            indL = i
            indR = part.argmin()
            m = cv2.DMatch(indL, indR, dist[indL][indR])
            matches.append(m)
            des_indL.append(indL)
            des_indR.append(indR)

    match = cv2.drawMatches(gray_L, kp_L, gray_R,
                                kp_R, matches, None, flags=2)

    des_indL = np.array(des_indL)
    des_indR = np.array(des_indR)
    save_img(f"{name_T}_match.jpg", match)

    # 3. run RANSAC to find a transformation
    #    matrix which has most innerliers

    kp_Lx = []
    kp_Ly = []
    for i in des_indL:
        kp_Lx.append([kp_L_loc[i][0]])
        kp_Ly.append([kp_L_loc[i][1]])

    kp_Rx = []
    kp_Ry = []
    for i in des_indR:
        kp_Rx.append([kp_R_loc[i][0]])
        kp_Ry.append([kp_R_loc[i][1]])

    kp_Lx = np.array(kp_Lx)
    kp_Ly = np.array(kp_Ly)
    kp_Rx = np.array(kp_Rx)
    kp_Ry = np.array(kp_Ry)

    kp_Lxy = np.hstack((kp_Lx, kp_Ly))
    kp_Rxy = np.hstack((kp_Rx, kp_Ry))

    iter = 800
    avg_res = None
    max_in = 0
    bH = None

    for i in range(iter):
        p = np.random.choice(len(des_indL), 4, replace=False)
        lx = []
        ly = []
        for i in des_indL[p]:
            lx.append([kp_L_loc[i][0]])
            ly.append([kp_L_loc[i][1]])
        rx = []
        ry = []
        for i in des_indR[p]:
            rx.append([kp_R_loc[i][0]])
            ry.append([kp_R_loc[i][1]])
        A = np.hstack((np.array(lx),np.array(ly),np.array(rx),np.array(ry)))
        H = fit_homography(A)
        trf = homography_transform(kp_Lxy, H)
        ransac_dist = np.linalg.norm(trf - kp_Rxy, axis=1)
        inl = np.sum(ransac_dist < 1)
        if inl > max_in:
            max_in = inl
            bH = H
            avg_res = np.mean(ransac_dist**2)
            
    print(f"Max Number of Inliers: {max_in}")
    # 4. warp one image by your transformation 
    #    matrix
    #
    #    Hint: 
    #    a. you can use opencv to warp image
    #    b. Be careful about final image size
    imgleft = cv2.imread(f"p2/{name_L}.jpg")
    imgleft = cv2.cvtColor(imgleft, cv2.COLOR_BGR2RGB)
    hl, wl, c = imgleft.shape
    trans_L = np.array([[1, 0, wl], [0, 1, hl], [0, 0, 1]])
    warp_L = cv2.warpPerspective(imgleft, trans_L @ bH, (2200, 1500))

    # 5. combine two images, use average of them
    #    in the overlap area
    imgright = cv2.imread(f"p2/{name_R}.jpg")
    imgright = cv2.cvtColor(imgright, cv2.COLOR_BGR2RGB)
    hr, wr, c = imgright.shape

    combined = warp_L.astype('int64')
    for i in range(hr):
        for j in range(wr):
            c_val = [combined[hl + i][wl + j][0], combined[hl + i][wl + j][1], combined[hl + i][wl + j][2]]
            if c_val == [0, 0, 0]:
                combined[hl + i][wl + j] = imgright[i][j]
            else:
                combined[hl + i][wl + j] = (combined[hl + i][wl + j]+imgright[i][j])/2

    return combined


def p2(p1, p2, savename):
    # read left and right images
    imgleft = read_colorimg(p1)
    imgright = read_colorimg(p2)
    # stitch image
    output = stitchimage(imgleft, imgright)
    # save stitched image
    save_img('./' + savename + '.jpg', output)


if __name__ == "__main__":
    # Problem 1
    p1()

    # Problem 2
    p2('p2/uttower_left.jpg', 'p2/uttower_right.jpg', 'uttower')
    p2('p2/bbb_left.jpg', 'p2/bbb_right.jpg', 'bbb')
