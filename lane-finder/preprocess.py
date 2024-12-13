import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from os import path

def calibrate_camera(nx, ny, basepath):
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    images = glob.glob(path.join(basepath, 'calibration*.jpg'))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imshow('input image', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dist_pickle = {"mtx": mtx, "dist": dist}
    destnation = path.join(basepath, 'calibration_pickle.p')
    pickle.dump(dist_pickle, open(destnation, "wb"))
    print("calibration data is written into: {}".format(destnation))

    return mtx, dist

def load_calibration(calib_file):
    with open(calib_file, 'rb') as file:
        data = pickle.load(file)
        mtx = data['mtx']
        dist = data['dist']

    return mtx, dist

def undistort_image(imagepath, calib_file, visulization_flag):
    mtx, dist = load_calibration(calib_file)
    img = cv2.imread(imagepath)
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_undistRGB = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

    if visulization_flag:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(imgRGB)
        ax1.set_title('Original Image', fontsize=30)
        ax1.axis('off')
        ax2.imshow(img_undistRGB)
        ax2.set_title('Undistorted Image', fontsize=30)
        ax2.axis('off')
        plt.show()

    return img_undistRGB

if __name__ == "__main__":
    nx, ny = 9, 6
    basepath = 'camera_cal/'
    calibrate_camera(nx, ny, basepath)
