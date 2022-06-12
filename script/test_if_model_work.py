import matplotlib.pyplot as plt
import deepeye
import numpy as np
import random
from scipy.spatial import distance
import tensorflow as tf
import os
import cv2

def alpha_blend(input_image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5):
    """Alpha Blending utility to overlay segmentation masks on input images
    Args:
        input_image: a np.ndarray with 1 or 3 channels
        segmentation_mask: a np.ndarray with 3 channels
        alpha: a float value
    """
    if len(input_image.shape) == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    blended = input_image.astype(np.float32) * alpha + segmentation_mask.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    return blended

def Feature_point(image, *args, thres=20, r=4):

    image = image.astype(int)
    h, w = image.shape[0], image.shape[1]
    if args:
        center = args[0]
    else:
        center = [h//2, w//2]
    center1 = center

    # stage 1
    features = []
    for i in range(18):
        delta_y, delta_x = r*np.sin(20*i * (np.pi/180)), r*np.cos(20*i * (np.pi/180))
        delta_y = delta_y.astype(int)
        delta_x = delta_x.astype(int)
        ray_y1, ray_x1 = center1[0]+delta_y, center1[1]+delta_x
        ray_y2, ray_x2 = ray_y1+delta_y, ray_x1+delta_x
        while ray_x2<w and ray_x2>0 and ray_y2<h and ray_y2>0:
            delta = image[ray_y2, ray_x2, 0]-image[ray_y1, ray_x1, 0]
            if delta > thres:
                point = [(ray_y2+ray_y1)//2, (ray_x2+ray_x1)//2, i, delta]
                features.append(point)
                break
            ray_y1, ray_x1 = ray_y2, ray_x2
            ray_y2, ray_x2 = ray_y2+delta_y, ray_x2+delta_x
    # for f in features:
    #     test[f[0], f[1], 1] = 255

    # stage 2
    features2 = []
    for f in features:
        ray_feat_num = 8*f[3]/thres
        ray_feat_num = ray_feat_num.astype(int)
        ray_center = [f[0], f[1]]
        for j in range(ray_feat_num):
            delta_y, delta_x = r*np.sin((20*f[2]-180-50+100/(ray_feat_num-1)*j) * (np.pi/180)), r*np.cos((20*f[2]-180-50+100/(ray_feat_num-1)*j) * (np.pi/180))
            delta_y = delta_y.astype(int)
            delta_x = delta_x.astype(int)
            ray_y1, ray_x1 = ray_center[0]+delta_y, ray_center[1]+delta_x
            ray_y2, ray_x2 = ray_y1+delta_y, ray_x1+delta_x
            while ray_x2<w and ray_x2>0 and ray_y2<h and ray_y2>0:
                delta = image[ray_y2, ray_x2, 0]-image[ray_y1, ray_x1, 0]
                if delta > thres:
                    point = [(ray_y2+ray_y1)//2, (ray_x2+ray_x1)//2, delta]
                    features2.append(point)
                    break
                ray_y1, ray_x1 = ray_y2, ray_x2
                ray_y2, ray_x2 = ray_y2+delta_y, ray_x2+delta_x
    # for f in features2:
    #     test[f[0], f[1], 2] = 255
    
    # update center
    total_features = [[f[0], f[1]] for f in features] + [[f[0], f[1]] for f in features2]
    total_features = np.array(total_features)
    center2 = total_features.mean(axis=0, dtype=np.uint)
    # for hh in range(3):
    #     for ww in range(3):
    #         test[center2[0]-1+hh, center2[1]-1+ww, 0:2] = 255

    if distance.euclidean(center1, center2) < 5:
        center = center2

    return total_features, center

def ellipse_fitting(f):
    N = f.shape[0]
    if N < 5:
        print('At least 5 points should be given.')
    A = np.zeros([N, 6])
    for i in range(N):
        A[i, :] = [f[i, 1]**2, f[i, 1]*f[i, 0], f[i, 0]**2, f[i, 1], f[i, 0], 1]
    
    U, S, V = np.linalg.svd(A)
    V = V.transpose()
    H = np.ones(6)
    H = V[:, -1]
    return H

def RANSAC(f, iter, thres):
    H_best = np.zeros(6)
    N = f.shape[0]
    k = 0
    max = 0
    while k<iter:
        ran_index = np.random.randint(f.shape[0], size=5)
        ran_point = f[ran_index]
        H = ellipse_fitting(ran_point)
        A = -(H[1]**2-4*H[0]*H[2])
        B = 4*H[0]*H[4] - 2*H[1]*H[3]
        C = 4*H[0]*H[5] - H[3]**2
        if (A <= 0) or (B**2-4*A*C<=0):
            pass
        else:
            k += 1
            inliner = 0
            for i in range(N):
                Z = H[0]/(-H[5]) * f[i, 1] ** 2 + H[1]/(-H[5]) * f[i, 1] * f[i, 0] + H[2]/(-H[5]) * f[i, 0]**2 + H[3]/(-H[5]) * f[i, 1] + H[4]/(-H[5]) * f[i, 0]
                dist = abs(Z - 1)
                if dist < thres:
                    inliner += 1
            if inliner > max:
                max = inliner
                H_best = H
    print(max/N)
    return H_best

def test_if_model_work():
    dataset_path = r'C:\Users\ShaneWu\OneDrive\Desktop\Hw(senior)\CV\Final\CV22S_Ganzin\model'
    # image = cv2.imread(os.path.join(dataset_path, "1.png"))
    image = tf.io.read_file(os.path.join(dataset_path, "1.png"))
    image = tf.image.decode_png(image, channels=1)
    test_name = os.path.join(dataset_path, 'test.txt')
    open(test_name, 'w').close()
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            with open(test_name, 'a') as f:
                f.write(str(image[h, w, :]) + '\n')
    # h, w = image.shape[0], image.shape[1]
    # frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # coords = eye_tracker.run(frame_gray)
    # center = [coords[1], coords[0]]
    # features, _ = Feature_point(image, center, thres=20, r=4)
    # test = np.zeros([h, w, 3], dtype=np.uint8)

    # for f in features:
    #     test[f[0], f[1], 1] = 255
    # cv2.imwrite(os.path.join(dataset_path, "test.png"), test)
    # thres = 5E-4
    # epsilon = 8E-4
    # H = RANSAC(features, iter=1000, thres=thres)

    # x_coord = np.linspace(0,640,640)
    # y_coord = np.linspace(0,480,480)
    # X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    # Z_coord = H[0]/(-H[5]) * X_coord ** 2 + H[1]/(-H[5]) * X_coord * Y_coord + H[2]/(-H[5]) * Y_coord**2 + H[3]/(-H[5]) * X_coord + H[4]/(-H[5]) * Y_coord
    # ellipse = Z_coord-1 >= thres-epsilon
    # test[:, :, 0] = ellipse.astype(np.uint8) * 255
    # blended = alpha_blend(image, test, 0.5)

    # cv2.imwrite(os.path.join(dataset_path, "test_ellipse.png"), blended)
    # cv2.imshow('DeepEye', blended)

    # cv2.waitKey(0)


if __name__ == "__main__":
    # If model works, the "test_prediction.png" should show the segmented area of pupil from "test_image.png"
    test_if_model_work()

