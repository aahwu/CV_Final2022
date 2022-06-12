import os
import cv2
import numpy as np
from scipy.spatial import distance
import deepeye
from tqdm import tqdm

def Feature_point(image, *args, thres=20, r=3):

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
    # for hh in range(3):
    #     for ww in range(3):
    #         test[center2[0]-1+hh, center2[1]-1+ww, 0:2] = 255

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

def ellipse_center(a):
    a = a.reshape(-1, 1)
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return (int(y0)+1, int(x0)+1)

def ellipse_angle_of_rotation(a):
    a = a.reshape(-1, 1)
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return int(np.rad2deg(0.5*np.arctan(2*b/(a-c))))

def ellipse_axis_length(a):
    a = a.reshape(-1, 1)
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (-1)*np.sqrt((a-c)**2+4*b*b)-(c+a))
    down2=(b*b-a*c)*( (1)*np.sqrt((a-c)**2+4*b*b)-(c+a))
    if down1==0:
        ratio = 1E2
    else:
        ratio = np.sqrt(down2/down1)
    return ratio

def RANSAC(center_d, f, iter, thres, *args):
    if args:
        H_best = args[0]    
    else:
        H_best = np.zeros(6)
    N = f.shape[0]
    k = 0
    max = 0
    while k<iter:
        if len(f)==0:
            return H_best, 0
        k += 1
        ran_index = np.random.randint(f.shape[0], size=5)
        ran_point = f[ran_index]
        H = ellipse_fitting(ran_point)
        A = -(H[1]**2-4*H[0]*H[2])
        B = 4*H[0]*H[4] - 2*H[1]*H[3]
        C = 4*H[0]*H[5] - H[3]**2
        if (A <= 0) or (B**2-4*A*C<=0):
            pass
        else:
            center = ellipse_center(H)
            dist = distance.euclidean(center_d, center)
            if dist > 10 or H[5]==0:
                pass
            else:
                inliner = 0
                for i in range(N):
                    Z = H[0]/(-H[5]) * f[i, 1] ** 2 + H[1]/(-H[5]) * f[i, 1] * f[i, 0] + H[2]/(-H[5]) * f[i, 0]**2 + H[3]/(-H[5]) * f[i, 1] + H[4]/(-H[5]) * f[i, 0]
                    dist = abs(Z - 1)
                    if dist < thres:
                        inliner += 1
                if inliner > max:
                    max = inliner
                    H_best = H
    return H_best, max/N

def starburst(dataset_path: str, subjects: list):

    # load model
    eye_tracker = deepeye.DeepEye()

    sequence_idx = 0
    for subject in subjects:

        solution_dataset = os.path.join(dataset_path, subject+"_starburst")
        if os.path.exists(solution_dataset) != True:
            os.mkdir(solution_dataset)
        H_last = np.zeros(6)
        for action_number in range(16, 17):
            sequence_idx += 1

            # folders path
            preimage_folder = os.path.join(dataset_path, subject+"_preimage", f'{action_number + 1:02d}')
            solution_folder = os.path.join(dataset_path, subject+"_starburst", f'{action_number + 1:02d}')

            # check whether folders exist
            if os.path.exists(solution_folder) != True:
                os.mkdir(solution_folder)
            if os.path.exists(preimage_folder) != True:
                os.mkdir(preimage_folder)
            
            # create and clear conf.txt, conf_real.txt
            conf_name = os.path.join(solution_folder, 'conf_star.txt')
            open(conf_name, 'w').close()
            area_name = os.path.join(solution_folder, 'area.txt')
            open(area_name, 'w').close()
            inliner_name = os.path.join(solution_folder, 'inliner.txt')
            open(inliner_name, 'w').close()
            axis_name = os.path.join(solution_folder, 'axis.txt')
            open(axis_name, 'w').close()

            nr_image = len([name for name in os.listdir(preimage_folder) if name.endswith('.jpg')])
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {preimage_folder}'):
                
                # file path
                label_name = os.path.join(solution_folder, f'{idx}.png')
                preimage_name = os.path.join(preimage_folder, f'{idx}.jpg')

                # load image
                image = cv2.imread(preimage_name)
                h, w = image.shape[0], image.shape[1]

                # inference
                frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                coords = eye_tracker.run(frame_gray)
                center = [coords[1], coords[0]]
                features, _ = Feature_point(image, center, thres=20, r=4)

                thres = 5E-4
                epsilon = 8E-4
                test = np.zeros([h, w, 3], dtype=np.uint8)
                if H_last.any() == False:
                    H, inliner = RANSAC(center, features, 1000, thres)
                else:
                    H, inliner = RANSAC(center, features, 200, thres, H_last)
                H_last = H
                x_coord = np.linspace(0,640,640)
                y_coord = np.linspace(0,480,480)
                X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
                Z_coord = H[0]/(-H[5]) * X_coord ** 2 + H[1]/(-H[5]) * X_coord * Y_coord + H[2]/(-H[5]) * Y_coord**2 + H[3]/(-H[5]) * X_coord + H[4]/(-H[5]) * Y_coord
                ellipse = Z_coord-1 >= thres-epsilon
                test[:, :, 1] = ellipse.astype(np.uint8) * 255
                area = np.count_nonzero(ellipse)
                axis_ratio = ellipse_axis_length(H)

                # for f in features:
                #     test[f[0], f[1], 1] = 0
                #     test[f[0], f[1], 2] = 255
                # for hh in range(3):
                #     for ww in range(3):
                #         test[center[0]-1+hh, center[1]-1+ww, 1] = 255
                #         test[center[0]-1+hh, center[1]-1+ww, 2] = 255

                # label
                # --output
                cv2.imwrite(label_name, test[:, :, 1])
                
                # confidence
                # --threshold
                if axis_ratio < 0.5 or axis_ratio > 2:
                    conf = 0
                else:
                    conf = 1
                # --output
                with open(conf_name, 'a') as f:
                    f.write(str(conf) + '\n')

                # inliner
                with open(inliner_name, 'a') as f:
                    f.write(str(inliner) + '\n')
                
                # axis
                with open(axis_name, 'a') as f:
                    f.write(str(axis_ratio) + '\n')
                    
                # area
                with open(area_name, 'a') as f:
                    f.write(str(area) + '\n')
                # # distance between deepeye and starburst
                # if idx==0:
                #     dist = 0
                # else:
                #     dist = distance.euclidean(center_last, center)
                # center_last = center
                # with open(dist_name, 'a') as f:
                #     f.write(str(dist) + '\n')
                
                