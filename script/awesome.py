import os
import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib
import matplotlib.pyplot as plt

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

def Feature_point(image, *args):

    image = image.astype(int)
    h, w = image.shape[0], image.shape[1]
    if args:
        center = args[0]
    else:
        center = [h//2, w//2]
    center1 = center
    test = np.zeros([h, w, 3], dtype=np.uint8)
    # test[center[0], center[1], 0] = 255

    iter = 50
    k = 0
    while True:
        features = []
        # stage 1
        thres = 20
        r = 8
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
        
        total_features = [[f[0], f[1]] for f in features] + [[f[0], f[1]] for f in features2]
        total_features = np.array(total_features)
        center2 = total_features.mean(axis=0, dtype=np.uint)
        # for hh in range(3):
        #     for ww in range(3):
        #         test[center2[0]-1+hh, center2[1]-1+ww, 0:2] = 255

        if distance.euclidean(center1, center2) < 10:
            center = center2
            break
        center1 = center2
        
        if k == iter:
            break
        k += 1

    return total_features, test, center

dataset_path = r'C:\Users\ShaneWu\OneDrive\Desktop\Hw(senior)\CV\Final\CV22S_Ganzin\dataset\public\S2\17'
nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
image = cv2.imread(os.path.join(dataset_path, '0.jpg'))
h = image.shape[0]
w = image.shape[1]
dpi = matplotlib.rcParams['figure.dpi']
fig = plt.figure(figsize=(w / dpi, h / dpi))
ax = fig.add_axes([0, 0, 1, 1])

for idx in range(nr_image):
    image_name = os.path.join(dataset_path, f'{idx}.jpg')
    image = cv2.imread(image_name)
    if idx==0:
        features, test, center = Feature_point(image)
    else:
        features, test, center = Feature_point(image, center)
    for f in features:
        test[f[0], f[1], 2] = 255
    blended = alpha_blend(image, test, 0.4)
    ax.clear()
    ax.imshow(blended)
    ax.axis('off')
    plt.draw()
    plt.pause(0.01)
# plt.show()
plt.close()

