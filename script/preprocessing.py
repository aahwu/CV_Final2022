import numpy as np
import os
import cv2
from tqdm import tqdm
    
def enhance_contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def brightness(img, mul):

    img = img.astype(float)
    image_mul = mul - img*((mul-1)/255)
    brightened_img = img * image_mul
    image_thres = brightened_img <= 255
    brightened_img = brightened_img*image_thres + 255*(1-image_thres)

    return brightened_img

def preprocessing(dataset_path: str, subjects: list):

    sequence_idx = 0
    for subject in subjects:

        preimage_dataset = os.path.join(dataset_path, subject+"_preimage")
        if os.path.exists(preimage_dataset) != True:
            os.mkdir(preimage_dataset)

        for action_number in range(26):
            sequence_idx += 1

            # folders path
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            preimage_folder = os.path.join(dataset_path, subject+"_preimage", f'{action_number + 1:02d}')

            # check whether folders exist
            if os.path.exists(preimage_folder) != True:
                os.mkdir(preimage_folder)

            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                
                # file path
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                preimage_name = os.path.join(preimage_folder, f'{idx}.jpg')

                # load image
                image = cv2.imread(image_name)

                # preprocessing
                # --brightness
                # image = brightness(image)

                # --contrastion
                image = enhance_contrast(image)
                # --output
                image = image.astype(np.uint8)
                cv2.imwrite(preimage_name, image)