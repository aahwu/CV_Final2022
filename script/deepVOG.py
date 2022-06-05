from DeepVOG_model import load_DeepVOG
import skimage.io as ski
import numpy as np
import os
import cv2
from tqdm import tqdm
    
def enhance_contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def result(dataset_path: str, subjects: list, conf_thres: int, nn_thres: int=0.5):

    # load model
    model = load_DeepVOG()

    sequence_idx = 0
    for subject in subjects:
        for action_number in range(26):
            sequence_idx += 1

            # folders path
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            preimage_folder = os.path.join(dataset_path, subject+"_preimage", f'{action_number + 1:02d}')
            solution_folder = os.path.join(dataset_path, subject+"_solution", f'{action_number + 1:02d}')

            # check whether folders exist
            if os.path.exists(solution_folder) != True:
                os.mkdir(solution_folder)
            if os.path.exists(preimage_folder) != True:
                os.mkdir(preimage_folder)
            
            # create and clear conf.txt, conf_real.txt
            conf_name = os.path.join(solution_folder, 'conf.txt')
            open(conf_name, 'w').close()
            conf_real_name = os.path.join(solution_folder, 'conf_real.txt')
            open(conf_real_name, 'w').close()

            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                
                # file path
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                label_name = os.path.join(solution_folder, f'{idx}.png')
                preimage_name = os.path.join(preimage_folder, f'{idx}.jpg')

                # load image
                image = cv2.imread(image_name)

                # preprocessing
                # --brightness
                # mul = 4
                # image = image.astype(float)
                # image_mul = mul - image*((mul-1)/255)
                # image *= image_mul
                # image_thres = image <= 255
                # image = image*image_thres + 255*(1-image_thres)

                # --contrastion
                image = enhance_contrast(image)

                # inference
                image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_LINEAR)
                img = np.zeros((1, 240, 320, 3))
                img[:,:,:,:] = (image[:, :, 0]/255).reshape(1, 240, 320, 1)
                prediction = model.predict(img)
                # --output
                image = image.astype(np.uint8)
                cv2.imwrite(preimage_name, image)

                # label
                # --threshold
                prediction_thres = prediction > nn_thres
                # --output
                prediction_label = prediction_thres.astype(np.uint8) * 255
                prediction_label = cv2.resize(prediction_label[0,:,:,:], (640, 480), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(label_name, prediction_label[:,:,1])
                
                # confidence
                # --average probability of label
                prediction *= prediction_thres
                if np.sum(prediction_thres[0,:,:,1])==0:
                    conf_real = 0
                else:
                    conf_real = np.sum(prediction[0,:,:,1])/np.sum(prediction_thres[0,:,:,1])
                # --threshold
                if conf_real < conf_thres:
                    conf = 0
                else:
                    conf = 1
                # --output
                # with open(conf_real_name, 'a') as f:
                #     f.write(str(conf_real) + '\n')
                with open(conf_name, 'a') as f:
                    f.write(str(conf) + '\n')
                # --remove conf_real.txt for submission
                if os.path.exists(conf_real_name) == True:
                    os.remove(conf_real_name)

if __name__ == '__main__':

    dataset_path = r'C:\Users\ShaneWu\OneDrive\Desktop\Hw(senior)\CV\Final\CV22S_Ganzin\dataset\public'
    subjects = ['S5']
    result(dataset_path, subjects, conf_thres=0.7)