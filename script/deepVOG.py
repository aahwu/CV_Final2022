from DeepVOG_model import load_DeepVOG
import numpy as np
import os
import cv2
from tqdm import tqdm

def deepVOG(dataset_path: str, subjects: list, conf_thres: int, nn_thres: int=0.5):

    # load model
    model = load_DeepVOG()

    sequence_idx = 0
    for subject in subjects:

        solution_dataset = os.path.join(dataset_path, subject+"_vog")
        if os.path.exists(solution_dataset) != True:
            os.mkdir(solution_dataset)

        for action_number in range(26):
            sequence_idx += 1

            # folders path
            preimage_folder = os.path.join(dataset_path, subject+"_preimage", f'{action_number + 1:02d}')
            solution_folder = os.path.join(dataset_path, subject+"_vog", f'{action_number + 1:02d}')

            # check whether folders exist
            if os.path.exists(solution_folder) != True:
                os.mkdir(solution_folder)
            if os.path.exists(preimage_folder) != True:
                os.mkdir(preimage_folder)
            
            # create and clear conf.txt, conf_real.txt
            conf_name = os.path.join(solution_folder, 'conf_vog.txt')
            open(conf_name, 'w').close()
            conf_real_name = os.path.join(solution_folder, 'conf_real.txt')
            open(conf_real_name, 'w').close()
            area_name = os.path.join(solution_folder, 'area.txt')
            open(area_name, 'w').close()

            nr_image = len([name for name in os.listdir(preimage_folder) if name.endswith('.jpg')])
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {preimage_folder}'):
                
                # file path
                label_name = os.path.join(solution_folder, f'{idx}.png')
                preimage_name = os.path.join(preimage_folder, f'{idx}.jpg')

                # load preimage
                image = cv2.imread(preimage_name)

                # inference
                image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_LINEAR)
                img = np.zeros((1, 240, 320, 3))
                img[:,:,:,:] = (image[:, :, 0]/255).reshape(1, 240, 320, 1)
                prediction = model.predict(img)

                # label
                # --threshold
                prediction_thres = prediction > nn_thres
                # --output
                prediction_label = prediction_thres.astype(np.uint8) * 255
                prediction_label = cv2.resize(prediction_label[0,:,:,:], (640, 480), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(label_name, prediction_label[:,:,1])
                area = np.count_nonzero(prediction_label[:,:,1])
                
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
                with open(conf_real_name, 'a') as f:
                    f.write(str(conf_real) + '\n')
                with open(conf_name, 'a') as f:
                    f.write(str(conf) + '\n')
                # --remove conf_real.txt for submission
                # if os.path.exists(conf_real_name) == True:
                #     os.remove(conf_real_name)
                
                # area
                with open(area_name, 'a') as f:
                    f.write(str(area) + '\n')