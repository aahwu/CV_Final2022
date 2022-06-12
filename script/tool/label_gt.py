import os
import numpy as np
import cv2
from tqdm import tqdm
from utils import AverageMeter

def label_gt(dataset_path: str, subjects: list):
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    sequence_idx = 0
    for subject in subjects:
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            label_name = os.path.join(image_folder, '0.png')
            if not os.path.exists(label_name):
                print(f'Labels are not available for {image_folder}')
                continue
            conf_gt = os.path.join(image_folder, 'conf_gt.txt')
            open(conf_gt, 'w').close()
            # for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
            for idx in range(nr_image):
                label_name = os.path.join(image_folder, f'{idx}.png')
                label = cv2.imread(label_name)
                # TODO: Modify the code below to run your method or load your results from disk
                # output, conf = my_awesome_algorithm(image)
                if np.sum(label.flatten()) > 0:
                    conf = 1
                else:  # empty ground truth label
                    conf = 0
                with open(conf_gt, 'a') as f:
                    f.write(str(conf) + '\n')



if __name__ == '__main__':
    dataset_path = r'C:\Users\ShaneWu\OneDrive\Desktop\Hw(senior)\CV\Final\CV22S_Ganzin\dataset\public'
    subjects = ['S4']
    label_gt(dataset_path, subjects)
