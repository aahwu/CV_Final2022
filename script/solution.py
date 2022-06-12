import os
import numpy as np
import cv2

def combine(dataset_path: str, subjects: list):
    
    sequence_idx = 0
    for subject in subjects:
        for action_number in range(26):
            sequence_idx += 1

            solution_folder = os.path.join(dataset_path, subject+"_solution", f'{action_number + 1:02d}')
            starburst_folder = os.path.join(dataset_path, subject+"_starburst", f'{action_number + 1:02d}')
            vog_folder = os.path.join(dataset_path, subject+"_vog", f'{action_number + 1:02d}')

            conf_name = os.path.join(solution_folder, 'conf.txt')
            area_name = os.path.join(solution_folder, 'area.txt')
            conf_star_name = os.path.join(starburst_folder, 'conf_star.txt')
            conf_vog_name = os.path.join(vog_folder, 'conf_vog.txt')
            
            area_name = os.path.join(solution_folder, 'area.txt')
            area_star_name = os.path.join(starburst_folder, 'area.txt')
            area_vog_name = os.path.join(vog_folder, 'area.txt')
            
            open(conf_name, 'w').close()
            with open(conf_star_name, 'r') as f:
                confs_star = f.readlines()
            with open(conf_vog_name, 'r') as f:
                confs_vog = f.readlines()
                
            open(area_name, 'w').close()
            with open(area_star_name, 'r') as f:
                areas_star = f.readlines()
            with open(area_vog_name, 'r') as f:
                areas_vog = f.readlines()

            nr_image = len([name for name in os.listdir(starburst_folder) if name.endswith('.png')])
            for idx in range(nr_image):
                solution_name = os.path.join(solution_folder, f'{idx}.png')
                star_name = os.path.join(starburst_folder, f'{idx}.png')
                vog_name = os.path.join(vog_folder, f'{idx}.png')
                star_label = cv2.imread(star_name)
                vog_label = cv2.imread(vog_name)
                h, w = star_label.shape[0], star_label.shape[1]
                nonLabel = np.zeros([h, w, 3])

                star_conf = float(confs_star[idx])
                vog_conf = float(confs_vog[idx])
                star_area = float(areas_star[idx])
                vog_area = float(areas_vog[idx])

                area_ratio = vog_area / star_area
                
                # with open(area_name, 'a') as f:
                #     f.write(f'{area_ratio:.4f}\n')

                if star_conf==0:
                    if vog_conf==0:
                        cv2.imwrite(solution_name, nonLabel)
                    else:
                        cv2.imwrite(solution_name, vog_label)
                else:
                    if area_ratio > 2:
                        cv2.imwrite(solution_name, star_label)
                    else:
                        cv2.imwrite(solution_name, vog_label)

                conf = star_conf * vog_conf

                with open(conf_name, 'a') as f:
                    f.write(f'{conf:.4f}\n')
                    
                if os.path.exists(area_name) == True:
                    os.remove(area_name)




if __name__ == '__main__':
    dataset_path = r'C:\Users\ShaneWu\OneDrive\Desktop\Hw(senior)\CV\Final\CV22S_Ganzin\dataset\public'
    subjects = ['S5']
    combine(dataset_path, subjects)