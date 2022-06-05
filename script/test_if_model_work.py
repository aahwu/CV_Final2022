from DeepVOG_model import load_DeepVOG
import skimage.io as ski
import numpy as np
import os
import cv2


def test_if_model_work():
    model = load_DeepVOG()
    dataset_path = r'C:\Users\ShaneWu\OneDrive\Desktop\Hw(senior)\CV\Final\CV22S_Ganzin\model'
    img = np.zeros((1, 240, 320, 3))
    image = cv2.imread(os.path.join(dataset_path, "S5_01_97.jpg"))
    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_LINEAR)
    image = image.astype(float)
    image *= 2
    image_thres = image <= 255
    image = image*image_thres + 255*(1-image_thres)
    img[:,:,:,:] = (image/255)[:, :, 0].reshape(1, 240, 320, 1)
    prediction = model.predict(img)
    prediction_t = prediction > 0.5
    prediction *= prediction_t
    print(np.sum(prediction[0,:,:,1])/np.sum(prediction_t[0,:,:,1]))
    prediction_t = prediction_t.astype(np.uint8)
    prediction_t *= 255
    prediction_t = cv2.resize(prediction_t[0,:,:,:], (640, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(dataset_path, "97_prediction.png"), prediction_t[:,:,1])
    cv2.imwrite(os.path.join(dataset_path, "97_image_new.png"), image)

if __name__ == "__main__":
    # If model works, the "test_prediction.png" should show the segmented area of pupil from "test_image.png"
    test_if_model_work()

