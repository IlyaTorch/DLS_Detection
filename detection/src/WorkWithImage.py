import os
import requests

import cv2  # openCV
import torch


class WorkWithImage:
    @staticmethod
    def create_folder(path: str):
        path = path.split('/')
        for i in range(1, len(path)):
            os.mkdir('/'.join(path[:i]))

    @staticmethod
    def write_image(img, path: str,  filename='image_before.jpg'):
        out = open(path + filename, "wb")
        out.write(img)
        out.close()

    @staticmethod
    def save_image_from_bytes(img):
        img = img.read()
        path = 'static/cache/'

        if not os.path.exists(path):
            WorkWithImage.create_folder(path)

        WorkWithImage.write_image(img, path)

    @staticmethod
    def save_image_from_url(url):
        path = 'static/cache/'
        img = requests.get(url).content

        if not os.path.exists(path):
            WorkWithImage.create_folder(path)

        WorkWithImage.write_image(img, path)

    @staticmethod
    def read_image_to_numpy_and_tensor(img_path: str):
        img = cv2.imread(img_path)[:, :, ::-1]  # invert channels

        # arr of bytes->float->numpy; put dim of channels to 1 place;
        img_tensor = torch.from_numpy(img.astype('float32')).permute(2, 0, 1) / 255

        img_tensor = img_tensor[None, :]  # add axis
        return img, img_tensor

    @staticmethod
    def plot_boxes(numpy_img, predictions, threshold=0.5):
        numpy_img = numpy_img.astype('float32')

        # prediction — dict, key 'boxes' — list of coordinates of boxes; filter predictions
        boxes = predictions['boxes'][predictions['scores'] > threshold]

        boxes = boxes.detach().numpy()  # detach from the graph of calculations
        for box in boxes:
            numpy_img = cv2.rectangle(numpy_img, (box[0], box[1]), (box[2], box[3]), 255, 3)
        return numpy_img[:, :, ::-1]
