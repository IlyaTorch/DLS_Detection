import os

import cv2
import requests

# import cv2  # openCV
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch


class WorkWithImage:
    @staticmethod
    def create_folder(path: str):
        path = path.split('/')
        for i in range(1, len(path)):
            os.mkdir('/'.join(path[:i]))

    @staticmethod
    def write_image(img, path: str, filename='image_before.jpg'):
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
    def read_from_path_to_numpy(img_path: str):
        original_image = cv2.imread(img_path)
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return original_image, image

    @staticmethod
    def add_prediction_to_image(image, boxes, labels, probs, class_names):
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        return image
