import os
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
    def get_prediction(img_tensor, model, threshold=0.5):
        prediction = model(img_tensor)
        # prediction — dict, key 'boxes' — list of coordinates of boxes; filter predictions
        boxes = prediction[0]['boxes'][prediction[0]['scores'] > threshold]
        classes = prediction[0]['labels'][prediction[0]['scores'] > threshold]
        boxes = boxes.detach().numpy()  # detach from the graph of calculations
        labels = [WorkWithImage.COCO_LABELS[index.item()] for index in classes]
        return boxes, labels

    @staticmethod
    def read_image_to_numpy_and_tensor(img_path: str):
        img = Image.open(img_path)
        print(img.size)
        print('**********')
        transform = transforms.Compose([
            transforms.Resize(800),
            transforms.CenterCrop(800),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img)

        img_tensor = img_tensor[None, :]  # add axis
        return img, img_tensor

    @staticmethod
    def plot_boxes_and_labels(img, boxes, labels):
        draw = ImageDraw.Draw(img)

        for index, box in enumerate(boxes):
            draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline='red', width=2)

            size = 30 if img.size[1] > 500 else 10
            font = ImageFont.truetype('arial.ttf', size)
            draw.text((box[0], box[1]), labels[index], fill='red', font=font)

        return img
