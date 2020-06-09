import json
import os
import requests

import cv2  # openCV
import torch


class WorkWithImage:
    COCO_LABELS = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

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
        img = cv2.imread(img_path)[:, :, ::-1]  # invert channels

        # arr of bytes->float->numpy; put dim of channels to 1 place;
        img_tensor = torch.from_numpy(img.astype('float32')).permute(2, 0, 1) / 255

        img_tensor = img_tensor[None, :]  # add axis
        return img, img_tensor

    @staticmethod
    def plot_boxes_and_labels(numpy_img, boxes, labels):
        numpy_img = numpy_img.astype('float32')

        for index, box in enumerate(boxes):
            numpy_img = cv2.rectangle(numpy_img, (box[0], box[1]), (box[2], box[3]), 255, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (box[0], box[1])
            font_scale = numpy_img.shape[1] / 1500.0 + 0.2  # for readability of labels
            color = (255, 0, 0)
            thickness = 2 if font_scale > 0.5 else 1  # for readability of labels
            numpy_img = cv2.putText(numpy_img, labels[index], org, font, font_scale, color,
                                    thickness)

        return numpy_img[:, :, ::-1]
