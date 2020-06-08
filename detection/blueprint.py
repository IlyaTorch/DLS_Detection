import cv2
from flask import Blueprint, render_template, request, jsonify, url_for
from .src.WorkWithImage import WorkWithImage

import torchvision
import torch
import numpy as np

detection = Blueprint('detection', __name__, template_folder='templates', static_folder='static')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval()


@detection.route('/', methods=['GET'])
def index():
    return render_template('detection/index.html')


@detection.route('/upload', methods=['POST'])
def detect_image():
    img = img_url = None
    try:
        img = request.files['image']
    except KeyError:
        try:
            img_url = request.form['image_url']
        except KeyError:
            return render_template('detection/error.html')
    if img:
        WorkWithImage.save_image_from_bytes(img)

    if img_url:
        WorkWithImage.save_image_from_url(img_url)

    img, img_tensor = WorkWithImage.read_image_to_numpy_and_tensor('static/cache/image_before.jpg')
    predictions = model(img_tensor)
    img_with_boxes = WorkWithImage.plot_boxes(img, predictions[0])
    cv2.imwrite('static/cache/image_after.jpg', img_with_boxes)

    return render_template('detection/detection_result.html', predictions=predictions[0])
