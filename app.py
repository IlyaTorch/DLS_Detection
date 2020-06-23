from flask import Flask, render_template, request
from config import Configuration
import cv2
from src.WorkWithImage import WorkWithImage

import torchvision

app = Flask(__name__)
app.config.from_object(Configuration)
# to disable caching files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def detect_image():
    img = img_url = None
    try:
        img = request.files['image']
    except KeyError:
        try:
            img_url = request.form['image_url']
        except KeyError:
            return render_template('error.html')
    if img:
        WorkWithImage.save_image_from_bytes(img)

    if img_url:
        WorkWithImage.save_image_from_url(img_url)

    img, img_tensor = WorkWithImage.read_image_to_numpy_and_tensor('static/cache/image_before.jpg')
    boxes, labels = WorkWithImage.get_prediction(img_tensor, model)
    img_with_boxes = WorkWithImage.plot_boxes_and_labels(img, boxes, labels)
    cv2.imwrite('static/cache/image_after.jpg', img_with_boxes)

    return render_template('detection_result.html')
