from flask import Flask, render_template, request
from config import Configuration
import cv2
from src.WorkWithImage import WorkWithImage
import torchvision

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


app = Flask(__name__)
app.config.from_object(Configuration)
# to disable caching files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


label_path = 'voc-model-labels.txt'
model_path = 'mb2-ssd-lite-mp-0_686.pth'

class_names = [name.strip() for name in open(label_path).readlines()]
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)


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
    import time
    img, img_tensor = WorkWithImage.read_image_to_numpy_and_tensor('static/cache/image_before.jpg')
    print(img_tensor.shape)
    print('======')
    start_time = time.monotonic()

    # ====================================
    boxes, labels, probs = predictor.predict(img, 10, 0.4)
    # ====================================


    boxes, labels = WorkWithImage.get_prediction(img_tensor, model)
    print(f'{time.monotonic() - start_time} seconds have passed')

    img_with_boxes = WorkWithImage.plot_boxes_and_labels(img, boxes, labels)
    with open('static/cache/image_after.jpg', 'w') as f:
        img_with_boxes.save(f)

    return render_template('detection_result.html')
