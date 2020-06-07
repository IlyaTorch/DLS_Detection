from flask import Blueprint, render_template, request
from .src.WorkWithImage import WorkWithImage


detection = Blueprint('detection', __name__, template_folder='templates')


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
        return render_template('detection/detection_result.html')

    if img_url:
        WorkWithImage.save_image_from_url(img_url)
        return render_template('detection/detection_result.html')

    return render_template('detection/error.html')
