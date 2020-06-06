from flask import Blueprint, render_template, request
from .src.WorkWithImage import WorkWithImage

detection = Blueprint('detection', __name__, template_folder='templates')


@detection.route('/', methods=['GET'])
def index():
    return render_template('detection/index.html')


@detection.route('/upload', methods=['POST'])
def detect_image():
    img = request.files['image']
    WorkWithImage.save_image(img)

    return render_template('detection/detection_result.html')
