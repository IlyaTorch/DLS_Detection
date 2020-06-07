from flask import Flask
from config import Configuration
from detection.blueprint import detection

app = Flask(__name__)
app.config.from_object(Configuration)
# to disable caching files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.register_blueprint(detection, url_prefix='/detection')
