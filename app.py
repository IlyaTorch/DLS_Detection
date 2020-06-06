from flask import Flask
from config import Configuration
from detection.blueprint import detection

app = Flask(__name__)
app.config.from_object(Configuration)

app.register_blueprint(detection, url_prefix='/detection')
