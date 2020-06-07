import os
import urllib.request
import requests


class WorkWithImage:
    @staticmethod
    def create_folder(path: str):
        path = path.split('/')
        for i in range(1, len(path)):
            os.mkdir('/'.join(path[:i]))

    @staticmethod
    def write_image(img, path: str,  filename='image_before.jpg'):
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
