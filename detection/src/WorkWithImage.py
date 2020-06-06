import os


class WorkWithImage:
    @staticmethod
    def save_image(img):
        img = img.read()
        path = 'static/cache/'

        if not os.path.exists(path):
            os.mkdir(path)

        out = open(path + 'image_before.jpg', "wb")
        out.write(img)
        out.close()
