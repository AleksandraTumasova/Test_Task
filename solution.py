import numpy as np
import os
from PIL import Image


def open_img(path):
    img_names = [im for im in os.listdir(path)]
    imgs = [Image.open(os.path.join(path, name)) for name in img_names]

    return img_names, imgs


def make_dict(img_names, imgs):
    dict_img = {img: i for img, i in zip(img_names, imgs)}

    return dict_img


def corr(v1, v2):
    return 1 - np.dot((v1 - v1.mean()), (v2 - v2.mean())) / (
                np.linalg.norm(v1 - v1.mean()) * np.linalg.norm(v2 - v2.mean()))


def Duplicate(img1, img2):
    return np.array_equal(img1, img2)


def make_flatten(img):
    img = np.array(img)
    return img.flatten()


def check_modified(img1, img2, dict_img):
    if (corr(make_flatten(dict_img[img1]), make_flatten(dict_img[img2])) < 0.1):
        return True
    else:
        return False


if __name__ == '__main__':

    path = "../img_project/dev_dataset"


    image_names, images = open_img(path)
    dict_img = make_dict(image_names, images)

    duplicates = []
    modified = []

    for img1 in dict_img.keys():

        for img2 in dict_img.keys():
            if (img1 != img2):

                if Duplicate(dict_img[img1], dict_img[img2]):
                    if not [img2, img1] in duplicates:
                        duplicates.append([img1, img2])
                        continue

                if [img2, img1] in modified:
                    continue

                if not np.array_equal(dict_img[img1].size, dict_img[img2].size):
                    dict_img[img2] = dict_img[img2].resize(dict_img[img1].size)

                if check_modified(img1, img2, dict_img):
                    modified.append([img1, img2])

    print(modified, duplicates)

