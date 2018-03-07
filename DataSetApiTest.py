import tensorflow as tf
import os

PICTURE_PATH = "F:/Traindata/MNIST_pictures/trainimage/pic2/"
NUM_CLASS = 10


def load_image():
    all_classes = []
    all_images = []
    all_labels = []

    for i in os.listdir(PICTURE_PATH):
        curren_dir = os.path.join(PICTURE_PATH, i)
        if os.path.isdir(curren_dir):
            all_classes.append(curren_dir)
            for img in os.listdir(curren_dir):
                if img.endswith('png') or img.endswith('bmp') or img.endswith('jpg'):
                    all_images.append(os.path.join(curren_dir, img))
                    all_labels.append(all_classes.index(i))
        else:
            print(curren_dir, " doesnt exist")

    return all_classes, all_images, all_labels
