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
            all_classes.append(i)
            for img in os.listdir(curren_dir):
                if img.endswith('png') or img.endswith('bmp') or img.endswith('jpg'):
                    all_images.append(os.path.join(curren_dir, img))
                    all_labels.append(all_classes.index(i))
        else:
            print(curren_dir, " doesnt exist")

    return all_classes, all_images, all_labels


def input_map_fn(image_path, label):
    one_hot = tf.one_hot(label, NUM_CLASS)
    image_f = tf.read_file(image_path)
    image_decode = tf.image.decode_image(image_f, channels=3)
    return image_decode, one_hot


classes, image_path, labels = load_image()
dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))
train_data = dataset.map(input_map_fn)  # 参数是处理dataset的方法

#dataset.shuffle(buffersize=1000).batch(32).repeat(10)的功能是：在每个epoch内将图片打乱组成大小为32的batch，并重复10次。
train_data = train_data.shuffle(buffer_size=100).batch(5)

result_log = tf.reshape(train_data,[5,28,28,3])
