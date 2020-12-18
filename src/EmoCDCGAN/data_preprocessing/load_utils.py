import PIL
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

def load_image(path):
    img=Image.open(path)
    img = img.convert('RGB')
    return np.array(img)

def preprocess_image(img, scale=True, resize=True, needed_shape=(224,224,3), bgr=False):
    if resize:
        img=np.array(Image.fromarray(img).resize(needed_shape[:2], resample=PIL.Image.BILINEAR))
    if bgr:
        img=img[...,::-1]
    if scale:
        img=img/255.
    return img

def load_images_from_paths(paths, scale=True, resize=True, images_shape=(224,224,3), bgr=False, preprocess_mode='MobileNetv2'):
    images=np.zeros((len(paths),)+images_shape)
    for i in range(len(paths)):
        img=load_image(paths[i])
        img=preprocess_image(img, scale, resize, images_shape, bgr)
        if preprocess_mode=='MobileNetv2':
            img=tf.keras.applications.mobilenet_v2.preprocess_input(img)
        images[i]=img
    return images

def crop_image(image, x, y, width, height):
    image=image[x:(x+width), y:(y+height)]
    return image


def load_AffectNet_labels(path_to_labels):
    labels=pd.read_csv(path_to_labels, sep=',')
    labels=labels[['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'expression', 'valence', 'arousal']]
    possible_class_values=[i for i in range(7)]
    labels=labels[np.isin(labels.expression, possible_class_values)]
    labels.subDirectory_filePath=labels.subDirectory_filePath.apply(lambda x:x.replace('/','\\'))
    return labels


if __name__ == "__main__":
    load_AffectNet_labels('D:\\Databases\\AffectNet\\AffectNet\\zip\\training.csv')
