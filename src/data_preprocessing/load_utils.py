import numpy as np
import pandas as pd
from PIL import Image


def load_image(path):
    img=Image.open(path)
    img = img.convert('RGB')
    return np.array(img)

def preprocess_image(img, scale=True, resize=True, needed_shape=(224,224,3), bgr=False):
    if resize:
        img=np.array(Image.fromarray(img).resize(needed_shape))
    if bgr:
        img=img[...,::-1]
    if scale:
        img=img/255.
    return img

def load_images_from_paths(paths, scale=True, resize=True, images_shape=(224,224,3), bgr=False):
    images=np.zeros((len(paths),)+images_shape)
    for i in range(len(paths)):
        img=load_image(paths[i])
        img=preprocess_image(img, scale, resize, images_shape, bgr)
        images[i]=img
    return images

