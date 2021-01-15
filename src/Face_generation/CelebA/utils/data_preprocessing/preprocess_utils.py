import PIL
import numpy as np
from PIL import Image
from numpy.core.multiarray import ndarray


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
        img=(img/255.)*2.-1
    return img

def preprocess_batch_images(images:ndarray, scale:bool=True, resize:bool=True, images_shape:tuple=(224,224,3), bgr:bool=False):
    if resize:
        new_images=np.zeros(shape=(images.shape[0],)+images_shape).astype('float32')
    else:
        new_images=images.astype('float32')
    for i in range(images.shape[0]):
        new_images[i]=preprocess_image(images[i], scale, resize, images_shape, bgr)
    new_images=new_images.astype('float32')
    return new_images

def load_images_from_paths(paths, scale=True, resize=True, images_shape=(224,224,3), bgr=False):
    images=np.zeros((len(paths),)+images_shape)
    for i in range(len(paths)):
        img=load_image(paths[i])
        img=preprocess_image(img, scale, resize, images_shape, bgr)
        images[i]=img
    return images

def crop_image(image, x, y, width, height):
    image=image[x:(x+width), y:(y+height)]
    return image

def shuffle_ndarrays(ndarrays:list):
    permutation=np.random.permutation(ndarrays[0].shape[0])
    for i in range(len(ndarrays)):
        ndarrays[i]=ndarrays[i][permutation]
    return ndarrays

def add_noise_in_labels(labels:ndarray):
    for i in range(labels.shape[0]):
        if labels[i]==0.:
            labels[i]=np.random.uniform(0.,0.1)
        elif labels[i]==1.:
            labels[i]=np.random.uniform(0.9, 1.0)
    return labels
if __name__ == "__main__":
    #load_AffectNet_labels('D:\\Databases\\AffectNet\\AffectNet\\zip\\training.csv')
    pass
