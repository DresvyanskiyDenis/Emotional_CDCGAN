from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf

from src.EmoCDCGAN.utils.data_preprocessing.preprocess_utils import load_AffectNet_labels, load_image, crop_image, \
    preprocess_image


def visualize_images(images, labels, path_to_save='images', save_name=None):
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    filename_to_save = os.path.join(path_to_save, save_name)
    #plt.figure(figsize=(2.2, 2.2))
    images=(images-images.min())/(images.max()-images.min())
    num_images = images.shape[0]
    image_size = images.shape[1]
    fig = plt.figure(figsize=(13, 9))
    rows=int(np.sqrt(num_images))
    columns=rows
    ax=[]
    for i in range(columns * rows):
        image=images[i]
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("real_class:" + str(labels[i]))  # set title
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(filename_to_save)
    #plt.show()

