import numpy as np
import pandas as pd
import os

from numpy.core.multiarray import ndarray
from pandas import DataFrame

from src.EmoCDCGAN.utils.data_preprocessing.preprocess_utils import load_image, preprocess_image, load_AffectNet_labels


def pack_and_save_batch_images_in_npy(images: ndarray, labels: DataFrame, path_to_save:str, filename:str):
    """ save images and labels to .npy and csv files. Images will be in unit8 format

    :param images: ndarray
                 ndarray of images
    :param labels: pandas DataFrame
                 DataFrame of labels to corresponding images
    :param path_to_save: str
                 path for saving files
    :param filename:
                 filename for saving files: can be number of batch, etc.
    :return: None
    """
    images=images.astype('uint8')
    np.save(os.path.join(path_to_save, filename+'.npy'), images)
    labels.to_csv(os.path.join(path_to_save, filename+'.csv'), index=False)

def pack_all_images_in_batches_AffectNet(labels: DataFrame, path_to_data:str,  batch_size:int, path_to_save:str, size_of_images:tuple=(224,224,3)):
    """ Pack all images, paths of which are available in labels "path" column into .npy files with corresponding
        DataFrame labels. The data will be divided on batches.

    :param labels: DataFrame
                 pandas DataFrame of labels and paths to corresponding images
    :param path_to_data: str
                 path to data location
    :param batch_size: int
                 size of batches, on which data will be divided
    :param path_to_save: str
                 path to save data and labels
    :param size_of_images: tuple
                 required sizes of image
    :return: None
    """
    labels = labels[['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'expression', 'valence', 'arousal']]
    num_batch=0
    for i in range(0,labels.shape[0],batch_size):
        start_idx=i
        end_idx=start_idx+batch_size if start_idx+batch_size<=labels.shape[0] else labels.shape[0]
        # variables for image loading
        labels_to_pack=labels.iloc[start_idx:end_idx]
        data_to_pack=np.zeros((batch_size,)+size_of_images)
        batch_idx=0
        # load images per batch
        for labels_idx in range(start_idx, end_idx):
            image_filename=labels.iloc[labels_idx]['subDirectory_filePath']
            image=load_image(os.path.join(path_to_data, image_filename))
            image=preprocess_image(img=image, scale=False, resize=True, needed_shape=size_of_images, bgr=False)
            data_to_pack[batch_idx]=image
            batch_idx+=1
        # pack images
        pack_and_save_batch_images_in_npy(images=data_to_pack, labels=labels_to_pack,
                                          path_to_save=path_to_save, filename='batch_num_%i'%(num_batch))
        num_batch+=1
        # clear RAM
        del data_to_pack

def unpack_data_and_labels_npy(path_to_folder, filename):
    data=np.load(os.path.join(path_to_folder, filename+'.npy'), allow_pickle=True)
    labels=pd.read_csv(os.path.join(path_to_folder, filename+'.csv'))
    return data, labels


if __name__ == "__main__":
    path_to_images='E:\\Databases\\AffectNet\\AffectNet\\zip\\Manually_Annotated_Images'
    path_to_labels='E:\\Databases\\AffectNet\\AffectNet\\zip\\training.csv'
    path_to_save='E:\\Databases\\AffectNet\\AffectNet\\Batches'
    labels=load_AffectNet_labels(path_to_labels)
    pack_all_images_in_batches_AffectNet(
        labels=labels, path_to_data=path_to_images, batch_size=64, path_to_save=path_to_save, size_of_images = (224, 224, 3))