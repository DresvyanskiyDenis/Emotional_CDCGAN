import numpy as np
import pandas as pd
import os

from numpy.core.multiarray import ndarray
from pandas import DataFrame

from src.Face_generation.CelebA.utils.data_preprocessing.preprocess_utils import load_image, preprocess_image


def pack_and_save_batch_images_in_npy(images: ndarray, path_to_save:str, filename:str):
    """ save images to .npy files. Images will be in unit8 format

    :param images: ndarray
                 ndarray of images

    :param path_to_save: str
                 path for saving files
    :param filename:
                 filename for saving files: can be number of batch, etc.
    :return: None
    """
    images=images.astype('uint8')
    np.save(os.path.join(path_to_save, filename+'.npy'), images)

def pack_all_images_in_batches_CelebA(path_to_data:str,  batch_size:int, path_to_save:str, size_of_images:tuple=(128,128,3)):
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
    filenames=np.array(os.listdir(path_to_data))
    num_batch=0
    for i in range(0,filenames.shape[0],batch_size):
        start_idx=i
        end_idx=start_idx+batch_size if start_idx+batch_size<=filenames.shape[0] else filenames.shape[0]
        # variables for image loading
        data_to_pack=np.zeros((batch_size,)+size_of_images)
        batch_idx=0
        # load images per batch
        for j in range(start_idx, end_idx):
            image_filename=filenames[j]
            image=load_image(os.path.join(path_to_data, image_filename))
            image=preprocess_image(img=image, scale=False, resize=False, needed_shape=size_of_images, bgr=False)
            data_to_pack[batch_idx]=image
            batch_idx+=1
        # pack images
        pack_and_save_batch_images_in_npy(images=data_to_pack, path_to_save=path_to_save, filename='batch_num_%i'%(num_batch))
        num_batch+=1
        # clear RAM
        del data_to_pack

def unpack_data_npy(path_to_folder, filename):
    data=np.load(os.path.join(path_to_folder, filename+'.npy'), allow_pickle=True)
    return data

if __name__ == "__main__":
    path_to_data='C:\\Users\\Dresvyanskiy\\Downloads\\archive\\100k\\100k'
    path_to_save='C:\\Users\\Dresvyanskiy\\Downloads\\archive\\Batches\\'
    pack_all_images_in_batches_CelebA(path_to_data=path_to_data, batch_size=64, path_to_save=path_to_save)