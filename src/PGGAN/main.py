import tensorflow as tf

from src.PGGAN.pggan import PGGAN

if __name__ == '__main__':
    #params
    batch_size = 64
    latent_space_shape=100
    steps_per_block = [5,5,5,5,8000,8000]
    batch_per_block=[64,64,64,64,64,64]
    first_image_size=(4,4,3)
    last_image_size=(128,128,3)

    path_to_data='D:\\Databases\\AffectNet\\AffectNet\\Batches'

    pggan=PGGAN(latent_space_shape=latent_space_shape, first_image_size=first_image_size, last_image_size=last_image_size)
    pggan.create_generators()
    pggan.create_discriminators()
    pggan.create_advesarials()
    pggan.train_process(steps_per_block, batch_per_block, path_to_data)