"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc


class Dataset():
    files = []
    images = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, file_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param file_list: list of files to read - filepaths
        :param image_size: Desired output image size
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        crop = True/ False
        crop_size = #size smaller than image - does central crop of square shape
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = file_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.images = np.array([self._transform(filename) for filename in self.files])
        print (self.images.shape)

    def _center_crop(self, x):
        crop_size = int(self.image_options["crop_size"])
        h, w = x.shape[:2]
        j = int(round((h - crop_size) / 2.))
        i = int(round((w - crop_size) / 2.))
        return x[j:j + crop_size, i:i + crop_size]

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.image_options.get("crop", False) and self.image_options["crop"]:
            cropped_image = self._center_crop(image)
        else:
            cropped_image = image

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(cropped_image,
                                         [resize_size, resize_size])
        else:
            resize_image = cropped_image

        return np.array(resize_image) / 127.5 - 1.

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            # perm = np.arange(self.images.shape[0])
            # np.random.shuffle(perm)
            # self.images = self.images[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end]
