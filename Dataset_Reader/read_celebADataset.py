__author__ = 'charlie'
import numpy as np
import os, sys, inspect
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

utils_path = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import utils as utils

DATA_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip'
random.seed(5)


class CelebA_Dataset():
    def __init__(self, dict):
        self.train_images = dict['train']
        self.test_images = dict['test']
        self.validation_images = dict['validation']


def read_dataset(data_dir):
    pickle_filename = "celebA.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        # utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        celebA_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        dir_path = os.path.join(data_dir, celebA_folder)
        if not os.path.exists(dir_path):
            print ("CelebA dataset needs to be downloaded and unzipped manually")
            print ("Download from: %s" % DATA_URL)
            raise ValueError("Dataset not found")

        result = create_image_lists(dir_path)
        print ("Training set: %d" % len(result['train']))
        print ("Test set: %d" % len(result['test']))
        print ("Validation set: %d" % len(result['validation']))
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        celebA = CelebA_Dataset(result)
        del result
    return celebA


def create_image_lists(image_dir, testing_percentage=0.0, validation_percentage=0.0):
    """
    Code modified from tensorflow/tensorflow/examples/image_retraining
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    training_images = []
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    file_list = []

    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(glob.glob(file_glob))

    if not file_list:
        print('No files found')
    else:
        # print "No. of files found: %d" % len(file_list)
        training_images.extend([f for f in file_list])

    random.shuffle(training_images)
    no_of_images = len(training_images)
    validation_offset = int(validation_percentage * no_of_images)
    validation_images = training_images[:validation_offset]
    test_offset = int(testing_percentage * no_of_images)
    testing_images = training_images[validation_offset:validation_offset + test_offset]
    training_images = training_images[validation_offset + test_offset:]

    result = {
        'train': training_images,
        'test': testing_images,
        'validation': validation_images,
    }
    return result
