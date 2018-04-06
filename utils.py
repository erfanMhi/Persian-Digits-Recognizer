from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import random
from PIL import Image
from IPython.display import display
from scipy import ndimage

image_size = 28
pixel_depth = 255.0

def load_letter(folder, min_num_images=0):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                            dtype=np.float32)
    num_images = 0
    for image in image_files :
        image_file = os.path.join(folder, image)
        try:
            image_data = (np.array(Image.open(os.path.join(folder, image_files[2])).resize((image_size,image_size))).astype(float) - 
                            pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        
    dataset = dataset[0:num_images, :, :]
    if num_images < 10:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))
        
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_save(root, min_num_images_per_class=0, force=False):
  data_folders = os.listdir(root)
  dataset_names = []
  for folder in data_folders:
    set_filename = root + folder + '.npy'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(root + folder, min_num_images_per_class)
      try:
        np.save(set_filename,dataset)   
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names