import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import skvideo.io
import os
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
fights=[]
import io
import imageio
import ipywidgets
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K


def frame_crop_center(video, cropf):
    ## the shape is  the number of frames, the number of rows, the number of columns, and the number of channels.
    f, _, _, _ = video.shape
    ## this formula can  calculate the center fo video number of frames.
    # the f is the all the video number of frames, for the examle the f=10. it's have a image is 10 number of frams.
    # the f//2 is 5 , if only we want 2 number of frames . so it's start point is 5-(2/2)=4  so from the startf is frome 4th.
    # it's mean keep center position not change , insted from the privous position to the center . so we can guss ,if also the from the center to the after this formula , we can write the " startf = f//2“ the end position " endf=f//2+cropf//2"

    # also have like this formula :
    # image center cropf . start_x = (w // 2) - (crop_w // 2)  ,start_y = (h // 2) - (crop_h // 2)
    # center scale : new_w = int(w * scale)
    #     new_h = int(h * scale)
    #     start_x = (new_w // 2) - (w // 2)
    #     start_y = (new_h // 2) - (h // 2)

    # boader cropf :start_x = 50
    #                 start_y = 50
    # end_x = 800 - 50 = 750
    # end_y = 600 - 50 = 550

    startf = f // 2 - cropf // 2
    ## this mean ,only keep the video form the startf to cropf ,this is the frist the positon behalf. for : : ; is mean
    # keep the video the number of frames and number fo colums and....
    return video[startf:startf + cropf, :, :, :]


fights = []
nofights = []

surv_fights = []
surv_no_fights = []

video_dims = []

####### Fight data
## this the formual is for the files

## excepet the listdir funciton ,there have this
'''
    os.scandir : return the iter and can loop for the files and subcatalog

    directory = '/path/to/directory'
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                print(entry.name)


    glob.glob:
        file_list = glob.glob('/path/to/directory/*.txt') # note *
    for file_path in file_list:
        print(file_path)

    from pathlib import Path

    directory = Path('/path/to/directory')
    file_list = directory.glob('*.txt')
    for file_path in file_list:
        print(file_path)


'''

for filename in os.listdir(r'D:\python\video_transformer\fight-detection-surv-dataset\fight'):
    print(filename)
    # os.path.abspath: Returns the absolute path of the given path. It converts a relative path to an absolute path based on the current working directory.

    # os.path.dirname: Returns the directory part of the given path. It extracts the directory component from a path and returns the directory's path.
    #
    # os.path.basename: Returns the file name part of the given path. It extracts the file name from a path and returns the file name.
    #
    # os.path.splitext: Splits the given path into the file name and extension tuple. It splits a path into the file name and extension, and returns a tuple where the first element is the file name and the second element is the extension (including the dot).
    #
    # os.path.exists: Checks if the given path exists. It returns a boolean value indicating whether the given path exists.
    #
    # os.path.isdir: Checks if the given path is a directory. It returns a boolean value indicating whether the given path is a directory.
    #
    # os.path.isfile: Checks if the given path is a file. It returns a boolean value indicating whether the given path is a file.

    f = os.path.join(r'D:\python\video_transformer\fight-detection-surv-dataset\fight', filename)
    # checking if it is a file

    video = skvideo.io.vread(f)
    video_dims.append(video.shape)
    L = []

    # resize video dimensions
    ## video.shape[0] is amount  number of frames
    for i in range(video.shape[0]):
        '''
      cv2.INTER_NEAREST   Nearest Neighbor Interpolation:- Fast computation<br>- Suitable for image upscaling	- May cause aliasing or jagged artifacts at image edges<br>- Relatively lower image quality
  
      cv2.INTER_LINEAR  Bilinear Interpolation	- Provides smoother image results compared to nearest neighbor interpolation<br>- Preserves overall structure of the image	- May result in blurriness or loss of details in some cases
  
  
      cv2.INTER_AREA  Area Interpolation	- Suitable for image downsampling<br>- Helps reduce loss of image details	- May lead to blurriness when upscaling images
  
     cv2.INTER_LANCZOS4  Lanczos Interpolation	- Provides better smoothing effect<br>- Helps reduce aliasing or jagged artifacts during image upscaling<br>- Preserves image details well	- Computationally expensive and relatively slower
  
      Cubic Interpolation	- Offers better smoothing and detail preservation compared to bilinear interpolation<br>- Reduces blurring and retains finer image details	- Computationally more expensive than bilinear interpolation<br>- Can introduce some ringing artifacts
  
  
      the best one is the cubic interpolation   cv2.INTER_CUBIC
      '''

        frame = cv2.resize(video[i], (128, 128), interpolation=cv2.INTER_CUBIC)
        L.append(frame)

    '''
    why there have not use the arry() use the asarray ? 
    np.array() is a common function for creating NumPy arrays, while np.asarray() can be selected to suit your needs, especially when you need to preserve data types or manipulate existing arrays.
    '''
    video = np.asarray(L)

    # center crop video to have consistent video frame number

    video = frame_crop_center(video, 42)

    fights.append(video)

for filename in os.listdir('/kaggle/working/fight-detection-surv-dataset/fight'):
    f = os.path.join('/kaggle/working/fight-detection-surv-dataset/fight', filename)
    # checking if it is a file
    video = skvideo.io.vread(f)
    video_dims.append(video.shape)

    L = []
    for i in range(video.shape[0]):
        frame = cv2.resize(video[i], (128, 128), interpolation=cv2.INTER_CUBIC)
        L.append(frame)

    video = np.asarray(L)
    video = frame_crop_center(video, 42)

    surv_fights.append(video)
