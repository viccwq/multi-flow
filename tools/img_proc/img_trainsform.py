import os,sys
import numpy as np
from PIL import Image
import cv2 as cv2
import shutil
from utils.dirs import *

def resize(rgb_img, width=0):
    h = rgb_img.shape[0]
    w = rgb_img.shape[1]
    if width > 0:
        ratio = width * 1.0 / w;
        h = int(h * ratio)
        w = width
    img = cv2.resize(rgb_img, (w, h))
    return img

def image_resize(src_dir, des_dir,  src_fix, des_fix, width):
    """

    :param src_dir:
    :param des_dir:
    :param width:
    :return:
    """
    file_list = get_files(path=src_dir, suffix=src_fix)
    for (i, file) in enumerate(file_list):
        file_des = os.path.join(des_dir, os.path.basename(file))
        file_des = file_des.replace(src_fix, des_fix)
        img = cv2.imread(file)
        img = resize(img, width=width)
        cv2.imwrite(file_des, img)
        print("save image:{}".format(file_des))

def label_duplicate(src_dir, des_dir, src_file, src_fix, des_fix):
    """
    :param src_dir:
    :param des_dir:
    :param src_file:
    :return:
    """
    file_list = get_files(path=src_dir, suffix=src_fix)
    for (i, file) in enumerate(file_list):
        file_des = os.path.join(des_dir, os.path.basename(file))
        file_des = file_des.replace(src_fix, des_fix)
        try:
            shutil.copyfile(src_file, file_des)
            data = "None"
            with open(src_file, "r") as f:
                data = f.read()
                data = data.replace("toBeReplaced.jpg", os.path.basename(file))
            with open(file_des, "w") as f:
                f.write(data)

            print("Duplicate file:{}".format(file_des))
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)
