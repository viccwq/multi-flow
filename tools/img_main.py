import os,sys
import numpy as np
from PIL import Image
import cv2 as cv2
import argparse
from utils.dirs import *
from tools.img_proc.img_trainsform import *

def get_args():
    argparser = argparse.ArgumentParser(description="patch file processing")
    argparser.add_argument(
        '-s', '--src-dir',
        default='None',
        help='The directory of the source')
    argparser.add_argument(
        '-d', '--des-dir',
        default='None',
        help='The directory of the destination')
    argparser.add_argument(
        '-c', '--config',
        required=False,
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-a', '--action',
        required=False,
        default='None',
        help='The action of image processing')
    argparser.add_argument(
        '-r', '--resize',
        required=False,
        default=False,
        action='store_true',
        help='shall resize the image')
    args = argparser.parse_args()
    return vars(args)

if __name__ == '__main__':
    try:
        args = get_args()
    except:
        print("missing or invalid arguments")
        exit(0)

    src_dir = args["src_dir"]
    des_dir = args["des_dir"]
    #create the destination dir
    create_dirs([des_dir])

    if ("None" == args["action"]):
        pass
    # resize the image
    #-s g:\loc\szh\Isilon2\Video\01_Incoming_Bin\00_PF\ARCHIVE_temp\Small_Objects\10kphwithhighbeam -d g:\loc\szh\Isilon2\Video\01_Incoming_Bin\00_PF\ARCHIVE_temp\Small_Objects\10kphwithhighbeam_edit -a resize
    elif (args['action'] in "resize"):
        image_resize(src_dir, des_dir, src_fix=".bmp", des_fix=".jpg", width=1024)
    # duplicate the label file
    #-s g:\loc\szh\Isilon2\Video\01_Incoming_Bin\00_PF\ARCHIVE_temp\Small_Objects\10kphwithhighbeam_edit -d g:\loc\szh\Isilon2\Video\01_Incoming_Bin\00_PF\ARCHIVE_temp\Small_Objects\10kphwithhighbeam_out -a dupLab
    #-s g:\loc\szh\Isilon2\Video\01_Incoming_Bin\00_PF\ARCHIVE_temp\Small_Objects\30kphwithhighbeam_edit -d g:\loc\szh\Isilon2\Video\01_Incoming_Bin\00_PF\ARCHIVE_temp\Small_Objects\30kphwithhighbeam_out -a dupLab
    elif (args['action'] in "dupLab"):
        src_file = os.path.join(src_dir, "example.json")
        label_duplicate(src_dir, des_dir, src_file, src_fix=".jpg", des_fix=".json")
    else:
        pass