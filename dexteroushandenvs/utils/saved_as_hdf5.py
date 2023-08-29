import cv2
import numpy as np# Load the normalized depth image
import torch
import open3d as o3d
import scipy.io as scio
import pickle

env_id = 4
steps_id = 3
import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np
import json
from PIL import Image
import scipy.io as scio

def gather_demonstrations_as_hdf5(out_dir):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        demo1 (group) - every demonstration has a group
            box_label (dataset) - 2d bbox of the lego
            color_image (dataset) - rgb image
            depth_image (dataset) - depth image
            label_image (dataset) - label image
            intrinsic_matrix (dataset) -camera intrinsic.
            poses (dataset) - 6d poses.
            center (dataset) - center point.

        demo2 (group)
        ...
    Args:
        out_dir (str): Path to where to store the hdf5 file. 
    """

    hdf5_path = os.path.join(out_dir, "lego_pose_datasets.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0

    label_dir = "./_output_headless/data/"

    total_dir = os.listdir(label_dir)
    max_envs = 0
    max_total_steps = 0
    for files in total_dir:
        num_steps = int(files.split("-")[0])
        num_env = int(files.split("-")[1])

        if num_steps >= max_total_steps:
            max_total_steps = num_steps
        if num_env >= max_envs:
            max_envs = num_env

    print("max_total_steps: ", max_total_steps)
    print("max_envs: ", max_envs)

    start_time = time.time()

    # max_envs = 4
    # max_total_steps = 6

    for j in range(max_envs):
        for i in range(max_total_steps):
            print("processing {}th envs {}th steps".format(j, i))
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))
            
            # exit()
            color_image = cv2.imread(label_dir + '{}-{}-color.jpg'.format(i, j))
            f = open(label_dir + '{}-{}-depth.pkl'.format(i, j), "rb")
            depth = pickle.load(f)
            depth_image = depth
            label_image = cv2.imread(label_dir + '{}-{}-label.jpg'.format(i, j))
            meta_label = scio.loadmat(label_dir + "{}-{}-meta.mat".format(i, j))

            # write datasets for states and actions
            ep_data_grp.create_dataset("color_image", data=color_image)
            ep_data_grp.create_dataset("depth_image", data=depth_image)
            ep_data_grp.create_dataset("label_image", data=label_image)
            ep_data_grp.create_dataset("poses", data=meta_label["poses"])
            ep_data_grp.create_dataset("ext_poses", data=meta_label["ext_poses"])

            num_eps += 1
    stop_time = time.time()
    print("process_runtime: {}".format(stop_time - start_time))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)

    f.close()

def read_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        print(f["data"]["demo_0"]["color_image"][:])
        
if __name__ == "__main__":
    gather_demonstrations_as_hdf5("./_output_headless/")
    # read_hdf5("/home/jmji/Orbit/_output_headless/lego_pose_datasets.hdf5")