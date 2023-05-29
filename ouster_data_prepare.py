import csv
import numpy as np
import math
import fileinput
import os, shutil, sys
import pandas as pd
import glob, pickle
import re
import open3d as o3d
from pathlib import Path
import random as rm
import math as mh
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
grid_size = 0.06


def crop():
    labels = '/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/test_data/g_test/sunset_park3/3/'
    labels = glob.glob(labels + '*.csv')
    labels = np.array(labels)
    store = '/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/test_data/g_test/sunset_park3/3/Crop/'
    k1 = 350 * 128
    k2 = 750 * 128
    for label in labels:
        f = store + label[-24:]
        with open(label, 'r') as csvfile:
            label = csv.reader(csvfile)
            label = list(label)
            label = np.array(label)
            label = label[k1:k2, :]
        with open(f, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerows(label)


def split():
    files = glob.glob('/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/RandLA-Net-master/data/Ouster_crop/points/*label.csv')
    # print(files)
    f70out = '/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/RandLA-Net-master/data/Ouster_crop/Train'
    f30out = '/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/RandLA-Net-master/data/Ouster_crop/Validation'
    for file in files:
        r = rm.random()
        if r <= 0.9:
            print(file[84:] + ' in train')
            shutil.copyfile(file, f70out + '/' + file[84:])
        else:
            print(file[84:] + ' in test')
            shutil.copyfile(file, f30out + '/' + file[84:])


def rotation(train_file, vali_file):
    # path = '10_rotation'
    # subdir = os.path.join(root, path)
    # files = glob.glob('*label.csv')
    for file in train_file:
        f = open(file, 'r')
        csvreader = csv.reader(f)
        csvfile = list(csvreader)
        csvfile = np.array(csvfile, dtype=float)
        points = csvfile[:, 0:3]
        points = np.transpose(points)
        length = len(points[1])
        file = str(file)
        for i in range(0, 35):  # Generate 100 rotated files
            roll = (i + 1) * mh.pi / 18
            pitch = (i + 1) * mh.pi / 18
            yaw = (i + 1) * mh.pi / 18
            Zalpha = np.array([[mh.cos(yaw), mh.sin(yaw), 0], [-mh.sin(yaw), mh.cos(yaw), 0], [0, 0, 1]])

            Ybeta = np.array([[mh.cos(pitch), 0, -mh.sin(pitch)], [0, 1, 0], [mh.sin(pitch), 0, mh.cos(pitch)]])

            Xgamma = np.array([[1, 0, 0], [0, mh.cos(roll), mh.sin(roll)], [0, -mh.sin(roll), mh.cos(roll)]])

            rotated = []
            for j in range(length):
                rotate = np.dot(Zalpha, points[:, j])
                rotate = np.dot(Ybeta, rotate)
                rotate = np.dot(Xgamma, rotate)
                rotated.append(rotate)
            rotated = np.asarray(rotated)
            df = pd.DataFrame(rotated, index=None, columns=None)
            df1 = pd.DataFrame(csvfile[:, 3], index=None, columns=None)
            df = pd.concat([df, df1], axis=1)
            filename = file[0:-4] + str((i + 1) * 10) + file[-4::]
            df.to_csv(filename, index=False, columns=None, header=False)

    for file in vali_file:
        f = open(file, 'r')
        csvreader = csv.reader(f)
        csvfile = list(csvreader)
        csvfile = np.array(csvfile, dtype=float)
        points = csvfile[:, 0:3]
        length = len(points)
        file = str(file)
        for i in range(0, 35):  # Generate 100 rotated files
            angle = (i + 1) * mh.pi / 18
            matrix = [[mh.cos(angle), 0, mh.sin(angle)], [0, 1, 0], [-mh.sin(angle), 0, mh.cos(angle)]]
            rotated = []
            for j in range(length):
                rotate = np.dot(points[j, :], matrix)
                rotated.append(rotate)
            rotated = np.asarray(rotated)
            df = pd.DataFrame(rotated, index=None, columns=None)
            df1 = pd.DataFrame(csvfile[:, 3], index=None, columns=None)
            df = pd.concat([df, df1], axis=1)
            filename = file[0:-4] + str((i + 1) * 10) + file[-4::]
            df.to_csv(filename, index=False, columns=None, header=False)


def padandsave(train_file, vali_file):
    point_dir = root + '/data/Ouster_crop/points'
    label_dir = root + '/data/Ouster_crop/labels'
    for file in train_file:
        f = open(file)
        f = csv.reader(f)
        f = list(f)
        f = np.asarray(f)
        f = f.astype(np.float)
        # f = np.load(file)
        length = len(f)
        # f = np.delete(f, np.where(~f.any(axis=1))[0], axis=0)
        # f = np.pad(f, ((0, maximum - length), (0, 0)), 'constant')
        points = f[:, 0:3]
        labels = f[:, 3]
        # labels = labels.astype('float32')
        labels = labels.astype('int32')
        labels = np.expand_dims(labels, axis=1)
        labels = labels + 1
        points = points.astype('float32')
        file = file.replace('.csv', '.npy')
        point_name = point_dir + '/' + file[83:]
        label_name = label_dir + '/' + file[83:]
        np.save(point_name, points)
        # np.savetxt(file, f, delimiter=',')
        np.save(label_name, labels)
        print(file + ' done')

    for file in vali_file:
        f = open(file)
        f = csv.reader(f)
        f = list(f)
        f = np.asarray(f)
        f = f.astype(np.float)
        # f = np.load(file)
        length = len(f)
        # f = np.delete(f, np.where(~f.any(axis=1))[0], axis=0)
        # f = np.pad(f, ((0, maximum - length), (0, 0)), 'constant')
        points = f[:, 0:3]
        labels = f[:, 3]
        # labels = labels.astype('float32')
        labels = labels.astype('int32')
        labels = np.expand_dims(labels, axis=1)
        labels = labels + 1
        points = points.astype('float32')
        file = file.replace('.csv', '.npy')
        point_name = point_dir + '/' + file[88:]
        label_name = label_dir + '/' + file[88:]
        np.save(point_name, points)
        # np.savetxt(file, f, delimiter=',')
        np.save(label_name, labels)
        print(file + ' done')


# May need to change path names
def generate_proj_KDTree():
    dataset_path = ROOT_DIR + '/data/Ouster_crop'
    seq_list = np.sort(os.listdir(dataset_path))  # Cropped Ouster Data

    pc_path = join(dataset_path, 'points')
    KDTree_path_out = join(dataset_path, 'KDTree')
    os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None
    label_path = join(dataset_path, 'labels')
    scan_list = np.sort(glob.glob(pc_path + '/*.npy'))

    for scan_id in scan_list:
        scan_id = scan_id[84:]
        print(scan_id)
        sub_points = np.load(join(pc_path, scan_id))
        sub_labels = np.load(join(label_path, scan_id))
        search_tree = KDTree(sub_points)
        KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
        np.save(join(pc_path, scan_id)[:-4], sub_points)
        np.save(join(label_path, scan_id)[:-4], sub_labels)
        with open(KDTree_save, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_path = join(dataset_path, 'proj')
        os.makedirs(proj_path) if not exists(proj_path) else None
        proj_inds = np.squeeze(search_tree.query(sub_points, return_distance=False))
        proj_inds = proj_inds.astype(np.int32)
        proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_inds], f)


if __name__ == "__main__":
    crop()  # Crop data if needed
    split()  # Split data into 9:1 train:test
    root = os.getcwd()
    train_files = glob.glob(root + '/data/Ouster_crop/Train/*label*.csv')
    validation_files = glob.glob(root + '/data/Ouster_crop/Validation/*label*.csv')
    padandsave(train_files, validation_files)  # Generate corresponding train and test .npy files into folders
    generate_proj_KDTree()  # Generate .proj and .pkl files for training
