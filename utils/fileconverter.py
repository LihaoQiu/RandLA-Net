#!/usr/bin/env python
import csv
import numpy as np
import math
import fileinput
import os
import pandas as pd
import glob
import re
import open3d as o3d
from pathlib import Path


# this file is to convert data files between
# 1) raw cloud point data .csv and .pcd for annotation
# 2) cloud point annotation .json and .csv with labeled cloud point data
# 3) load labeled cloud point data to training network

class CloudPoint:
    def __init__(self, cord):
        self.x = float(cord[0])
        self.y = float(cord[1])
        self.z = float(cord[2])


class Cuboid:
    def __init__(self, cuboidjson):
        self.px = float(cuboidjson[0])
        self.py = float(cuboidjson[1])
        self.pz = float(cuboidjson[2])
        self.rx = float(cuboidjson[3])
        self.ry = float(cuboidjson[4])
        self.rz = float(cuboidjson[5])
        self.dx = float(cuboidjson[6])
        self.dy = float(cuboidjson[7])
        self.dz = float(cuboidjson[8])


def DecimalToBinary(num):
    if num >= 1:
        DecimalToBinary(num // 2)
    print(num % 2, end='')
    return num % 2


def binaryOfFraction(fraction):
    # Declaring an empty string
    # to store binary bits.
    binary = str()

    # Iterating through
    # fraction until it
    # becomes Zero.
    while (fraction):

        # Multiplying fraction by 2.
        fraction *= 2

        # Storing Integer Part of
        # Fraction in int_part.
        if (fraction >= 1):
            int_part = 1
            fraction -= 1
        else:
            int_part = 0

        # Adding int_part to binary
        # after every iteration.
        binary += str(int_part)

    # Returning the binary string.
    return binary


# Function to get sign  bit,
# exp bits and mantissa bits,
# from given real no.
def floatingPoint(real_no):
    # Setting Sign bit
    # default to zero.
    sign_bit = 0

    # Sign bit will set to
    # 1 for negative no.
    if (real_no < 0):
        sign_bit = 1

    # converting given no. to
    # absolute value as we have
    # already set the sign bit.
    real_no = abs(real_no)

    # Converting Integer Part
    # of Real no to Binary
    int_str = bin(int(real_no))[2:]

    # Function call to convert
    # Fraction part of real no
    # to Binary.
    fraction_str = binaryOfFraction(real_no - int(real_no))

    # Getting the index where
    # Bit was high for the first
    # Time in binary repres
    # of Integer part of real no.
    ind = int_str.index('1')

    # The Exponent is the no.
    # By which we have right
    # Shifted the decimal and
    # it is given below.
    # Also converting it to bias
    # exp by adding 127.
    exp_str = bin((len(int_str) - ind - 1) + 127)[2:]

    # getting mantissa string
    # By adding int_str and fraction_str.
    # the zeroes in MSB of int_str
    # have no significance so they
    # are ignored by slicing.
    mant_str = int_str[ind + 1:] + fraction_str

    # Adding Zeroes in LSB of
    # mantissa string so as to make
    # it's length of 23 bits.
    mant_str = mant_str + ('0' * (23 - len(mant_str)))

    # Returning the sign, Exp
    # and Mantissa Bit strings.
    return sign_bit, exp_str, mant_str


def convertToInt(mantissa_str):
    # variable to make a count
    # of negative power of 2.
    power_count = -1

    # variable to store
    # float value of mantissa.
    mantissa_int = 0

    # Iterations through binary
    # Number. Standard form of
    # Mantissa is 1.M so we have
    # 0.M therefore we are taking
    # negative powers on 2 for
    # conversion.
    for i in mantissa_str:
        # Adding converted value of
        # Binary bits in every
        # iteration to float mantissa.
        mantissa_int += (int(i) * pow(2, power_count))

        # count will decrease by 1
        # as we move toward right.
        power_count -= 1

    # returning mantissa in 1.M form.
    return (mantissa_int + 1)

# read json text file and extract geometry info into csv
def json2csv(jsonfile):
    # split single line of json to multiple lines
    # each line with a .pcd file annotation
    fjson = open(jsonfile, 'r').read()
    fjson1 = fjson.replace("}}}]}", "\n")
    fjson2 = fjson1.replace(":", ",")
    fjson3 = fjson2.replace("\"", "")
    fjson4 = fjson3.replace("[", "")
    fjson5 = fjson4.replace("]", "")
    fjson6 = fjson5.replace("{", "")
    fjson7 = fjson6.replace("}", "")
    fjson8 = fjson7.replace(",id", "id")

    f = open('test.json', 'w')
    f.write(fjson8)
    f.close()

    # read each line of seperate .pcd annotation
    # get value for each parameter and make a list
    f1 = open('test.json', 'r')
    lines = f1.readlines()

    for line in lines:
        list_line = line.split(",")
        len_list = len(list_line)
        id = list_line[1]
        filename = list_line[3]

        csvfile = filename.replace('.pcd', '_json.csv')
        csvfile = 'pcd_json_csv\\' + csvfile
        # write json data to csv
        fcsv = open(csvfile, 'w', newline='')
        csvwriter = csv.writer(fcsv)

        # title/initial line
        fields = ['id', 'filename', 'classname', 'classid', 'px', 'py', 'pz', 'rx', 'ry', 'rz', 'dx', 'dy', 'dz']
        csvwriter.writerow(fields)

        num_obj = (len_list - 5) // 28
        for i in range(num_obj):
            classname = list_line[5 + 28 * i + 1]
            classid = list_line[5 + 28 * i + 5]
            px = list_line[5 + 28 * i + 9]
            py = list_line[5 + 28 * i + 11]
            pz = list_line[5 + 28 * i + 13]
            rx = list_line[5 + 28 * i + 16]
            ry = list_line[5 + 28 * i + 18]
            rz = list_line[5 + 28 * i + 20]
            dx = list_line[5 + 28 * i + 23]
            dy = list_line[5 + 28 * i + 25]
            dz = list_line[5 + 28 * i + 27]
            csvrow = [id, filename, classname, classid, px, py, pz, rx, ry, rz, dx, dy, dz]
            csvwriter.writerow(csvrow)

        fcsv.close()

    f1.close()


def json22csv(list_of_json):
    # split single line of json to multiple lines
    # each line with a .pcd file annotation

    n = len(list_of_json)
    for jsonfile in list_of_json:
        fjson = open(jsonfile, 'r').read()
        fjson = ''.join(fjson)
        fjson1 = fjson.split('figures', 1)
        fjson1 = fjson1[1]
        fjson2 = fjson1.replace(":", ",")
        fjson2 = fjson2.replace("\"", "")
        fjson2 = fjson2.replace("[", "")
        fjson2 = fjson2.replace("]", "")
        fjson2 = fjson2.replace("{", "")
        fjson2 = fjson2.replace("}", "")
        fjson2 = fjson2.replace("\n", "")
        fjson2 = ''.join(fjson2.split())
        count = fjson2.count('zhum2')
        fjson2 = fjson2.split(",")
        fjson2.remove('')

        #
        # f = open('test.json', 'w')
        # f.write(fjson8)
        # f.close()
        #
        # # read each line of seperate .pcd annotation
        # # get value for each parameter and make a list
        # f1 = open('test.json', 'r')
        # lines = f1.readlines()
        #
        # for line in lines:
        #     list_line = line.split(",")
        #     len_list = len(list_line)
        #     id = list_line[1]
        #     filename = list_line[3]
        #
        csvfile = jsonfile.replace('.json', '.csv')
        # csvfile = 'pcd_json_csv\\' + csvfile
        # write json data to csv
        with open(csvfile, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            # title/initial line
            fields = ['id', 'filename', 'classname', 'classid', 'px', 'py', 'pz', 'rx', 'ry', 'rz', 'dx', 'dy', 'dz']
            csvwriter.writerow(fields)

            num_obj = count
            for i in range(num_obj):
                classname = 'rail'
                classid = 'rail'
                px = fjson2[i * 38 + 9]
                py = fjson2[i * 38 + 11]
                pz = fjson2[i * 38 + 13]
                rx = fjson2[i * 38 + 16]
                ry = fjson2[i * 38 + 18]
                rz = fjson2[i * 38 + 20]
                dx = fjson2[i * 38 + 23]
                dy = fjson2[i * 38 + 25]
                dz = fjson2[i * 38 + 27]
                csvrow = [id, jsonfile, classname, classid, px, py, pz, rx, ry, rz, dx, dy, dz]
                csvwriter.writerow(csvrow)
        csvfile.close()
        print("json2csv done")

    # f1.close()


# assuming cuboid only rotate around x-direction,
# i.e. vertical/perpenticular to ground
def ptincuboid(pt, cuboid):
    px = cuboid.px
    py = cuboid.py
    pz = cuboid.pz
    rx = cuboid.rx
    ry = cuboid.ry
    rz = cuboid.rz    # ry and rz may reverse
    dx = cuboid.dx
    dy = cuboid.dy
    dz = cuboid.dz

    # Front bottom left point
    x1 = 0.5 * dx
    y1 = - 0.5 * dy
    z1 = - 0.5 * dz
    # Front bottom right point
    x2 = 0.5 * dx
    y2 = 0.5 * dy
    z2 = - 0.5 * dz
    # Front top left point
    x3 = 0.5 * dx
    y3 = - 0.5 * dy
    z3 = 0.5 * dz
    # Back bottom left point
    x4 = - 0.5 * dx
    y4 = - 0.5 * dy
    z4 = - 0.5 * dz

    P1 = [[x1], [y1], [z1]]  # Point 1
    P2 = [[x4], [y4], [z4]]  # Point 2
    P4 = [[x3], [y3], [z3]]  # Point 4
    P5 = [[x2], [y2], [z2]]  # Point 5
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    P4 = np.asarray(P4)
    P5 = np.asarray(P5)

    # # Camera Coordinate
    # P1 = P1 - np.asarray([[px], [py], [pz]])
    # P2 = P2 - np.asarray([[px], [py], [pz]])
    # P4 = P4 - np.asarray([[px], [py], [pz]])
    # P5 = P5 - np.asarray([[px], [py], [pz]])


    # Roll
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])

    # Pitch
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])

    # Yaw
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])

    D = Rx
    Rx = Rz
    Rz = D
    # P1 = Rz.dot(Ry).dot(Rx).dot(P1)
    # P2 = Rz.dot(Ry).dot(Rx).dot(P2)
    # P4 = Rz.dot(Ry).dot(Rx).dot(P4)
    # P5 = Rz.dot(Ry).dot(Rx).dot(P5)

    P1 = Rz.dot(Ry).dot(Rx).dot(P1) + np.array([[px], [py], [pz]])
    P2 = Rz.dot(Ry).dot(Rx).dot(P2) + np.array([[px], [py], [pz]])
    P4 = Rz.dot(Ry).dot(Rx).dot(P4) + np.array([[px], [py], [pz]])
    P5 = Rz.dot(Ry).dot(Rx).dot(P5) + np.array([[px], [py], [pz]])

    # Rotation Matrix
    # A = [[math.cos(rz) * math.cos(ry), -math.sin(rz) * math.cos(rx) + math.cos(rz) * math.sin(ry) * math.sin(rx), \
    #       math.sin(rz) * math.sin(rx) + math.cos(rz) * math.sin(ry) * math.cos(rx)] \
    #     , [math.sin(rz) * math.cos(ry), math.cos(rz) * math.cos(rx) + math.sin(rz) * math.sin(ry) * math.sin(rx), \
    #        math.sin(rz) * math.sin(ry) * math.cos(rx) - math.cos(rz) * math.sin(rx)] \
    #     , [-math.sin(ry), math.cos(ry) * math.sin(rx), math.cos(ry) * math.cos(rx)]]
    # A = np.asarray(A)
    # A = np.linalg.inv(A)
    # # Apply Rotation
    # P1 = np.dot(A, P1)
    # P2 = np.dot(A, P2)
    # P4 = np.dot(A, P4)
    # P5 = np.dot(A, P5)
    #
    # P1 = P1 + np.asarray([px, py, pz])
    # P2 = P2 + np.asarray([px, py, pz])
    # P4 = P4 + np.asarray([px, py, pz])
    # P5 = P5 + np.asarray([px, py, pz])

    # Construct vector
    # Front bottom left - Back bottom left
    u = [P1[0, 0] - P2[0, 0], P1[1, 0] - P2[1, 0], P1[2, 0] - P2[2, 0]]
    u = np.asarray(u)
    # Front bottom left - front bottom right
    v = [P1[0, 0] - P4[0, 0], P1[1, 0] - P4[1, 0], P1[2, 0] - P4[2, 0]]
    v = np.asarray(v)
    # Front bottom left - front top left
    w = [P1[0, 0] - P5[0, 0], P1[1, 0] - P5[1, 0], P1[2, 0] - P5[2, 0]]
    w = np.asarray(w)
    eta = [[pt.x], [pt.y], [pt.z]]
    eta = np.asarray(eta)

    a = np.dot(u, P1)
    b = np.dot(u, eta)
    c = np.dot(u, P2)
    d = np.dot(v, P1)
    e = np.dot(v, eta)
    f = np.dot(v, P4)
    g = np.dot(w, P1)
    h = np.dot(w, eta)
    i = np.dot(w, P5)

    if min(a, c) < b < max(a, c) and min(d, f) < e < max(d, f) and min(g, i) < h < max(g, i):
        return True
    else:
        return False


# read cloud point and cuboid info from csv file, respectively
# label pt according to cuboid info
def labelpt(cpcsv, jsoncsv, labelcsv):
    f1 = open(cpcsv, 'r')
    f2 = open(jsoncsv, 'r')
    f3 = open(labelcsv, 'w', newline='')
    csvwriter = csv.writer(f3)
    cpreader = csv.reader(f1)
    cpdata = list(cpreader)
    cpdata = np.array(cpdata)
    cpdata = cpdata[1:, 0:3]
    n_cp = cpdata.shape[0]
    s_cp = []
    for n in range(n_cp):
        s_cp.append(CloudPoint(cpdata[n]))

    cuboidreader = csv.reader(f2)
    cuboiddata = list(cuboidreader)
    cuboiddata = np.array(cuboiddata)

    n_cuboid_rail = cuboiddata.shape[0] - 1
    # n_cuboid_train = cuboiddata.shape[0] - 1 - b             #Rail number?
    s_cuboid_rail = []
    # s_cuboid_train = []
    for n in range(n_cuboid_rail):
        s_cuboid_rail.append(Cuboid(cuboiddata[n + 1][4:]))
    # for n in range(n_cuboid_train):
    #     s_cuboid_train.append(Cuboid(cuboiddata[n+b+1][4:]))
    for i in range(n_cp):
        label = 0
        for j in range(n_cuboid_rail):
            if ptincuboid(s_cp[i], s_cuboid_rail[j]):
                label = 1

        cp_label = []
        for j in range(cpdata.shape[1]):
            cp_label.append(cpdata[i][j])
        cp_label.append(label)
        csvwriter.writerow(cp_label)

    f1.close()
    f2.close()
    f3.close()
    print("done")


def load_csv_data_label_seg(csv_filename):
    csvfilename = csv_filename.replace('.csv', '')

    f = open(csv_filename, errors='ignore')
    reader = csv.reader(f)
    table = list(reader)
    table = np.array(table)
    # table = np.delete(table, 0, 0)

    data = []
    label = []
    seg = []
    for i in range(11):
        data_row = []
        label.append([0])
        seg_row = []

        for j in range(2048):
            row = 400 + i * 2048 + j
            data_row.append([float(table[row, 0]), float(table[row, 1]), float(table[row, 2])])
            seg_row.append(int(table[400 + i * 2048 + j, 3]))

        data.append(data_row)
        seg.append(seg_row)

    data = np.array(data)
    label = np.array(label)
    seg = np.array(seg)
    f.close()

    return (csvfilename, data, label, seg)


def save_csv_data_label_seg(TBD):
    # cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_test_filename)
    filename, cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_test_filename)

    f = open(filename + '_pred.csv', 'w', newline='')
    csvwriter = csv.writer(f)

    for j in range(num_batch):
        n_rows = cur_data.shape[1]
        for k in range(n_rows):
            datarow = []
            datarow.append(cur_data[j][k][0])
            datarow.append(cur_data[j][k][1])
            datarow.append(cur_data[j][k][2])
            datarow.append(pred_seg_res[0][k])
            csvwriter.writerow(datarow)


def csv2pcd(list_filename):
    list_csv = [line.rstrip() for line in open(list_filename)]
    num_csv_file = len(list_csv)
    for csvfile in list_csv:
        print('Opening ' + str(csvfile))
        f1 = open(csvfile, errors='ignore')
        reader1 = csv.reader(f1)
        table1 = list(reader1)
        table1 = np.array(table1)

        filename = csvfile.replace('.csv', '')
        pcdname = filename + '.pcd'
        f2 = open(pcdname, 'w', newline='')
        f2.write('# .PCD v0.7 - Point Cloud Data file format\n')
        f2.write('VERSION 0.7\n')
        f2.write('FIELDS x y z\n')
        f2.write('SIZE 4 4 4\n')
        f2.write('TYPE F F F\n')
        f2.write('COUNT 1 1 1\n')
        f2.write('WIDTH ' + str(table1.shape[0]) + '\n')
        f2.write('HEIGHT 1\n')
        f2.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f2.write('POINTS ' + str(table1.shape[0]) + '\n')
        f2.write('DATA ascii\n')

        for i in range(table1.shape[0]):
            strdata = str(table1[i][0])
            for j in range(2):
                strdata = strdata + ' ' + str(table1[i][j + 1])
            strdata = strdata + '\n'
            f2.write(strdata)

        f1.close()
        f2.close()


def pcd2csv(list_of_pcd):
    pts = []
    # list_pcd = [line.rstrip() for line in open(list_of_pcd)]
    for filename in list_of_pcd:
        f = open(filename, 'r')
        # print(str(filename))
        filename = str(filename)
        filename = filename.replace("pcd", "csv")
        print(filename)
        data = f.readlines()
        f.close()
        line = data[9]
        # print line
        line = line.strip('\n')
        i = line.split(' ')
        pts_num = eval(i[-1])
        with open(filename, 'a') as csvfile:
            for line in data[11:]:
                line = line.strip('\n')
                xyzargb = line.split(' ')
                x, y, z = [eval(i) for i in xyzargb[:3]]
                intensity = xyzargb[3]
                t = xyzargb[4]
                reflectivity = xyzargb[5]
                ring = xyzargb[6]
                noise = xyzargb[7]
                ran = xyzargb[8]
                # print type(bgra)
                # pts = ([x, y, z, intensity, t, reflectivity, ring, noise, ran])
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([x, y, z, intensity, t, reflectivity, ring, noise, ran])


def lvx2csv(list_lvx_file):
    list_lvx = [line.rstrip() for line in open(list_lvx_file)]
    num_lvx = len(list_lvx)
    o = 0
    for lvxfile in list_lvx:
        fname = lvxfile.replace('.lvx', '')
        f1 = open(lvxfile, 'rb')
        arr = list(f1.read())
        sz_arr = len(arr)

        # fcsv = open(fname+'.csv', 'w', newline='')
        # csvwriter = csv.writer(fcsv)

        public_private_header = arr[0:28]
        # # check if first line includes 'livox_tech' and magic word
        if public_private_header[0:10] != [0x6C, 0x69, 0x76, 0x6F, 0x78, 0x5F, 0x74, 0x65, 0x63, 0x68] or \
                public_private_header[16:24] != [0x01, 0x01, 0x00, 0x00, 0x67, 0xa7, 0x0e, 0xac] or \
                public_private_header[24:28] != [0x32, 0x00, 0x00, 0x00]:
            print('file corrupted')
        else:
            num_dev = arr[28]
            pt = 29
            ptcnt = 0
            dev_info = []
            lidar_sn = []
            hub_sn = []
            dev_idx = []
            dev_type = []
            extr_en = []
            rpy_xyz = []
            for i in range(num_dev):
                dev_info.append(arr[pt: pt + 59])
                lidar_sn.append(dev_info[i][0:16])
                hub_sn.append(dev_info[i][16:32])
                dev_idx.append(dev_info[i][32])
                dev_type.append(dev_info[i][33])
                extr_en.append(dev_info[i][34])
                rpy_xyz.append(dev_info[i][35:59])
                pt = pt + 59

            frame_header = []
            frame_cnt = 0
            # # starting (multi) frame data reading
            while pt < sz_arr:
                frame_header.append(arr[pt: pt + 24])
                cur_offset = 0
                for j in range(8):
                    cur_offset += frame_header[frame_cnt][j] * (256 ** j)
                next_offset = 0
                for j in range(8):
                    next_offset += frame_header[frame_cnt][j + 8] * (256 ** j)
                frame_idx = frame_header[frame_cnt][16:24]
                pt = pt + 24
                pkg_header = []
                pkg_cnt = 0

                # Create csv file
                fcsv = open(str(o) + '.csv', 'w', newline='')
                csvwriter = csv.writer(fcsv)
                fimu = open(str(o) + '_imu.csv', 'w', newline='')
                imuwriter = csv.writer(fimu)

                # # for each frame, read
                while pt < next_offset:
                    pkg_header.append(arr[pt: pt + 19])
                    pt = pt + 19
                    data_type = pkg_header[pkg_cnt][10]
                    if data_type == 0:
                        pt_cnt = 100
                        for k in range(pt_cnt):
                            if arr[pt + 3] >= 128:
                                data_pt1 = arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + (
                                        arr[pt + 3] - 256) * (256 ** 3)
                            else:
                                data_pt1 = arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + arr[pt + 3] * (
                                        256 ** 3)

                            if arr[pt + 7] >= 128:
                                data_pt2 = arr[pt + 4] + arr[pt + 5] * 256 + arr[pt + 6] * (256 ** 2) + (
                                        arr[pt + 7] - 256) * (256 ** 3)
                            else:
                                data_pt2 = arr[pt + 4] + arr[pt + 5] * 256 + arr[pt + 6] * (256 ** 2) + arr[pt + 7] * (
                                        256 ** 3)

                            if arr[pt + 11] >= 128:
                                data_pt3 = arr[pt + 8] + arr[pt + 9] * 256 + arr[pt + 10] * (256 ** 2) + (
                                        arr[pt + 11] - 256) * (256 ** 3)
                            else:
                                data_pt3 = arr[pt + 8] + arr[pt + 9] * 256 + arr[pt + 10] * (256 ** 2) + arr[
                                    pt + 11] * (256 ** 3)

                            data_pt = [data_pt1, data_pt2, data_pt3, arr[pt + 12]]
                            csvwriter.writerow(data_pt)
                            pt = pt + 13

                    elif data_type == 1:
                        pt_cnt = 100
                        for k in range(pt_cnt):
                            if arr[pt + 3] >= 128:
                                data_pt1 = arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + (
                                        arr[pt + 3] - 256) * (256 ** 3)
                            else:
                                data_pt1 = arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + arr[pt + 3] * (
                                        256 ** 3)

                            data_pt2 = arr[pt + 4] + arr[pt + 5] * 256
                            data_pt3 = arr[pt + 6] + arr[pt + 7] * 256
                            data_pt = [data_pt1, data_pt2, data_pt3, arr[pt + 8]]
                            csvwriter.writerow(data_pt)
                            pt = pt + 9

                    elif data_type == 2:
                        pt_cnt = 96
                        for k in range(pt_cnt):
                            if arr[pt + 3] >= 128:
                                data_pt1 = (arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + (
                                        arr[pt + 3] - 256) * (256 ** 3)) / 1000
                            else:
                                data_pt1 = (arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + arr[pt + 3] * (
                                        256 ** 3)) / 1000

                            if arr[pt + 7] >= 128:
                                data_pt2 = (arr[pt + 4] + arr[pt + 5] * 256 + arr[pt + 6] * (256 ** 2) + (
                                        arr[pt + 7] - 256) * (256 ** 3)) / 1000
                            else:
                                data_pt2 = (arr[pt + 4] + arr[pt + 5] * 256 + arr[pt + 6] * (256 ** 2) + arr[pt + 7] * (
                                        256 ** 3)) / 1000

                            if arr[pt + 11] >= 128:
                                data_pt3 = (arr[pt + 8] + arr[pt + 9] * 256 + arr[pt + 10] * (256 ** 2) + (
                                        arr[pt + 11] - 256) * (256 ** 3)) / 1000
                            else:
                                data_pt3 = (arr[pt + 8] + arr[pt + 9] * 256 + arr[pt + 10] * (256 ** 2) + arr[
                                    pt + 11] * (256 ** 3)) / 1000

                            data_pt = [data_pt1, data_pt2, data_pt3, arr[pt + 12], arr[pt + 13]]
                            csvwriter.writerow(data_pt)
                            pt = pt + 14

                    elif data_type == 3:
                        pt_cnt = 96
                        for k in range(pt_cnt):
                            if arr[pt + 3] >= 128:
                                data_pt1 = arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + (
                                        arr[pt + 3] - 256) * (256 ** 3)
                            else:
                                data_pt1 = arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + arr[pt + 3] * (
                                        256 ** 3)

                            data_pt2 = arr[pt + 4] + arr[pt + 5] * 256
                            data_pt3 = arr[pt + 6] + arr[pt + 7] * 256
                            data_pt = [data_pt1, data_pt2, data_pt3, arr[pt + 8], arr[pt + 9]]
                            csvwriter.writerow(data_pt)
                            pt = pt + 10

                    elif data_type == 4:
                        pt_cnt = 48
                        for k in range(pt_cnt):
                            if arr[pt + 3] >= 128:
                                data_pt1 = arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + (
                                        arr[pt + 3] - 256) * (256 ** 3)
                            else:
                                data_pt1 = arr[pt] + arr[pt + 1] * 256 + arr[pt + 2] * (256 ** 2) + arr[pt + 3] * (
                                        256 ** 3)

                            if arr[pt + 7] >= 128:
                                data_pt2 = arr[pt + 4] + arr[pt + 5] * 256 + arr[pt + 6] * (256 ** 2) + (
                                        arr[pt + 7] - 256) * (256 ** 3)
                            else:
                                data_pt2 = arr[pt + 4] + arr[pt + 5] * 256 + arr[pt + 6] * (256 ** 2) + arr[pt + 7] * (
                                        256 ** 3)

                            if arr[pt + 11] >= 128:
                                data_pt3 = arr[pt + 8] + arr[pt + 9] * 256 + arr[pt + 10] * (256 ** 2) + (
                                        arr[pt + 11] - 256) * (256 ** 3)
                            else:
                                data_pt3 = arr[pt + 8] + arr[pt + 9] * 256 + arr[pt + 10] * (256 ** 2) + arr[
                                    pt + 11] * (256 ** 3)

                            if arr[pt + 17] >= 128:
                                data_pt4 = arr[pt + 14] + arr[pt + 15] * 256 + arr[pt + 16] * (256 ** 2) + (
                                        arr[pt + 17] - 256) * (256 ** 3)
                            else:
                                data_pt4 = arr[pt + 14] + arr[pt + 15] * 256 + arr[pt + 16] * (256 ** 2) + arr[
                                    pt + 17] * (256 ** 3)

                            if arr[pt + 21] >= 128:
                                data_pt5 = arr[pt + 18] + arr[pt + 19] * 256 + arr[pt + 20] * (256 ** 2) + (
                                        arr[pt + 21] - 256) * (256 ** 3)
                            else:
                                data_pt5 = arr[pt + 18] + arr[pt + 19] * 256 + arr[pt + 20] * (256 ** 2) + arr[
                                    pt + 21] * (256 ** 3)

                            if arr[pt + 25] >= 128:
                                data_pt6 = arr[pt + 22] + arr[pt + 23] * 256 + arr[pt + 24] * (256 ** 2) + (
                                        arr[pt + 25] - 256) * (256 ** 3)
                            else:
                                data_pt6 = arr[pt + 22] + arr[pt + 23] * 256 + arr[pt + 24] * (256 ** 2) + arr[
                                    pt + 25] * (256 ** 3)

                            data_pt = [data_pt1, data_pt2, data_pt3, arr[pt + 12], arr[pt + 13], data_pt4, data_pt5,
                                       data_pt6, arr[pt + 26], arr[pt + 27]]
                            csvwriter.writerow(data_pt)
                            pt = pt + 28

                    elif data_type == 5:
                        pt_cnt = 48
                        for k in range(pt_cnt):
                            data_pt1 = arr[pt] + arr[pt + 1] * 256
                            data_pt2 = arr[pt + 2] + arr[pt + 3] * 256
                            if arr[pt + 7] >= 128:
                                data_pt3 = arr[pt + 4] + arr[pt + 5] * 256 + arr[pt + 6] * (256 ** 2) + (
                                        arr[pt + 7] - 256) * (256 ** 3)
                            else:
                                data_pt3 = arr[pt + 4] + arr[pt + 5] * 256 + arr[pt + 6] * (256 ** 2) + arr[pt + 7] * (
                                        256 ** 3)

                            if arr[pt + 13] >= 128:
                                data_pt4 = arr[pt + 10] + arr[pt + 11] * 256 + arr[pt + 12] * (256 ** 2) + (
                                        arr[pt + 13] - 256) * (256 ** 3)
                            else:
                                data_pt4 = arr[pt + 10] + arr[pt + 11] * 256 + arr[pt + 12] * (256 ** 2) + arr[
                                    pt + 13] * (256 ** 3)

                            data_pt = [data_pt1, data_pt2, data_pt3, arr[pt + 8], arr[pt + 9], data_pt4, arr[pt + 14],
                                       arr[pt + 15]]
                            csvwriter.writerow(data_pt)
                            pt = pt + 16

                    elif data_type == 6:
                        data_imu = []
                        for k in range(6):
                            tmp = 0
                            if arr[pt + 4 * k + 3] >= 128:
                                # tmp = np.binary_repr(arr[pt + 4 * k] + arr[pt + 4 * k + 1] * 256 + arr[pt + 4 * k + 2] * (256 ** 2) + (
                                #         arr[pt + 4 * k + 3] - 256) * (256 ** 3), width=8)
                                chunk1 = np.binary_repr(arr[pt + 4 * k], width=8) # First 8 bits
                                chunk2 = np.binary_repr(arr[pt + 4 * k + 1], width=8)
                                chunk3 = np.binary_repr(arr[pt + 4 * k + 2], width=8)
                                chunk4 = np.binary_repr(arr[pt + 4 * k + 3], width=8)
                                chunk_32 = chunk4 + chunk3 + chunk2 + chunk1
                                sign_bit = int(chunk_32[0])
                                exponent_bias = int(chunk_32[2: 10], 2)
                                exponent_unbias = exponent_bias - 127
                                mantissa_str = chunk_32[11:]
                                mantissa_int = convertToInt(mantissa_str)
                                real_no = pow(-1, sign_bit) * mantissa_int * pow(2, exponent_unbias)
                                # tmp = DecimalToBinary(tmp)
                            else:
                                # tmp = np.binary_repr(arr[pt + 4 * k] + arr[pt + 4 * k + 1] * 256 + arr[pt + 4 * k + 2] * (256 ** 2) + \
                                #       arr[pt + 4 * k + 3] * (256 ** 3), width=8)
                                chunk1 = np.binary_repr(arr[pt + 4 * k], width=8) # First 8 bits
                                chunk2 = np.binary_repr(arr[pt + 4 * k + 1], width=8)
                                chunk3 = np.binary_repr(arr[pt + 4 * k + 2], width=8)
                                chunk4 = np.binary_repr(arr[pt + 4 * k + 3], width=8)
                                chunk_32 = chunk4 + chunk3 + chunk2 + chunk1
                                sign_bit = int(chunk_32[0])
                                exponent_bias = int(chunk_32[2: 10], 2)
                                exponent_unbias = exponent_bias - 127
                                mantissa_str = chunk_32[11:]
                                mantissa_int = convertToInt(mantissa_str)
                                real_no = pow(-1, sign_bit) * mantissa_int * pow(2, exponent_unbias)
                                # tmp = DecimalToBinary(tmp)
                            data_imu.append(real_no)

                        imuwriter.writerow(data_imu)
                        pt = pt + 24

                    pkg_cnt += 1

                frame_cnt = frame_cnt + 1
                pt = next_offset
                o = o + 1
                fcsv.close()
                fimu.close()

        # fcsv.close()
        f1.close()


def bag2csv(list_bag_file):
    list_bag = [line.rstrip() for line in open(list_bag_file)]
    num_bag = len(list_bag)
    for bagfile in list_bag:
        fname = bagfile.replace('.bag', '')
        f1 = open(bagfile, 'rb')
        arr = list(f1.read())
        sz_arr = len(arr)

        fcsv = open(fname + '.csv', 'w', newline='')
        csvwriter = csv.writer(fcsv)

        fheader = arr[0:13]
        if fheader != [0x23, 0x52, 0x4F, 0x53, 0x42, 0x41, 0x47, 0x20, 0x56, 0x32, 0x2E, 0x30, 0x0A]:
            print('file corrupted, header wrong')
        else:
            pt = 13
            rec_cnt = 0
            header_start = []
            header_end = []
            data_start = []
            data_end = []
            while pt < sz_arr:
                # # header and data general info
                header_len = 0
                for i in range(4):
                    header_len += arr[pt + i] * (256 ** i)
                header_start.append(pt + 4)
                header_end.append(header_start[rec_cnt] + header_len)
                header = arr[header_start[rec_cnt]: header_end[rec_cnt]]

                data_len = 0
                for i in range(4):
                    data_len += arr[header_end[rec_cnt] + i] * (256 ** i)
                data_start.append(header_end[rec_cnt] + 4)
                data_end.append(data_start[rec_cnt] + data_len)
                data = arr[data_start[rec_cnt]: data_end[rec_cnt]]

                # # decode header
                field_cnt = 0
                pt = header_start[rec_cnt]
                field_start = []
                field_end = []
                field_name = []
                field_name_str = []
                field_data = []
                while pt < header_end[rec_cnt]:
                    field_len = 0
                    for i in range(4):
                        field_len += arr[pt + i] * (256 ** i)
                    pt += 4
                    field_start.append(pt)
                    field_end.append(pt + field_len)

                    while pt < field_end[field_cnt]:
                        if arr[pt] == 0x3d:
                            field_name.append(arr[field_start[field_cnt]: pt])
                            res = ""
                            for val in field_name[field_cnt]:
                                res += chr(val)
                            field_name_str.append(res)
                            field_data.append(arr[pt + 1: field_end[field_cnt]])
                            pt = field_end[field_cnt]
                            break
                        else:
                            pt += 1

                    field_cnt += 1

                pt = data_end[rec_cnt]

                rec_cnt += 1

        fcsv.close()
        f1.close()


# def replace(newfile, oldfile):
#     #Create temp file
#     with open newfile(fh,'r') as new_file:
#         with open(file_path) as old_file:
#             for line in old_file:
#                 new_file.write(line.replace(pattern, subst))

def Conto2frames(list_csv_file):
    k = 0
    m = 0
    li = []
    list_csv = [line.rstrip() for line in open(list_csv_file)]
    for file_name in list_csv:
        print('Opening ' + file_name)
        df = pd.read_csv(file_name, index_col=None, header=None, error_bad_lines=False)
        if k == 0:
            df1 = df
        if k != 0:
            df1 = pd.concat([df1, df], ignore_index=True)
        # li.append(df)
        k += 1
        if k == 2:
            print('Saving to ' + str(m) + ' Con.csv')
            # frame = pd.concat(li, axis=0, ignore_index=True)
            df1.to_csv('Con' + str(m) + '.csv', index=False, header=False)
            k = 0
            m += 1


def get_file_name():
    # def get_key(fp):
    # filename = os.path.splitext(os.path.basename(fp))[0]
    # int_part = filename.split()[0]
    # return int(int_part)
    # list_of_files = sorted(glob.glob('*.csv'), key=os.path.getmtime)
    # file_object = open("list_filename", 'w')
    # for file_name in list_of_files:
    #     file_object.writelines(file_name + '\n')

    list_of_files = sorted(glob.glob('Con*'), key=os.path.getmtime)
    file_object = open("list_filename", 'w')
    for file_name in list_of_files:
        file_object.writelines(file_name + '\n')

    # list_of_cons = glob.glob('Con*')
    # file_object = open("list_csv_con.txt", 'w')
    # for file_name in list_of_cons:
    #     file_object.writelines(file_name + '\n')


# def read_pcd(file_path):
#     pcd = o3d.io.read_point_cloud(file_path)
#     print(np.asarray(pcd.points))
#     colors = np.asarray(pcd.colors) * 255
# 	points = np.asarray(pcd.points)
# 	print(points.shape, colors.shape)
# 	return np.concatenate([points, colors], axis=-1)

def ryrz():
    files = glob.glob('*pcd.csv')
    for file in files:
        with open(file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            f = list(csvreader)
            f = np.array(f)
            f[1:, 8] = 0
            f[1:, 9] = 0
        csvfile.close()
        with open(file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(f)


def cuboidshape(cpcsv, jsoncsv, labelcsv):
    f1 = open(cpcsv, 'r')
    f2 = open(jsoncsv, 'r')
    f3 = open(labelcsv, 'a', newline='')
    csvwriter = csv.writer(f3)
    cpreader = csv.reader(f1)
    cpdata = list(cpreader)
    cpdata = np.array(cpdata)
    n_cp = cpdata.shape[0]
    label = np.zeros((n_cp, 1))
    cpdata = np.concatenate((cpdata, label), axis=1)
    cpdata = cpdata[:, :4]
    # for i in range(n_cp):
    #     cpdata[i, 3] = 1
    s_cp = []

    cuboidreader = csv.reader(f2)
    cuboiddata = list(cuboidreader)
    cuboiddata = np.array(cuboiddata)
    n_cuboid_rail = cuboiddata.shape[0] - 1
    csvwriter.writerows(cpdata)
    cupoint = []

    for i in range(1, n_cuboid_rail+1):
        rx = float(cuboiddata[i, 7])
        ry = float(cuboiddata[i, 9])
        rz = float(cuboiddata[i, 8])

        A = [[math.cos(rz) * math.cos(ry), -math.sin(rz) * math.cos(rx) + math.cos(rz) * math.sin(ry) * math.sin(rx), \
              math.sin(rz) * math.sin(rx) + math.cos(rz) * math.sin(ry) * math.cos(rx)] \
            , [math.sin(rz) * math.cos(ry), math.cos(rz) * math.cos(rx) + math.sin(rz) * math.sin(ry) * math.sin(rx), \
               math.sin(rz) * math.sin(ry) * math.cos(rx) - math.cos(rz) * math.sin(rx)] \
            , [-math.sin(ry), math.cos(ry) * math.sin(rx), math.cos(ry) * math.cos(rx)]]
        A = np.asarray(A)
        x1 = float(cuboiddata[i, 4]) + 0.5 * float(cuboiddata[i, 10])
        y1 = float(cuboiddata[i, 5]) - 0.5 * float(cuboiddata[i, 11])
        z1 = float(cuboiddata[i, 6]) - 0.5 * float(cuboiddata[i, 12])
        # [x1, y1, z1] = np.asarray([x1, y1, z1])
        [x1, y1, z1] = np.asarray([x1, y1, z1]) - np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 5]), \
                                                              float(cuboiddata[i, 6])])
        [x1, y1, z1] = np.dot([x1, y1, z1], A)
        # [x1, y1, z1] = np.asarray([x1, y1, z1])
        [x1, y1, z1] = np.asarray([x1, y1, z1]) + np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 5]), \
                                                              float(cuboiddata[i, 6])])
        cupoint.append([x1, y1, z1, 1])
        x2 = float(cuboiddata[i, 4]) + 0.5 * float(cuboiddata[i, 10])
        y2 = float(cuboiddata[i, 5]) + 0.5 * float(cuboiddata[i, 11])
        z2 = float(cuboiddata[i, 6]) - 0.5 * float(cuboiddata[i, 12])
        # [x2, y2, z2] = np.asarray([x2, y2, z2])
        [x2, y2, z2] = np.asarray([x2, y2, z2]) - np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 5]), \
                                                              float(cuboiddata[i, 6])])
        [x2, y2, z2] = np.dot([x2, y2, z2], A)
        # [x2, y2, z2] = np.asarray([x2, y2, z2])
        [x2, y2, z2] = np.asarray([x2, y2, z2]) + np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 5]), \
                                                              float(cuboiddata[i, 6])])
        cupoint.append([x2, y2, z2, 1])

        x3 = float(cuboiddata[i, 4]) + 0.5 * float(cuboiddata[i, 10])
        y3 = float(cuboiddata[i, 5]) - 0.5 * float(cuboiddata[i, 11])
        z3 = float(cuboiddata[i, 6]) + 0.5 * float(cuboiddata[i, 12])
        # [x3, y3, z3] = np.asarray([x3, y3, z3])
        [x3, y3, z3] = np.asarray([x3, y3, z3]) - np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 5]), \
                                                              float(cuboiddata[i, 6])])
        [x3, y3, z3] = np.dot([x3, y3, z3], A)
        # [x3, y3, z3] = np.asarray([x3, y3, z3])
        [x3, y3, z3] = np.asarray([x3, y3, z3]) + np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 5]), \
                                                              float(cuboiddata[i, 6])])
        cupoint.append([x3, y3, z3, 1])

        x4 = float(cuboiddata[i, 4]) - 0.5 * float(cuboiddata[i, 10])
        y4 = float(cuboiddata[i, 5]) - 0.5 * float(cuboiddata[i, 11])
        z4 = float(cuboiddata[i, 6]) - 0.5 * float(cuboiddata[i, 12])
        # [x4, y4, z4] = np.asarray([x4, y4, z4])
        [x4, y4, z4] = np.asarray([x4, y4, z4]) - np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 5]), \
                                                              float(cuboiddata[i, 6])])
        [x4, y4, z4] = np.dot([x4, y4, z4], A)
        # [x4, y4, z4] = np.asarray([x4, y4, z4])
        [x4, y4, z4] = np.asarray([x4, y4, z4]) + np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 5]), \
                                                              float(cuboiddata[i, 6])])
        cupoint.append([x4, y4, z4, 1])
        cupoint.append([cuboiddata[i, 4], cuboiddata[i, 5], cuboiddata[i, 6], 1])

    csvwriter.writerows(cupoint)
    print(labelcsv + ' done')

    #
    # for i in range(1, n_cuboid_rail+1):
    #     rx = float(cuboiddata[i, 7])
    #     ry = float(cuboiddata[i, 9])
    #     rz = float(cuboiddata[i, 8])
    #
    #     A = [[math.cos(rz) * math.cos(ry), -math.sin(rz) * math.cos(rx) + math.cos(rz) * math.sin(ry) * math.sin(rx), \
    #           math.sin(rz) * math.sin(rx) + math.cos(rz) * math.sin(ry) * math.cos(rx)] \
    #         , [math.sin(rz) * math.cos(ry), math.cos(rz) * math.cos(rx) + math.sin(rz) * math.sin(ry) * math.sin(rx), \
    #            math.sin(rz) * math.sin(ry) * math.cos(rx) - math.cos(rz) * math.sin(rx)] \
    #         , [-math.sin(ry), math.cos(ry) * math.sin(rx), math.cos(ry) * math.cos(rx)]]
    #     A = np.asarray(A)
    #     x1 = float(cuboiddata[i, 4]) + 0.5 * float(cuboiddata[i, 10])
    #     y1 = float(cuboiddata[i, 6]) - 0.5 * float(cuboiddata[i, 12])
    #     z1 = float(cuboiddata[i, 5]) - 0.5 * float(cuboiddata[i, 11])
    #     # [x1, y1, z1] = np.asarray([x1, y1, z1])
    #     [x1, y1, z1] = np.asarray([x1, y1, z1]) - np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 6]), \
    #                                                           float(cuboiddata[i, 5])])
    #     [x1, y1, z1] = np.dot([x1, y1, z1], A)
    #     # [x1, y1, z1] = np.asarray([x1, y1, z1])
    #     [x1, y1, z1] = np.asarray([x1, y1, z1]) + np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 6]), \
    #                                                           float(cuboiddata[i, 5])])
    #     cupoint.append([x1, y1, z1, 1, 0])
    #     x2 = float(cuboiddata[i, 4]) + 0.5 * float(cuboiddata[i, 10])
    #     y2 = float(cuboiddata[i, 6]) + 0.5 * float(cuboiddata[i, 12])
    #     z2 = float(cuboiddata[i, 5]) - 0.5 * float(cuboiddata[i, 11])
    #     # [x2, y2, z2] = np.asarray([x2, y2, z2])
    #     [x2, y2, z2] = np.asarray([x2, y2, z2]) - np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 6]), \
    #                                                           float(cuboiddata[i, 5])])
    #     [x2, y2, z2] = np.dot([x2, y2, z2], A)
    #     # [x2, y2, z2] = np.asarray([x2, y2, z2])
    #     [x2, y2, z2] = np.asarray([x2, y2, z2]) + np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 6]), \
    #                                                           float(cuboiddata[i, 5])])
    #     cupoint.append([x2, y2, z2, 1, 0])
    #
    #     x3 = float(cuboiddata[i, 4]) + 0.5 * float(cuboiddata[i, 10])
    #     y3 = float(cuboiddata[i, 6]) - 0.5 * float(cuboiddata[i, 12])
    #     z3 = float(cuboiddata[i, 5]) + 0.5 * float(cuboiddata[i, 11])
    #     # [x3, y3, z3] = np.asarray([x3, y3, z3])
    #     [x3, y3, z3] = np.asarray([x3, y3, z3]) - np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 6]), \
    #                                                           float(cuboiddata[i, 5])])
    #     [x3, y3, z3] = np.dot([x3, y3, z3], A)
    #     # [x3, y3, z3] = np.asarray([x3, y3, z3])
    #     [x3, y3, z3] = np.asarray([x3, y3, z3]) + np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 6]), \
    #                                                           float(cuboiddata[i, 5])])
    #     cupoint.append([x3, y3, z3, 1, 0])
    #
    #     x4 = float(cuboiddata[i, 4]) - 0.5 * float(cuboiddata[i, 10])
    #     y4 = float(cuboiddata[i, 6]) - 0.5 * float(cuboiddata[i, 12])
    #     z4 = float(cuboiddata[i, 5]) - 0.5 * float(cuboiddata[i, 11])
    #     # [x4, y4, z4] = np.asarray([x4, y4, z4])
    #     [x4, y4, z4] = np.asarray([x4, y4, z4]) - np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 6]), \
    #                                                           float(cuboiddata[i, 5])])
    #     [x4, y4, z4] = np.dot([x4, y4, z4], A)
    #     # [x4, y4, z4] = np.asarray([x4, y4, z4])
    #     [x4, y4, z4] = np.asarray([x4, y4, z4]) + np.asarray([float(cuboiddata[i, 4]), float(cuboiddata[i, 6]), \
    #                                                           float(cuboiddata[i, 5])])
    #     cupoint.append([x4, y4, z4, 1, 0])
    #     cupoint.append([cuboiddata[i, 4], cuboiddata[i, 6], cuboiddata[i, 5], 1, 0])
    #
    # csvwriter.writerows(cupoint)
    # print(labelcsv + ' done')

def  delete():
    files = glob.glob('*label.csv')
    for file in files:
        with open(file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            f = list(csvreader)
            f = np.array(f)
            a = f.shape[0]
            f = f[:, 3]
            sum = 0
            for i in range(a):
                sum = sum + int(f[i])
            if sum == 0:
                os.remove(file)
                print('Removing ' + file)


def crop():
    labels = '/home/lihaoq/Documents/Convert/Labeled/'
    labels = glob.glob(labels + '*.csv')
    labels = np.array(labels)
    store = '/home/lihaoq/Documents/Convert/Labeled/Crop/'
    k1 = 350 * 128
    k2 = 750 * 128
    for label in labels:
        f = store + label[-14:]
        with open(label, 'r') as csvfile:
            label = csv.reader(csvfile)
            label = list(label)
            label = np.array(label)
            label = label[k1:k2, :]
        with open(f, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerows(label)






if __name__ == '__main__':
    # step 1: convert bag file to pcd files for annotation on supervisely.
    os.system('rosrun pcl_ros bag_to_pcd <input_file.bag> <topic> <output_directory>')
    # input_file.bag: xxx.bag
    # topic: default topic is /ouster/points
    # output_directory: any directory you want the pcds to be saved to

    # step 2: convert annotation from .json to .csv.
    pcd_files = glob.glob('/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/rail_dataset/0/label/rail_dataset/ds0/pointcloud/*.pcd')
    json_files = glob.glob('/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/rail_dataset/0/label/rail_dataset/ds0/ann/*.pcd.json')
    json22csv(json_files)

    # Specify data path and annotation path
    point_path = '/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/rail_dataset/0/label/rail_dataset/ds0/pointcloud/'
    json_path = '/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/rail_dataset/0/label/rail_dataset/ds0/ann/'
    json = glob.glob('/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/rail_dataset/0/label/rail_dataset/ds0/ann/*.pcd.csv')
    label_path = json_path.replace('ann/', 'labeled/')
    os.makedirs(label_path) if not os.path.exists(label_path) else None
    # Label each frame with annotation
    for jsoncsv in json:
        cpcsv = jsoncsv.replace('.pcd.csv', '.csv')
        cpcsv = cpcsv.replace('ann', 'pointcloud')
        labelcsv = jsoncsv.replace('.pcd.csv', '_label.csv')
        labelcsv = labelcsv.replace('ann', 'labeled')
        labelpt(cpcsv, jsoncsv, labelcsv)
