from os import makedirs
from os.path import exists, join, isfile, dirname, abspath
from helper_tool import DataProcessing as DP
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import yaml
import pickle
import time
from datetime import date, datetime

BASE_DIR = dirname(abspath(__file__))

data_config = join(BASE_DIR, 'utils', 'Ouster.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map_inv"]

# make lookup table for mapping
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

remap_dict_val = DATA["learning_map"]
max_key = max(remap_dict_val.keys())
remap_lut_val = np.zeros((max_key + 36830), dtype=np.int32)
remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open('log_test_' + dataset.name + '.txt', 'a')

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)
        self.test_probs = 0
        self.idx = 0

    def test(self, model, dataset):

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)
        self.test_probs = [np.zeros(shape=[len(l), model.config.num_classes], dtype=np.float16)
                           for l in dataset.possibility]

        test_path = join('test', 'sequences')
        makedirs(test_path) if not exists(test_path) else None
        save_path = join(test_path, 'predictions')
        makedirs(save_path) if not exists(save_path) else None
        test_smooth = 0.98
        epoch_ind = 0

        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M%S")
        while True:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],
                       model.inputs['xyz'])
                stacked_probs, labels, point_inds, cloud_inds, points = self.sess.run(ops, {model.is_training: False})

                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size,
                                                           model.config.num_points,
                                                           model.config.num_classes])

                # Save prediction
                pred_valid = np.argmax(stacked_probs, 1)
                print('pred_valid shape: ' + str(pred_valid.shape))
                labels_valid = labels

                if self.idx % 10 == 0:
                    print('step ' + str(self.idx))
                    pc_xyz_sess = points
                    pc_xyz_sess = pc_xyz_sess[0, :, :]
                    # print('pc_xyz_sess shape: ' + str(pc_xyz_sess.shape))
                    # print('pc_xyz_sess shape: ' + str(pc_xyz_sess.shape))
                    labels_sess = labels
                    pred_sess = pred_valid
                    output = np.insert(pc_xyz_sess, 3, pred_sess, axis=1)
                    print('BASE_DIR: ' + BASE_DIR)
                    save_dir = join(BASE_DIR, '/data/Ouster_crop/test_result')
                    save_dir = join(save_dir, dt_string)
                    makedirs(save_dir) if not exists(save_dir) else None
                    np.savetxt(join(save_dir, str(self.idx)) + '.csv', output, delimiter=',')
                    print('File number self.idx = ' + str(self.idx))
                    # np.savetxt(os.path.join())

                self.idx += 1

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_inds[j, :]
                    c_i = cloud_inds[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs

            except tf.errors.OutOfRangeError:
                new_min = np.min(dataset.min_possibility)
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_ind, new_min), self.Log_file)
                if np.min(dataset.min_possibility) > 0.5:  # 0.5
                    log_out(' Min possibility = {:.1f}'.format(np.min(dataset.min_possibility)), self.Log_file)
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))

                    # For validation set
                    num_classes = 3
                    gt_classes = [0 for _ in range(num_classes)]
                    positive_classes = [0 for _ in range(num_classes)]
                    true_positive_classes = [0 for _ in range(num_classes)]
                    val_total_correct = 0
                    val_total_seen = 0

                    for j in range(len(self.test_probs)):
                        test_file_name = dataset.test_list[j]
                        frame = test_file_name.split('/')[-1][:-4]
                        proj_path = join(dataset.dataset_path, 'proj')
                        proj_file = join(proj_path, str(frame) + '_proj.pkl')

                        if isfile(proj_file):
                            with open(proj_file, 'rb') as f:
                                proj_inds = pickle.load(f)
                        probs = self.test_probs[j][proj_inds[0], :]
                        pred = np.argmax(probs, 1)
                        # if dataset.test_scan_number == '08':
                        label_path = join(dirname(dataset.dataset_path), 'Ouster_crop', 'labels')
                        label_file = join(label_path, str(frame) + '.npy')
                        labels = DP.load_label_kitti(label_file, remap_lut_val)
                        invalid_idx = np.where(labels == 0)[0]
                        labels_valid = np.delete(labels, invalid_idx)
                        # pred_valid = np.delete(pred, invalid_idx)
                        pred_valid = pred
                        labels_valid = labels_valid - 1
                        correct = np.sum(pred_valid == labels_valid)
                        val_total_correct += correct
                        val_total_seen += len(labels_valid)
                        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, num_classes, 1))
                        gt_classes += np.sum(conf_matrix, axis=1)
                        positive_classes += np.sum(conf_matrix, axis=0)
                        true_positive_classes += np.diagonal(conf_matrix)
                        # else:
                        #     store_path = join(test_path, dataset.test_scan_number, 'predictions',
                        #                       str(frame) + '.label')
                        #     pred = pred + 1
                        #     pred = pred.astype(np.uint32)
                        #     upper_half = pred >> 16  # get upper half for instances
                        #     lower_half = pred & 0xFFFF  # get lower half for semantics
                        #     lower_half = remap_lut[lower_half]  # do the remapping of semantics
                        #     pred = (upper_half << 16) + lower_half  # reconstruct full label
                        #     pred = pred.astype(np.uint32)
                        #     pred.tofile(store_path)
                    log_out('finished', self.Log_file)
                    # if dataset.test_scan_number=='08':
                    iou_list = []


                    for n in range(0, num_classes, 1):
                        print('gt_classes[n]: ' + str(gt_classes[n]))
                        print('positive_classes[n]: ' + str(positive_classes[n]))
                        print('true_positive_classes[n]: ' + str(true_positive_classes[n]))
                        iou = true_positive_classes[n] / float(
                            gt_classes[n] + positive_classes[n] - true_positive_classes[n])
                        iou_list.append(iou)
                    mean_iou = sum(iou_list) / float(num_classes)

                    log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
                    log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

                    mean_iou = 100 * mean_iou
                    print('Mean IoU = {:.1f}%'.format(mean_iou))
                    s = '{:5.2f} | '.format(mean_iou)
                    for IoU in iou_list:
                        s += '{:5.2f} '.format(100 * IoU)
                    print('-' * len(s))
                    print(s)
                    print('-' * len(s) + '\n')
                    self.sess.close()
                    return
                self.sess.run(dataset.test_init_op)
                epoch_ind += 1
                continue
