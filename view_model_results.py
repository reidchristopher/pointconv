"""
Modified from PointNet++: https://github.com/charlesq34/pointnet2
Author: Wenxuan Wu
Date: July 2018
"""
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'scannet'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='fluid_sim_model', help='Model name [default: model]')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 501]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
EPOCH_CNT_WHOLE = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BANDWIDTH = 0.05

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
Point_Util = os.path.join(BASE_DIR, 'utils', 'pointconv_util.py')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_STATES = 6

def log_string(out_str):

    print(out_str)



def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def run():
    with tf.Graph().as_default():
        with tf.device('/gpu:1'):
            point_pl, point_labels_pl, beam_labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            num_point_pl = tf.placeholder(tf.int32, shape=())

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model")
            # Get model and loss
            point_pred, beam_pred = MODEL.get_model(point_pl, num_point_pl, is_training_pl, NUM_STATES, BANDWIDTH,
                                                    bn_decay=bn_decay)


        saver = tf.train.Saver()
        model_file = "/home/rchristopher/classes/cs535/project/pointconv/log2020_03_20_19_12_41/model.ckpt"

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, model_file)

        # Add summary writers
        merged = tf.summary.merge_all()

        ops = {'point_pl': point_pl,
               'point_labels_pl': point_labels_pl,
               'point_pred': point_pred,
               'beam_labels_pl': beam_labels_pl,
               'beam_pred': beam_pred,
               'num_point_pl': num_point_pl,
               'is_training_pl': is_training_pl,
               'merged': merged,
               'step': batch}

        '''
        point to experiment numpy file
        '''
        test_file = "/home/rchristopher/classes/cs535/project/cs535_project/Data/data_0.146_0.292_0.08_0.292.npz"

        # raw_input("Press enter to load data")

        data = np.load(test_file)
        sim_points = data['points']
        inputs = data['inputs']
        outputs = data['outputs']
        times = np.array(data['times'], dtype=np.float32)
        time_diff = times[10::10] - times[:-10:10]

        net_points = [None for _ in range(len(sim_points))]
        point_state = sim_points[0]

        print("--- Data Loaded ---")

        import matplotlib.pyplot as plt
        import time

        total_time = 0
        plt.ion()
        beam_outputs = np.zeros((len(time_diff), 3))
        for i in range(len(time_diff)):
            print("Time step %d" % i)

            point_x = point_state[:, 0]
            point_y = point_state[:, 1]

            start = time.time()

            # raw_input("Press enter to run network")

            points_dot, beam_state = sess.run([point_pred, beam_pred], feed_dict={ops['point_pl'] : [point_state],
                                                            ops['is_training_pl'] : False,
                                                            ops['num_point_pl']: sim_points.shape[1]})

            # beam_loss = tf.multiply(tf.losses.absolute_difference(labels=outputs[i], predictions=beam_state[0]), 0.01)
            #
            # print(beam_state[0])
            # print(outputs[i])
            # print(sess.run(beam_loss))

            beam_outputs[i] = beam_state[0]

            plt.clf()
            plt.scatter(point_x, point_y)
            plt.scatter(sim_points[i * 10, :, 0], sim_points[i * 10, :, 1])

            plt.plot(times[:-10:10], beam_outputs[:, 0])
            plt.plot(inputs[:, -1], outputs[:, 0])

            plt.draw()
            plt.show()
            plt.pause(0.001)

            # network technically predicts an approximate time derivative of states
            # this is because dts were non constant
            # get predicted change by multiplying prediction with dt
            point_state[:, :-3] += points_dot[0] * time_diff[i]

            # make sure to ignore Z-data
            point_state[:, 2] = 0

            point_state = sim_points[i * 10, :, :]
            net_points[i] = point_state[:, :3]




if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    run()
