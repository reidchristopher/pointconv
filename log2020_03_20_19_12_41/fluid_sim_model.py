import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from PointConv import feature_encoding_layer, feature_decoding_layer

def placeholder_inputs(batch_size, num_point):
    point_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 9))
    point_labels_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 6))
    beam_labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    return point_pl, point_labels_pl, beam_labels_pl

def get_model(point_cloud, num_point, is_training, num_states, sigma, bn_decay=None, weight_decay = None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = point_cloud.get_shape()[0].value
    end_points = {}
    l0_xyz = point_cloud[:, :, 0:3]
    l0_points = point_cloud

    # Feature encoding layers
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.01, sigma = sigma, K=32, mlp=[8,8,16], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.02, sigma = 2 * sigma, K=32, mlp=[16,16,32], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.04, sigma = 4 * sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    # l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[100,100,120], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')

    # beam predictions
    # flatten l4_points
    # run two ReLU layers
    # have 3 outputs
    _, single_point = feature_encoding_layer(l3_xyz, l3_points, npoint=1, radius=100, sigma=16 * sigma, K=36,
                                               mlp=[64, 64, 128], is_training=is_training, bn_decay=bn_decay,
                                               weight_decay=weight_decay, scope='beam_encoding_layer')
    flattened_encoding = tf.layers.flatten(single_point)
    beam_output = tf.layers.dense(flattened_encoding, 128, activation=tf.nn.leaky_relu)
    beam_output = tf.layers.dense(beam_output, 64, activation=tf.nn.leaky_relu)
    beam_output = tf.layers.dense(beam_output, 3, activation=None)

    # Feature decoding layers
    # l3_points = feature_decoding_layer(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8 * sigma, 16, [120,120], is_training, bn_decay, weight_decay, scope='fa_layer1')
    l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, 0.04, 4 * sigma, 16, [64,32], is_training, bn_decay, weight_decay, scope='fa_layer2')
    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 0.02, 2 * sigma, 16, [64,32], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.01, sigma, 16, [64,64,64], is_training, bn_decay, weight_decay, scope='fa_layer4', npoint=num_point)

    # FC layers
    point_output = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = point_output
    point_output = tf_util.dropout(point_output, keep_prob=0.5, is_training=is_training, scope='dp1')
    point_output = tf_util.conv1d(point_output, num_states, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc2')

    return point_output, beam_output


def get_loss(point_pred, point_label, beam_pred, beam_label, num_point):
    """ pred: BxNxC,
        label: BxN,
    smpw: BxN """

    weights = [[[1, 1, 0, 1, 1, 2e-6]]] # [x, y, z, vx, vy, p]
    point_loss = tf.div(tf.losses.absolute_difference(labels=point_label, predictions=point_pred, weights=weights), tf.cast(num_point, tf.float32))
    beam_loss = tf.multiply(tf.losses.absolute_difference(labels=beam_label, predictions=beam_pred), 0.01)

    weight_reg = tf.add_n(tf.get_collection('losses')) * 1e-6
    total_loss = point_loss + beam_loss + weight_reg
    tf.summary.scalar('point loss', point_loss)
    tf.summary.scalar('beam_loss', beam_loss)
    tf.summary.scalar('weight_reg_loss', weight_reg)
    tf.summary.scalar('total loss', total_loss)
    return total_loss

if __name__=='__main__':
    # import pdb
    # pdb.set_trace()

    with tf.device("GPU:1"):

        inputs = tf.random_normal((1,2048,3))
        point_predictions, beam_predictions = get_model(inputs, tf.constant(True), 10, 1.0)
        init_g = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        print(session.run(inputs))
        session.run(init_g)
        print(session.run(point_predictions))
        print(session.run(beam_predictions))

    print(point_predictions)
    print(beam_predictions)
