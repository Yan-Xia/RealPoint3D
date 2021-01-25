""" 
Author: Yan Xia
"""
import os
import sys
from doctest import master
import  tflearn
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import pointnet2.utils.tf_util as tf_util
from pointnet2.utils.pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    gt_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    img_pl = tf.placeholder ( tf.float32, shape=[batch_size, 128, 128, 3])
    return pointclouds_pl, gt_pl, img_pl


def get_model(point_cloud, image, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    print 'l1',l1_points.shape
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    print 'l2',l2_points.shape
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    print 'l3',l3_points.shape
    
    # net = tf.reshape(l3_points, [batch_size,8,8,16])
    # print 'net',net.shape
    net = tf.reshape(l3_points, [batch_size, -1])

    ########image conv#########
    x=image

    x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
    print '1'
    print x.shape
    # x=tflearn.layers.conv.max_pool_2d(x,16)
    x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
    print '2'
    print x.shape
    x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
    print '3'
    print x.shape
    # 96 128
    x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
    print '4'
    print x.shape
    x1=x
    x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
    print '5'
    print x.shape
    img_feamaps=tf.reshape(x,[batch_size,2048])
    print img_feamaps.shape


    x_fusion = tf.concat([img_feamaps, net], 1)
    print 'x_fusion',x_fusion.shape

    x_additional = tflearn.layers.core.fully_connected(x_fusion, 2048, activation='relu', weight_decay=1e-3, regularizer='L2')
    x_additional = tflearn.layers.core.fully_connected(x_additional, 2048, activation='relu', weight_decay=1e-3, regularizer='L2')
    print 'x_additional',x_additional.shape

    x=tf.reshape(x_additional,[batch_size,16,16,8])
    print 'reshape',x.shape
    x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')

    x_deconv=tflearn.layers.conv.conv_2d_transpose(x,32,[5,5],[16,16],strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    print 'x_deconv',x_deconv.shape
    x=tf.nn.relu(tf.add(x_deconv,x1))
    print 'first add',x.shape 

    x = tflearn.layers.conv.conv_2d(x, 128, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
    print '6',x.shape 
    x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
    print '7',x.shape 
    x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
    print '8',x.shape 
    x_additional = tflearn.layers.core.fully_connected(x, 1024*3, activation='linear', weight_decay=1e-5, regularizer='L2')
    print 'x_additional',x_additional.shape

    x_out = tf.reshape(x_additional, (batch_size, 1024, 3))
    return x_out, end_points

 





