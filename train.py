""" 
Author: Yan Xia
"""
import os
import sys
import argparse
import tensorflow as tf
import provider
import numpy as np
from tensorflow.python.framework import ops

import pointnet2_encoder
from PIL import Image
os.environ["CUDA_DEVICE_ORDER" ] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES" ] = "4"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log_v4_128_car_pure', help='Log dir [default: log]')
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
# GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

nn_distance_module=tf.load_op_library('./tf_nndistance_so.so')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/shapenet/train_files.txt'))

def print_config():
    print('%s : %s' % ('log_dir', FLAGS.log_dir))
    print('%s : %s' % ('num_point', FLAGS.num_point))
    print('%s : %s' % ('max_epoch', FLAGS.max_epoch))
    print('%s : %s' % ('batch_size', FLAGS.batch_size))
    print('%s : %s' % ('learning_rate', FLAGS.learning_rate))
    print('%s : %s' % ('momentum', FLAGS.momentum))
    print('%s : %s' % ('optimizer', FLAGS.optimizer))
    print('%s : %s' % ('decay_step', FLAGS.decay_step))
    print('%s : %s' % ('decay_rate', FLAGS.decay_rate))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def nn_distance(xyz1,xyz2):
    """
Computes the distance of nearest neighbors for a pair of point clouds
input: xyz1: (batch_size,#points_1,3)  the first point cloud
input: xyz2: (batch_size,#points_2,3)  the second point cloud
output: dist1: (batch_size,#point_1)   distance from first to second
output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
output: dist2: (batch_size,#point_2)   distance from second to first
output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    """
    return nn_distance_module.nn_distance(xyz1,xyz2)

@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
    xyz1=op.inputs[0]
    xyz2=op.inputs[1]
    idx1=op.outputs[1]
    idx2=op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1,xyz2,grad_dist1,idx1,grad_dist2,idx2)

def train():
    with tf.Graph().as_default():
        # with tf.device('/gpu:'+str(GPU_INDEX)):
        
        print_config()

        """num_point = 1024 """
        is_training_pl = tf.placeholder(dtype = tf.bool, shape=())
        # keep_prob_pl = tf.placeholder(dtype = tf.float32, shape=())

        """get pointnet2 input placeholder"""
        pointclouds_pl, gt_pl, img_pl = pointnet2_encoder_v4.placeholder_inputs(BATCH_SIZE, NUM_POINT)

        batch = tf.Variable(0)
        bn_decay = get_bn_decay(batch)
        tf.summary.scalar('bn_decay', bn_decay)

        """input image and point cloud to encoder to get offset of every point"""
        # point_offset, _ = pointnet2_encoder.get_model(pointclouds_pl, img_pl, is_training_pl, keep_prob_pl, bn_decay = bn_decay)
        point_offset, _ = pointnet2_encoder_v4.get_model(pointclouds_pl, img_pl, is_training_pl, bn_decay = bn_decay)
        '''print(point_offset.get_shape()) => (32, 1024, 3)'''

        """use offset to ori point cloud to get new point cloud"""
        # pred_points = tf.add(pointclouds_pl, point_offset)
        pred_points = point_offset

        print 'gt_pl'
        print gt_pl.shape
        print type(gt_pl)
        print 'pred_points'
        print pred_points.shape
        print type(pred_points)


        """compute new point cloend_idxud with ori pc with CD distance (loss)"""
        reta, retb, retc, retd = nn_distance(gt_pl, pred_points)
        # reta, retb, retc, retd = nn_distance(pointclouds_pl, pred_points)
        # loss = tf.reduce_sum(reta) + tf.reduce_sum(retc)
        mindist=reta
        dist0 = mindist[0, :]
        dists_forward = tf.reduce_mean(reta)
        dists_backward = tf.reduce_mean(retc)
        loss_nodecay = (dists_forward + dists_backward / 2.0) * 1000000
        loss = loss_nodecay + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.1

        """tensorborad show loss"""
        tf.summary.scalar('loss', loss)

        """Get training operator"""
        learning_rate = get_learning_rate(batch)
        tf.summary.scalar('learning_rate', learning_rate)

        """choose BP algorithmn"""
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        if OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op = optimizer.minimize(loss, global_step=batch)

        """Add op_dic to save and restore all the variables."""
        saver = tf.train.Saver()
        # with tf.device('/gpu:'+str(GPU_INDEX)):

        '''Create a session'''
        config = tf.ConfigProto()
        
        # config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        '''Add summary writers'''
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
        # test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        '''Init variables'''
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init, {is_training_pl: True})

        xnet_saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./log_v4_128')
        if ckpt and ckpt.model_checkpoint_path:
        	xnet_saver.restore(sess, ckpt.model_checkpoint_path)
        	print '*** model is loaded ***\n'


        print pointclouds_pl.shape
        print gt_pl.shape
        print type(gt_pl)

        op_dic = {'pointclouds_pl': pointclouds_pl,
                'gt_pl': gt_pl,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'img_pl':img_pl}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()

            train_one_epoch(sess, op_dic, train_writer)
            # eval_one_epoch(sess, op_dic, test_writer)

            '''Save the variables to disk.'''
            if epoch % 15 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, op_dic, train_writer):
    """ op_dic: dict mapping from string to tf op_dic """
    is_training = True

    # Shuffle train files
    # train_file_idxs = {0,1,2,3,4}
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    c = 0
    for fn in range(len(TRAIN_FILES)):
        log_string('----- train : ' + str(fn) + '-----')
        current_data, current_retrival, current_img = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        # current_data, current_retrival, _, current_img = provider.shuffle_data(current_data, np.squeeze(current_retrival), current_img)
        
        current_retrival = current_retrival[:,0:NUM_POINT,:]
        current_data, current_retrival, _, current_img = provider.shuffle_data(current_data, current_retrival, current_img)
        
        file_size = current_retrival.shape[0]
        num_batches = file_size / BATCH_SIZE

        '''read image'''

        loss_sum = 0
        for batch_idx in range(int(num_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            '''get image by start_idx and end_idx'''
            img_batch = current_img[start_idx:end_idx]
            img_batch = img_batch.reshape((-1, 128, 128, 3))
            '''next i should write a loop for input all three view into network'''
            '''now i just input the first view'''
            '''now img_batch shape is (BATCH_SIZE, 500, 500, 3)'''
            # img_batch = img_batch[:, :, 0:256, :]
            # img_batch = img_batch[:]

            ###########add groundtruth#######
            gt_data=current_data[start_idx:end_idx, :, :]
             ###########add groundtruth#######
            '''Augment batched 49point clouds by rotation and jittering'''
            rotated_data = provider.rotate_point_cloud(current_retrival[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)

            # _photo_batch = sess.run(photo_batch)
            # _photo_batch = np.random.rand(BATCH_SIZE, 256, 256, 3)
            feed_dict = {op_dic['pointclouds_pl']: jittered_data,
                         op_dic['gt_pl']: gt_data,
                         op_dic['is_training_pl']: is_training,
                         op_dic['img_pl']: img_batch}
                         
            
            summary, step, _, loss_val = sess.run([op_dic['merged'], op_dic['step'], op_dic['train_op'], op_dic['loss']], feed_dict = feed_dict)
            
            train_writer.add_summary(summary, step)
            loss_sum += loss_val
            # log_string('one batch loss: %f' % loss_val)
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
