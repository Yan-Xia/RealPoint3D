import os
import sys
import argparse
import tensorflow as tf
import provider
import numpy as np
import cv2
# import PSGN_encoder
from tensorflow.python.framework import ops
os.environ["CUDA_DEVICE_ORDER" ] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES" ] = "0"
import pointnet2_encoder
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')

parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='/public/home/scsc/xy/GBNET-6/log_v4_128_car_pure/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')

parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', type=bool, default=False, help='Whether to dump image for error case [default: False]')
parser.add_argument('--num', type=int, default=2, help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()

OUT_DIR_IMG = 'out_v4_128_car_apolo_6/img'
OUT_DIR_LAS = 'out_v4_128_car_apolo_6/las'
if not os.path.exists(OUT_DIR_IMG):
    os.mkdir(OUT_DIR_IMG)

if not os.path.exists(OUT_DIR_LAS):
    os.mkdir(OUT_DIR_LAS)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
num=FLAGS.num
# MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
nn_distance_module=tf.load_op_library('./tf_nndistance_so.so')


# NUM_CLASSES = 40
# SHAPE_NAMES = [line.rstrip() for line in \
#     open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

# HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
# TRAIN_FILES = provider.getDataFiles( \
#     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/apolo_test/test.txt'))
# def get_bn_decay(batch):
#     bn_momentum = tf.train.exponential_decay(
#                       BN_INIT_DECAY,
#                       batch*BATCH_SIZE,
#                       BN_DECAY_DECAY_STEP,
#                       BN_DECAY_DECAY_RATE,
#                       staircase=True)
#     bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
#     return bn_decay

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

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        # pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        #is_training_pl = tf.placeholder(tf.bool, shape=())

        ######################   our   #########################

        pointclouds_pl, gt_pl, img_pl = pointnet2_encoder_v4.placeholder_inputs(num, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        pred_points, _ = pointnet2_encoder_v4.get_model(pointclouds_pl, img_pl, is_training_pl)


        ######################   PSGN   #########################

        # gt_pl, img_pl = PSGN_encoder.placeholder_inputs(num, NUM_POINT)
        # is_training_pl = tf.placeholder(tf.bool, shape=())
        # pred_points = PSGN_encoder.get_model( img_pl, is_training_pl)



        reta, retb, retc, retd = nn_distance(gt_pl, pred_points)
        loss = tf.reduce_sum(reta) + tf.reduce_sum(retc)
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")
    
    ######################   our   #########################

    ops_ = {'pointclouds_pl': pointclouds_pl,
           'gt_pl': gt_pl,
           'is_training_pl': is_training_pl,
           'img_pl': img_pl,
           'pred_points':pred_points,
           'loss': loss}

    ######################   PSGN   #########################

    # ops_ = {'gt_pl': gt_pl,
    #        'is_training_pl': is_training_pl,
    #        'img_pl': img_pl,
    #        'pred_points':pred_points,
    #        'loss': loss}

    eval_one_epoch(sess, ops_, num_votes)

   
def eval_one_epoch(sess, ops_, num_votes=1, topk=1):
    # error_cnt = 0
    is_training = False
    # total_correct = 0
    # total_seen = 0
    # loss_sum = 0
    # total_seen_class = [0 for _ in range(NUM_CLASSES)]
    # total_correct_class = [0 for _ in range(NUM_CLASSES)]
    # fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        # current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        # current_data = current_data[:,0:NUM_POINT,:]
        # current_label = np.squeeze(current_label)
        # print current_data.shape

        ######################   our   #########################

        current_data, current_retrival, current_image = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_retrival = current_retrival[:, 0:NUM_POINT, :]
        
        ######################   PSGN   #########################

        # current_data, _, current_image = provider.loadDataFile(TEST_FILES[fn])
        # current_data = current_data[:, 0:NUM_POINT, :]
        


        file_size = current_data.shape[0]
        # num_batches = file_size / BATCH_SIZE
        # print file_size
        
        # for batch_idx in range(num):
        start_idx =  0
        end_idx = num
        # cur_batch_size = end_idx - start_idx
        img_batch=current_image[start_idx:end_idx]
        img_batch=img_batch.reshape((-1,128,128,3))
        gt_data=current_data[start_idx:end_idx,:,:]

        loss_sum = 0  # sum of losses for the batch
        for vote_idx in range(num_votes):

            ######################   our   #########################

            rotated_data = provider.rotate_point_cloud_by_angle(current_retrival[start_idx:end_idx, :, :],
                                              vote_idx/float(num_votes) * np.pi * 2)
            
            feed_dict = {ops_['gt_pl']:gt_data,
            			 ops_['pointclouds_pl']: rotated_data,
                         ops_['img_pl']: img_batch,
                         ops_['is_training_pl']: is_training}
            

            #####################   PSGN   #########################

            # feed_dict = {ops_['gt_pl']:gt_data,
            #              ops_['img_pl']: img_batch,
            #              ops_['is_training_pl']: is_training}

            pred_points, loss_val = sess.run(
                [ops_['pred_points'], ops_['loss']], feed_dict=feed_dict)
            print 'pred_points'
            print pred_points.shape
            print 'loss_val: ', loss_val
            for i in range(num):
                print i
                print pred_points[i,...].shape  

                #####################   our   #########################

                np.savetxt(OUT_DIR_LAS + '/pre_%d.txt' % i, pred_points[i,...])
                np.savetxt(OUT_DIR_LAS + '/gt_%d.txt' % i, gt_data[i,...])
                np.savetxt(OUT_DIR_LAS + '/retrival_%d.txt' % i, current_retrival[i,...])
                cv2.imwrite(OUT_DIR_IMG + '/img_%d.jpg' % i, img_batch[i,...])

                #####################   PSGN   #########################

                # np.savetxt('out_PSGN/las/pre_%d.txt' % i, pred_points[i,...])
                # np.savetxt('out_PSGN/las/gt_%d.txt' % i, gt_data[i,...])
                # cv2.imwrite('out_PSGN/img/img_%d.jpg' % i, img_batch[i,...])

            loss_sum += loss_val
            print loss_val
            log_string('one batch loss: %f' % loss_val)
        log_string('mean loss: %f' % (loss_sum / float(num)))



if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
