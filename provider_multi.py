import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')

def point_cloud_to_volume(points, vsize, radius=1.0):
    vol = np.zeros((vsize,vsize,vsize,1))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2],0] = 1.0
    return vol

def shuffle_data(las_ground, las_1):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(las_ground))
    np.random.shuffle(idx)
    # idx2 = np.arange(len(las_1))
    # np.random.shuffle(idx2)

    print "shuffled!"
    return las_ground[idx, ...], las_1[idx, ...]

def shuffle_data2(las_ground, las_1):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx1 = np.arange(len(las_ground))
    np.random.shuffle(idx1)
    idx2 = np.arange(len(las_1))
    np.random.shuffle(idx2)
    voxel1 = []
    voxel_ground = []
    print "shuffled!"
    for i in range(las_1.shape[0]):
        voxel1.append(point_cloud_to_volume(las_1[i], 32))
        voxel_ground.append(point_cloud_to_volume(las_ground[i], 32))
        # print voxel1[i]
    voxel1 = np.array(voxel1)
    voxel_ground = np.array(voxel_ground)
    return voxel_ground[idx1, ...], voxel1[idx2, ...]

def data_to_voxel(las_ground,las_1):
    voxel1 = []
    voxel_ground = []
    for i in range(las_1.shape[0]):
        voxel1.append(point_cloud_to_volume(las_1[i], 64))
        voxel_ground.append(point_cloud_to_volume(las_ground[i], 64))
        # print voxel1[i]
    voxel1 = np.array(voxel1)
    voxel_ground = np.array(voxel_ground)

    return voxel_ground, voxel1

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    # for k in xrange(batch_data.shape[0]):
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    print batch_data.shape
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    # image = f['image'][:]
    las_ground = f['las_ground'][:] 
    las_1 = f['las_1'][:] 
    las_2 = f['las_2'][:] 
    las_3 = f['las_3'][:]
    print "Loaded!"
    return (las_ground, las_1, las_2, las_3)

def load_h5_2(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h51(h5_filename):
    f = h5py.File(h5_filename)
    # image = f['image'][:]
    las_ground = f['las_ground'][:] 
    las_1 = f['las_1'][:] 
    las_2 = f['las_2'][:] 
    las_3 = f['las_3'][:]
    print "Loaded!"
    return (las_ground, las_1, las_2, las_3)
    

def loadDataFile(filename):
    return load_h5_2(filename)

