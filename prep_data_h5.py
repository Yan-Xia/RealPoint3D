import h5py
import numpy as np
from liblas import file
import os
from random import choice
import cv2
import pickle

category = 'airplane'
point_n = 1024
img_size = [128,128]
train_ratio = 0.8
view = 'randview'   #  multi or fixed

las_num = 1024
pkl_file=open('/home/xy/NewDisk/h5_src/%s_h5/dirlist_%s.pkl'%(category,category),'rb')
ret=pickle.load(pkl_file)
dir_las = '/home/xy/NewDisk/h5_src/%s_h5/las'%(category)
dir_pic = '/home/xy/NewDisk/h5_src/%s_h5/v0/img_%s_v0'%(category,view)

def create_h5(name,data, retrival, image):
    file = h5py.File(name, 'w')
    file.create_dataset('data', data=data)
    file.create_dataset('retrival', data=retrival)
    file.create_dataset('image', data=image)
    file.close()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:] 
    retrival = f['retrival'][:]
    image = f['image'][:]
    return (data, retrival,image)

def read_las(las_filepath):
    points = []
    p = []
    f = file.File(las_filepath)
    for point in f:
        p = [point.x, point.y, point.z]
        points.append(p)
    return points

def read_retrieval_las(las_filepath,pointnum):
    points = []
    p = []
    f = file.File(las_filepath)
    for point in f:
        p = [point.x, point.y, point.z]
        points.append(p)
    points_one=[]
    if len(points)>point_n:
        points=np.array(points)
        idx = np.arange(len(points))
        np.random.shuffle(idx)
        temp_data=points[idx,...]
        points_one=temp_data[0:point_n]
    return points_one
all_tup=[]
image=[]
retrieval=[]


dir_las_list = []
dir_pic_list = []
dir_pic_model = []
dir_pic_list = os.listdir(dir_pic)

for index, i in enumerate(dir_pic_list):
    basename = i
    lasname = basename.split('.')[0] +'.las'
    print 'read_las:'+os.path.join(dir_las,lasname)
    rp = read_las(os.path.join(dir_las,lasname))
    lis=np.array(rp)
    idx = np.arange(len(rp))
    np.random.shuffle(idx)
    temp_data=lis[idx,...]
    lis_one=temp_data[0:point_n]

    all_tup.append(lis_one)
    retrieval.append(lis_one)

    print 'read_pic: '+os.path.join(dir_pic, i)
    img = cv2.imread(os.path.join(dir_pic, i))
    img = cv2.resize(img, (img_size[0],img_size[1]))
    image.append(img)
    print os.path.join(dir_las, os.path.basename(ret[i]).split('.')[0].split('_')[0]+'.las')
    rp = read_retrieval_las(os.path.join(dir_las, os.path.basename(ret[i]).split('.')[0].split('_')[0]+'.las') , point_n)
    if len(rp)>point_n:
        rp1 = np.array(rp1)
        idx_r = np.arange(len(rp1))
        np.random.shuffle(idx_r)
        temp_rp = rp[idx_r,...]
        rp_one = temp_rp[0:point_n]

        #print len(lis_one)
        #print len(rp_one)
        # all_tup.append(lis_one)
        image.append(img)
        retrieval.append(rp_one)

print 'all_tup:'+str(len(all_tup))
print 'image:'+str(len(image))
print 'retrieval:'+str(len(retrieval))

idx = int(train_ratio*len(all_tup))

create_h5('%s_%s_las_%s_%s_train.h5'%(category,str(point_n),str(img_size[0]),view), all_tup[0:idx],retrieval[0:idx],image[0:idx])
create_h5('%s_%s_las_%s_%s_test.h5'%(category,str(point_n),str(img_size[0]),view), all_tup[idx:],retrieval[idx:],image[idx:])

print 'train:'+str(idx)
print 'test:'+str(len(all_tup)-idx)
