"""Test ImageNet pretrained DenseNet"""
from __future__ import print_function
import sys
sys.path.insert(0,'Keras-2.0.8')
from multiprocessing.dummy import Pool as ThreadPool
import random
from medpy.io import load
import numpy as np
import argparse
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from loss import weighted_crossentropy_2ddense
import os
from keras.utils2.multi_gpu import make_parallel
from denseunet import DenseUNet
from skimage.transform import resize
K.set_image_dim_ordering('tf')

#  global parameters
parser = argparse.ArgumentParser(description='Keras 2d denseunet Training')
#  data folder
parser.add_argument('-data', type=str, default='data/', help='test images')
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
# lizx: batch_size
parser.add_argument('-batch_size', type=int, default=40)
parser.add_argument('-input_size', type=int, default=512)
parser.add_argument('-model_weight', type=str, default='./model/densenet161_weights_tf.h5')
parser.add_argument('-input_cols', type=int, default=3)
parser.add_argument('-n_gpus',type=int,default=1)
#  data augment
parser.add_argument('-mean', type=int, default=48)
parser.add_argument('-thread_num', type=int, default=14)
args = parser.parse_args()
print(args)

MEAN = args.mean
thread_num = args.thread_num

liverlist = [32,34,38,41,47,87,89,91,105,106,114,115,119]

N_TRAINING_SAMPLES=2
def slice_dimension(array,i_min,i_max,dim):
    if i_min>=i_max:
        return np.array([])
    else:
        if i_min>=0:
            pad1=0
            start_idx=i_min
        else:
            pad1=-i_min
            start_idx=0
        if i_max<=array.shape[dim]:
            pad2=0
            end_idx=i_max
        else:
            pad2=i_max-array.shape[dim]
            end_idx=array.shape[dim]
        concat_list=[]
        if pad1>0:
            shape=list(array.shape)
            shape[dim]=pad1
            concat_list.append(np.zeros(shape))
        concat_list.append(np.take(array,range(start_idx,end_idx),dim))
        if pad2>0:
            shape=list(array.shape)
            shape[dim]=pad2
            concat_list.append(np.zeros(shape))
        return np.concatenate(concat_list,dim)
        
def slice_array(array,a_min,a_max,b_min,b_max,c_min,c_max):
    assert len(array.shape)==3
    array=slice_dimension(array,a_min,a_max,0)
    array=slice_dimension(array,b_min,b_max,1)
    array=slice_dimension(array,c_min,c_max,2)
    return array

def load_seq_crop_data_masktumor_try(Parameter_List):
    img = Parameter_List[0]
    tumor = Parameter_List[1]
    lines = Parameter_List[2]
    numid = Parameter_List[3]
    minindex = Parameter_List[4]
    maxindex = Parameter_List[5]
    #  randomly scale
    scale = np.random.uniform(0.8,1.2)
    deps = int(args.input_size * scale)
    rows = int(args.input_size * scale)
    cols = 3

    sed = np.random.randint(1,numid)
    cen = lines[sed-1]
    cen = np.fromstring(cen, dtype=int, sep=' ')

    a = min(max(minindex[0] + deps/2, cen[0]), maxindex[0]- deps/2-1)
    b = min(max(minindex[1] + rows/2, cen[1]), maxindex[1]- rows/2-1)
    c = min(max(minindex[2] + cols/2, cen[2]), maxindex[2]- cols/2-1)

    # lizx:
    a_min=a-deps/2
    b_min=b - rows / 2
    c_min=c - cols / 2
    a_max=a + deps / 2
    b_max=b + rows / 2
    c_max=c + cols/2+1
    cropp_img=slice_array(img,a_min,a_max,b_min,b_max,c_min,c_max)
    cropp_tumor=slice_array(tumor,a_min,a_max,b_min,b_max,c_min,c_max)


    cropp_img -= MEAN
     # randomly flipping
    flip_num = np.random.randint(0, 8)
    if flip_num == 1:
        # problem
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
    elif flip_num == 2:
        # problem
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
    elif flip_num == 3:
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 4:
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 5:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 6:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 7:
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)

    cropp_tumor = resize(cropp_tumor, (args.input_size,args.input_size,args.input_cols), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    cropp_img   = resize(cropp_img, (args.input_size,args.input_size,args.input_cols), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    return cropp_img, cropp_tumor[:,:,1]

def generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list):
    while 1:
        X = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols), dtype='float32')
        Y = np.zeros((batch_size, args.input_size, args.input_size, 1), dtype='int16')
        Parameter_List = []
        for idx in xrange(batch_size):
            count = random.choice(trainidx)
            img = img_list[count]
            tumor = tumor_list[count]
            minindex = minindex_list[count]
            maxindex = maxindex_list[count]
            num = np.random.randint(0,6)
            if num < 3 or (count in liverlist):
                lines = liverlines[count]
                numid = liveridx[count]
            else:
                lines = tumorlines[count]
                numid = tumoridx[count]
            Parameter_List.append([img, tumor, lines, numid, minindex, maxindex])
        # TODO: uncomment
        # pool = ThreadPool(thread_num)
        # result_list = pool.map(load_seq_crop_data_masktumor_try, Parameter_List)
        # pool.close()
        # pool.join()
        result_list=[]
        for i in range(len(Parameter_List)):
            result_list.append(load_seq_crop_data_masktumor_try(Parameter_List[i]))
        for idx in xrange(len(result_list)):
            X[idx, :, :, :] = result_list[idx][0]
            Y[idx, :, :, 0] = result_list[idx][1]
        yield (X,Y)


def load_fast_files(args):

    trainidx = list(range(N_TRAINING_SAMPLES))
    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    liveridx = []
    liverlines = []
    for idx in xrange(N_TRAINING_SAMPLES):
        img, img_header = load(args.data+ '/myTrainingData/volume-' + str(idx) + '.nii')
        tumor, tumor_header = load(args.data + '/myTrainingData/segmentation-' + str(idx) + '.nii')
        img_list.append(img)
        tumor_list.append(tumor)
        maxmin = np.loadtxt(args.data + '/myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        minindex[0] = max(minindex[0] - 3, 0)
        minindex[1] = max(minindex[1] - 3, 0)
        minindex[2] = max(minindex[2] - 3, 0)
        maxindex[0] = min(img.shape[0], maxindex[0] + 3)
        maxindex[1] = min(img.shape[1], maxindex[1] + 3)
        maxindex[2] = min(img.shape[2], maxindex[2] + 3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)
        f1 = open(args.data + '/myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt', 'r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()
        f2 = open(args.data + '/myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt', 'r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()

    return trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list

def train_and_predict():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = DenseUNet(reduction=0.5)
    model.load_weights(args.model_weight, by_name=True)

    if args.n_gpus>1:
        model = make_parallel(model, args.n_gpus, mini_batch=max(args.batch_size/args.n_gpus,1))
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[weighted_crossentropy_2ddense])

    trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list = load_fast_files(args)

    print('-'*30)
    print('Fitting model......')
    print('-'*30)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(args.save_path + "/model"):
        os.mkdir(args.save_path + '/model')
        os.mkdir(args.save_path + '/history')
    else:
        if os.path.exists(args.save_path+ "/history/lossbatch.txt"):
            os.remove(args.save_path + '/history/lossbatch.txt')
        if os.path.exists(args.save_path + "/history/lossepoch.txt"):
            os.remove(args.save_path + '/history/lossepoch.txt')

    model_checkpoint = ModelCheckpoint(args.save_path + '/model/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose = 1,
                                       save_best_only=False,save_weights_only=False,mode = 'min', period = 1)
    # TODO: remove
    gen=generate_arrays_from_file(args.batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx,
                                                  liveridx, minindex_list, maxindex_list)
    data=(batch[0] for batch in [next(gen),next(gen),next(gen)])
    print(next(data))
    import ipdb; ipdb.set_trace()

    print(model.predict_generator(data,2,workers=0))
    model.fit_generator(generate_arrays_from_file(args.batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx,
                                                  liveridx, minindex_list, maxindex_list),steps_per_epoch=1, #lizx: lizx changed to 1
                                                    epochs= 1, verbose = 2, callbacks = [model_checkpoint], max_queue_size=10,
                                                    workers=3, use_multiprocessing=True)

    print ('Finised Training .......')
def main2():
    arr=np.arange(64).reshape(4,4,4)
    print(slice_array(arr,-1,3,0,3,3,5))
    print(slice_array(arr,-1,3,0,3,3,5).shape)

if __name__ == '__main__':
    train_and_predict()