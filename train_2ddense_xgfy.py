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
import glob
import scipy
K.set_image_dim_ordering('tf')

#  global parameters
parser = argparse.ArgumentParser(description='Keras 2d denseunet Training')
#  data folder
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
def load_seq_crop_data_masktumor_evaluate(img,index):
    assert 0<=index<img.shape[2]

    #  randomly scale
    deps = args.input_size
    rows = args.input_size
    cols = 3

    # lizx:
    a_min=0
    b_min=0
    c_min=index - cols / 2
    a_max=img.shape[0]
    b_max=img.shape[1]
    c_max=index + cols/2+1

    cropp_img = slice_array(img,a_min,a_max,b_min,b_max,c_min,c_max).copy()
    cropp_img -= MEAN
    return cropp_img
def get_prediction(model,img,batch_size=args.batch_size):
    pred=np.zeros_like(img)
    for i in range(0,img.shape[2],batch_size):
        print("[%d/%d]"%(i+1,img.shape[2]))
        inp=np.zeros((batch_size,img.shape[0],img.shape[1],3))
        for j in range(batch_size):
            if i+j<img.shape[2]:
                inp[j,:,:,:]=load_seq_crop_data_masktumor_evaluate(img,i+j)
            else:
                inp[j,:,:,:]=np.zeros((img.shape[0],img.shape[1],3))
        prediction=model.predict(inp,batch_size=batch_size)
        prediction_seg=prediction.argmax(axis=3)
        prediction_mask=(prediction_seg==2).astype(np.float32)
        pred[:,:,i:(min(i+batch_size,img.shape[2]))]=prediction_mask[0:(min(batch_size,img.shape[2]-i)),:,:].transpose((1,2,0))
    return pred
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
            if num < 3 : # removed liver_list
                lines = liverlines[count]
                numid = liveridx[count]
            else:
                lines = tumorlines[count]
                numid = tumoridx[count]
            Parameter_List.append([img, tumor, lines, numid, minindex, maxindex])

        # pool = ThreadPool(thread_num)
        result_list = map(load_seq_crop_data_masktumor_try, Parameter_List)
        # pool.close()
        # pool.join()
        for idx in xrange(len(result_list)):
            X[idx, :, :, :] = result_list[idx][0]
            Y[idx, :, :, 0] = result_list[idx][1]
        print("here")
        yield (X,Y)

def load_fast_files(args):
    data_dir="/ibex/scratch/projects/c2052/COVID-19/arrays_raw"
    lung_mask_dir="/ibex/scratch/projects/c2052/COVID-19/arrays_raw_lung_seg"
    train_dir="xgfy_data/myTrainingData"
    test_dir= "xgfy_data/myTestData"
    train_txt_dir="xgfy_data/myTrainingDataTxt"
    lung_mask_txt_dir="xgfy_data/myTrainingDataTxt/lung_mask"
    label_txt_dir="xgfy_data/myTrainingDataTxt/label"
    box_txt_dir="xgfy_data/myTrainingDataTxt/box"

    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    liveridx = []
    liverlines = []
    volume_paths=glob.glob(os.path.join(train_dir,"*_volume.npy"))
    # TODO: remove
    volume_paths=volume_paths
    trainidx = list(range(len(volume_paths)))

    for i,volume_path in enumerate(volume_paths):
        print("[%d/%d]\r"%(i+1,len(volume_paths)))
        noext=os.path.splitext(os.path.basename(volume_path))[0]
        fid=noext.replace('_volume','')
        img=np.load(volume_path,mmap_mode='r')
        tumor=np.load(volume_path.replace('_volume','_segmentation-and-lung-mask'),mmap_mode='r')
        img_list.append(img)
        tumor_list.append(tumor)
        maxmin = np.loadtxt(os.path.join(box_txt_dir,fid+".txt"))
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
        f1 = open(os.path.join(label_txt_dir,fid+".txt"), 'r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()
        f2 = open(os.path.join(lung_mask_txt_dir,fid+".txt"), 'r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()

    return trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list

def f1_score_evaluation(pred,gt, lung_mask, filter_ground_truth=True):  # 2.97 is tested as close to opitmal on test

    if filter_ground_truth:
        gt=scipy.ndimage.maximum_filter(gt,3)

    Precision, Recall, F1_score = calculate_f1_score_cpu(pred, gt)
    print('\tcombined: Precision, Recall, F1_score', Precision, Recall, F1_score)
    eps=1e-6
    pred_percentage=np.sum(pred*lung_mask)/(np.sum(lung_mask)+eps)
    gt_percentage=np.sum(gt*lung_mask)/(np.sum(lung_mask)+eps)
    result = dict(Precision=Precision, Recall=Recall, F1_score=F1_score,
    pred_percentage=pred_percentage,gt_percentage=gt_percentage)
    for k,v in result.items():
        if np.isnan(v) or np.isinf(v):
            result[k]=0
    return result

def calculate_f1_score_cpu(prediction, ground_truth, strict=False):
    # this is the patient level acc function.
    # the input for prediction and ground_truth must be the same, and the shape should be [height, width, thickness]
    # the ground_truth should in range 0 and 1
    if np.max(prediction) > 1 or np.min(prediction) < 0:
        print("prediction is not probabilistic distribution")
        exit(0)

    if not strict:
        ground_truth = np.array(ground_truth > 0, 'float32')

        TP = np.sum(prediction * ground_truth)
        FN = np.sum((1 - prediction) * ground_truth)
        FP = np.sum(prediction) - TP
    else:
        difference = prediction - ground_truth

        TP = np.sum(prediction) - np.sum(np.array(difference > 0, 'float32') * difference)
        FN = np.sum(np.array(-difference > 0, 'float32') * (-difference))
        FP = np.sum(np.array(difference > 0, 'float32') * difference)
    eps=1e-6
    F1_score = (2*TP+eps)/(FN+FP+2*TP+eps)
    Precision=(TP+eps)/(TP+FP+eps)
    Recall=(TP+eps)/(TP+FN+eps)
    return Precision, Recall, F1_score

def train_and_predict():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = DenseUNet(reduction=0.5)
    import ipdb; ipdb.set_trace()
    model.load_weights(args.model_weight, by_name=True)

    if args.n_gpus>1:
        print("Using %d GPUs"%(args.n_gpus))
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


    model.fit_generator(generate_arrays_from_file(args.batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx,
                                                  liveridx, minindex_list, maxindex_list),steps_per_epoch=20, #lizx: lizx changed to 1
                                                    epochs= 50000, verbose = 2, callbacks = [model_checkpoint], max_queue_size=10,
                                                    workers=1, use_multiprocessing=False)

    print ('Finised Training .......')

if __name__ == '__main__':
    train_and_predict()