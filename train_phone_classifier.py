import tensorflow as tf
import numpy as np
import math
import sys
import glob
import os
import time
#np.set_printoptions(threshold = 1e6)
from PhoneClassifer import phone_classifer
from utils.load_data import get_data,default_config,phns,get_data_onebyone
from LOAD_SAVE_PARAMS.LOAD_SAVE_PARAMS import save_weight,load_weight

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.compat.v1.keras.backend.clear_session()



os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
np.random.seed(1)
def train(model,params,batch_size):
    acc=[]
    wav_files_list = glob.glob(default_config["data_path_phcls"])
    #print(wav_files_list)
    iter_data=get_data_onebyone(wav_files_list,batch_size,False)#get_data(wav_files_list,batch_size,data_augment=False)#二分类开这个会无法收敛
    los=[]
    for x,y in iter_data:
        e_start=time.time()
        X=tf.concat(x,0)
        
        y=[tf.cast(y1,dtype=tf.float32) for y1 in y]

        y_hard=y.copy()
       
        y_hard=tf.one_hot(y_hard,int(len(phns)))
        
        Y_hard=tf.concat(y_hard,0)
        
        X=tf.cast(X,dtype=tf.float32)
        
        Y_hard=tf.cast(Y_hard,dtype=tf.float32)
       
        with tf.GradientTape() as tape:
            tape.watch(params)

            logits,ppgs,preds=model(X,dropout_rate=0.0)
            loss=model.loss_sigmoid(logits,Y_hard)
            
        print("loss:%f"%loss)
        los.append(loss)
        grads=tape.gradient(loss,params)
        #grads,_tf.clip_by_global_norm(grads,clip_norm)
        model.update_params(grads,params)
        ac=return_accuracy(Y_hard,ppgs)
        print("训练准确率：%f"%ac)

        acc.append(ac)
        e_end=time.time()
        print("一次耗时%f秒"%(e_end-e_start))
        
        
    filepath="512hd_acc.txt"
    flie=open(filepath,"a+")
    
    flie.write(str(tf.math.reduce_mean(acc).numpy())+"\n")
    flie.close()

    

    filepath="512hd_loss.txt"
    flie=open(filepath,"a+")
    
    flie.write(str(tf.math.reduce_mean(los).numpy())+"\n")
    flie.close()
    

    
def return_accuracy(Y,Y_pre):
    #print(Y.shape)
    #print(Y_pre.shape)
    num=Y.shape[0]*Y.shape[1]
    
    rowMaxSoft=np.argmax(Y_pre, axis=-1)+1
    rowMax=np.argmax(Y, axis=-1)+1
    
    rowMaxSoft=rowMaxSoft.reshape([1,-1])
    rowMax=rowMax.reshape([1,-1])
    
    nonO=rowMaxSoft-rowMax
   
    exist = (nonO != 0) * 1.0
    factor = np.ones([nonO.shape[1],1])
    res = np.dot(exist, factor)
   
    accuracy=(float(num)-res[0][0])/float(num)
    
    return accuracy
    
if __name__ == "__main__":
    
    batch_size=1

    input_nums=default_config["n_mfcc"]

    num_hiddens=512#default_config["hidden_units"]

    num_outputs=default_config["n_mfcc"]

    layer_nums=12

    multi_head=12
    
    max_position_dim=1024

    clip_norm=1.0
    #lr=1e-5#5e-6
    model=phone_classifer(lr=1e-5,input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,max_position_dim=max_position_dim,layers_encoder=layer_nums,labels_num=len(phns),multi_head=multi_head)

    params=model.get_params()+model.get_params_vc()
    
    #params=model.get_params_bert_position()+model.get_params_bert_layer()+model.get_params_vc()

    epochs=3000

    isContinue=True

    if isContinue==True:
        load_weight("ckp","params_pc",params)


    for i in range(epochs):
        with tf.device('/gpu:0'):
            train(model,params,batch_size)
            save_weight("ckp","params_pc",params)
