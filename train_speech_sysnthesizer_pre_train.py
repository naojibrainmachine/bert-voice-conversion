import tensorflow as tf
import numpy as np
import math
import sys
import glob
import os
import time
np.set_printoptions(threshold = 1e6)
from SpeechSynthesizer import speech_synthesizer
from PhoneClassifer import phone_classifer
from utils.load_data import get_data_ss,default_config,phns
from LOAD_SAVE_PARAMS.LOAD_SAVE_PARAMS import save_weight,load_weight

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.compat.v1.keras.backend.clear_session()



os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

def train(model,params,batch_size):
    wav_files_list = glob.glob(default_config["ss_pre_train"])
    print(len(wav_files_list))
    iter_data=get_data_ss(wav_files_list,batch_size,random_crop=True,bReSample=True)
    los=[]
    count=1
    for x_mfccs,y_spec,y_mel in iter_data:
        e_start=time.time()
        
        X_mfccs=tf.concat(x_mfccs,0)
        
        Y_spec=tf.concat(y_spec,0)
        
        Y_mel=tf.concat(y_mel,0)
        
        X_mfccs=tf.cast(X_mfccs,dtype=tf.float32)
        
        Y_spec=tf.cast(Y_spec,dtype=tf.float32)

        Y_mel=tf.cast(Y_mel,dtype=tf.float32)
        
    
        with tf.GradientTape() as tape:
            tape.watch(params)
            
            pred_mel,pred_spec=model(X_mfccs,dropout_rate=0.0)
            
            loss=model.loss_ss(pred_mel,Y_mel,pred_spec,Y_spec)

        print("loss:%f"%loss)
        los.append(loss)
        grads=tape.gradient(loss,params)
        #grads,_tf.clip_by_global_norm(grads,clip_norm)
        model.update_params(grads,params)
        e_end=time.time()
        print("一次耗时%f秒"%(e_end-e_start))
        if count%1000==0:
            save_weight("ckp","params_ss_pre_train",params)
            
            filepath="loss_ss_pre_train.txt"
            flie=open(filepath,"a+")
            flie.write(str(tf.math.reduce_mean(los).numpy())+"\n")
            flie.close()
            
            count=count+1
        else:
            count=count+1
        
    
    filepath="loss_ss_pre_train.txt"
    flie=open(filepath,"a+")
    
    flie.write(str(tf.math.reduce_mean(los).numpy())+"\n")
    flie.close()
    

    
def return_accuracy(Y,Y_pre):
    #print(Y.shape)
    #print(Y_pre.shape)
    #print(Y)
    #print(Y_pre)
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

    n_mel=default_config["n_mels"]

    n_spec=(default_config["n_fft"]//2)+1

    input_nums=default_config["n_mfcc"]

    num_hiddens=512#default_config["hidden_units"]#768#768电脑带不动

    num_outputs=default_config["n_mfcc"]

    layer_nums=12

    multi_head=12

    mel_layers=6

    spec_layers=6
    
    max_position_dim=1024

    clip_norm=1.0

    epochs=3000

    isContinue=True
    
    #lr=1e-5#5e-6
    model_pc=phone_classifer(lr=1e-5,input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,max_position_dim=max_position_dim,layers_encoder=layer_nums,labels_num=len(phns),multi_head=multi_head)

    params_pc=model_pc.get_params()+model_pc.get_params_vc()

    model_ss=speech_synthesizer(lr=1e-5,mel_in_nums=len(phns),mel_hi_nums=num_hiddens,mel_out_nums=len(phns),mel_layers=mel_layers,n_mel=n_mel,spec_in_nums=n_mel,spec_hi_nums=num_hiddens,spec_out_nums=n_mel,spec_layers=spec_layers,n_spec=n_spec,max_position_dim=max_position_dim,multi_head=multi_head)

    
    
    params_mel=model_ss.get_params_mel()
    
    params_spec=model_ss.get_params_spec()

    params_ss=params_mel+params_spec

    try:
        load_weight("ckp","params_pc",params_pc)
    except:
        raise("未发现已经训练好的音素分类模型数据")

    model_ss.init_pc(model_pc)#加载pc模型

    if isContinue==True:
        try:
            load_weight("ckp","params_ss",params_ss)
            
        except:
            raise("未发现已经训练好的语音合成模型数据")
 
    for i in range(epochs):
        with tf.device('/gpu:0'):
            train(model_ss,params_ss,batch_size)
            save_weight("ckp","params_ss_pre_train",params_ss)
