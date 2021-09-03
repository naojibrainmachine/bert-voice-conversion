import tensorflow as tf
import numpy as np
import math
from BERT import bert
from utils.load_data import get_data,phns
class speech_synthesizer:
    def __init__(self,lr,mel_in_nums,mel_hi_nums,mel_out_nums,mel_layers,n_mel,spec_in_nums,spec_hi_nums,spec_out_nums,spec_layers,n_spec,max_position_dim=512,multi_head=12):

        self.pc=None
        
        self.ss_mel=bert(lr=0,input_nums=mel_in_nums,hidden_nums=mel_hi_nums,output_nums=mel_out_nums,layers_encoder=mel_layers,labels_num=n_mel,max_position_dim=max_position_dim,multi_head=multi_head)

        self.ss_spec=bert(lr=0,input_nums=n_mel,hidden_nums=spec_hi_nums,output_nums=spec_out_nums,layers_encoder=spec_layers,labels_num=n_spec,max_position_dim=max_position_dim,multi_head=multi_head)

        
        self.opt=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
        
    def __call__(self,x_mfccs,dropout_rate):
        
        if self.pc==None:
            raise ("音素分类模型未初始化")

        
        _,ppgs,_=self.pc(x_mfccs,dropout_rate)#位置信息这里已经传入
        '''
        test_iter=get_data(["SA1.WAV","SA1.WAV"],1,False)
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
        for x,y in test_iter:
           
            print("aaa")
            
            
            X=tf.concat(x,0)
            
            y=[tf.cast(y1,dtype=tf.float32) for y1 in y]

            y_hard=y.copy()
            
            #y_soft=tf.one_hot(y,int(len(phns)),on_value=confidence,off_value=low_confidence)

            y_hard=tf.one_hot(y_hard,int(len(phns)))
            
            #Y_soft=tf.concat(y_soft,0)
            Y_hard=tf.concat(y_hard,0)
            
            X=tf.cast(X,dtype=tf.float32)
            #Y_soft=tf.cast(Y_soft,dtype=tf.float32)
           
            Y_hard=tf.cast(Y_hard,dtype=tf.float32)
            
            
           
            print("aaa")
            logits,ppgs,preds=self.pc(X,dropout_rate=0.0)
            loss=self.pc.loss_sigmoid(logits,Y_hard)
            print("loss:%f"%loss)
            #los.append(loss)
            ac=return_accuracy(Y_hard,ppgs)
            print("训练准确率：%f"%ac)
            sys.exit(0)
        '''
        pred_mel,_,_=self.ss_mel(ppgs)#这些就没必要再加入位置的Embedding了
        #print(pred_mel.shape)
        pred_spec,_,_=self.ss_spec(pred_mel)#这些就没必要再加入位置的Embedding了
        #print(pred_spec.shape)
        return pred_mel,pred_spec
        

    def init_pc(self,pc):

        self.pc=pc
        
    def predict(self,x_mfccs,dropout_rate):

        _,ppgs,_=self.pc(x_mfccs,dropout_rate)
        
        pred_mel,_,_=self.ss_mel(ppgs,dropout_rate)
        
        pred_spec,_,_=self.ss_spec(pred_mel,dropout_rate)

        return pred_spec,ppgs
    
    def loss_ss(self,pre_mel,true_mel,pre_spec,true_spec):
        
        loss_mel=tf.reduce_mean(tf.math.square(pre_mel-true_mel))
        loss_spec=tf.reduce_mean(tf.math.square(pre_spec-true_spec))

        loss=loss_mel+loss_spec

        return loss
    
    def get_params_mel(self):
        params=[]

        params.extend(self.ss_mel.get_params())

        params.extend(self.ss_mel.get_params_vc())

        return params
    def get_params_spec(self):
        
        params=[]

        params.extend(self.ss_spec.get_params())

        params.extend(self.ss_spec.get_params_vc())

        return params

    def get_params_pc(self):

        if self.pc==None:
            raise ("音素分类模型未初始化")
    
        params=[]

        params.extend(self.pc.get_params())

        params.extend(self.pc.get_params_vc())
       
        return params
    

    def update_params(self,grads,params):
        self.opt.apply_gradients(grads_and_vars=zip(grads,params))

    
    def circle_loss(self,hidden_representation,hidden_representions_t,label,m=0.25,gamma=256):
        
        if len(label.shape)==1:
            label=np.expand_dims(label,0)
        
        return self.ss_mel.circle_loss(hidden_representation,hidden_representions_t,label,m=m,gamma=gamma)
