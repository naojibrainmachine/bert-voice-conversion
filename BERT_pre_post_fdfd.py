import tensorflow as tf
import numpy as np
import math
import random

from ENCODER import ENCODER_pre_post_fdfd# import encoder
from ENCODER import ENCODER# import encoder

from EMBEDDING.EMBEDDING import embedding
from ADD_NORM.ADD_NORM import add,layer_norm
class bert:
    def __init__(self,lr,input_nums,hidden_nums,output_nums,max_position_dim=512,layers_encoder=12,labels_num=2,multi_head=12):#input_num=768 

        self.input_nums=input_nums
        self.max_position_dim=max_position_dim#512

        self.encoders=[ENCODER_pre_post_fdfd.encoder(input_nums=input_nums,hidden_nums=hidden_nums,output_nums=output_nums,multi_head=multi_head)]+[ENCODER.encoder(input_nums=input_nums,hidden_nums=hidden_nums,output_nums=output_nums,multi_head=multi_head) for _ in range(layers_encoder-1)]
        
        self.embed_position=embedding(max_position_dim,input_nums)

        self.w_vc=tf.Variable(tf.random.truncated_normal([output_nums,labels_num],stddev=tf.math.sqrt(2.0/(output_nums+labels_num))))
        self.b_vc=tf.Variable(tf.zeros(labels_num))
    
        self.opt=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
        
    def __call__(self,x,dropout_rate=0.1):
        '''
        function:bert的主体操作
        parameter:
        '''
        embedd_x=x
        
       
        pos=range(x.shape[1])

        pos_one_hot=tf.one_hot(pos,self.max_position_dim)
        pos_one_hot=tf.reshape(pos_one_hot,[1,pos_one_hot.shape[0],pos_one_hot.shape[-1]])
        embedd_pos=self.embed_position.embedding_lookup(pos_one_hot)

        
        embedd_x=embedd_x+embedd_pos
        
        
        output_x=embedd_x
        
        
        temp=[]
        final_output=output_x
        for i in range(len(self.encoders)):
           
            output_x_2,_,_=self.encoders[i](output_x,dropout_rate)
            output_x=output_x_2+output_x
            output_x=layer_norm(output_x)#这里有ln代表，到最后所有的输出相加之前就已经经历了无数个ln，有嵌套的ln

        
        
        logits=tf.matmul(output_x,self.w_vc)+self.b_vc

        ppgs=tf.nn.softmax(logits)
        #print(ppgs)
        #print(tf.math.reduce_sum(ppgs[:,0,:]))
        preds = tf.cast(tf.argmax(logits, axis=-1),dtype=tf.int32)
        
        return logits,ppgs,preds#tf.concat([mid_result,output_x],-1)#把最后结果和中间结果拼接
    
            
        
   
    def loss_sigmoid(self,output,y):#gamma=5
        
        return -1.*tf.reduce_mean(tf.multiply(tf.math.log(tf.math.sigmoid(output+ 1e-10)),y)+tf.reduce_mean(tf.multiply(tf.math.log((1-tf.math.sigmoid(output+ 1e-10))),(1-y))))

    def loss(self,output,y,gamma=3):#gamma=5
        
        return -1*tf.reduce_mean(tf.multiply(tf.multiply(tf.math.log(output+ 1e-10),y),tf.math.pow((1-output),gamma)))#Focal loss
    def loss_ce(self,output,y):
        return -1*tf.reduce_mean(tf.multiply(tf.math.log(output+ 1e-10),y))
    def am_softmax_loss(self,output,y,margin=0.35,scale=30):
        
        y_pred = (y * (output - margin) + (1 - y) * output) * scale
        y_pred=tf.nn.softmax(y_pred,axis=-1)
        #print(tf.reduce_sum(tf.nn.softmax(y_pred,axis=-1)[0,0,:]))
        
        return self.loss_cr(y_pred,y)

    def get_params_vc(self):
        params=[]

        params.append(self.w_vc)

        params.append(self.b_vc)


        return params

    
    def get_params_bert_layer(self):
        params=[]

        #params.extend(self.embed_token.get_params())

        #params.extend(self.embed_position.get_params())
        
        params.extend([inner_cell for cell in self.encoders for inner_cell in cell.get_params()])

        return params
    
    def get_params_bert_position(self):
        params=[]

        params.extend(self.embed_position.get_params())
        
        return params
    
    def get_params(self):
        params=[]

        #params.extend(self.embed_token.get_params())

        params.extend(self.embed_position.get_params())
        
        params.extend([inner_cell for cell in self.encoders for inner_cell in cell.get_params()])

        return params
    
    def update_params(self,grads,params):
        self.opt.apply_gradients(grads_and_vars=zip(grads,params))
        
    def gelu(self,input_tensor):
        #cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.math.sqrt(2.0)))

        cdf2 = 0.5 * (1.0 + tf.nn.tanh((input_tensor+0.044715*tf.math.pow(input_tensor,3))*tf.math.sqrt(2.0/math.pi)))
        
        return input_tensor*cdf2

    


