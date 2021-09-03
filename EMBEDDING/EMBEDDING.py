#构建可训练的分布式词向量
import tensorflow as tf
import numpy as np
import math
class embedding:
    def __init__(self,vocabulary_size,embedding_size):
        '''
        构建embedding层
        vocabulary_size：为词库大小
        embedding_size：分布式词向量大小
        '''
        self.vocabulary_size=vocabulary_size
        self.embedding_size=embedding_size
        self.embeddings=tf.Variable(tf.random.uniform([self.vocabulary_size,self.embedding_size],-1.0,1.0))
        
    def embedding_lookup(self,oneHot_data):
        '''
        embedding的查找表函数
        oneHot_data：词库大小的one_hot向量
        '''
        outputs_i=[]
        #外层训练是遍历一个批次的所有语句，内层训练是遍历语句的每个one_hot数据
        for i in range(oneHot_data.shape[0]):
            outputs_j=[]
            for j in range(oneHot_data.shape[1]):
                if(tf.reduce_sum(oneHot_data[i,j,:]).numpy()==0.0):
                    #遇到-填充则返回全零的词向量
                    outputs_j.append(tf.zeros([1,self.embeddings.shape[-1]]))
                else:
                    #以查找表方式提取对应行的词向量
                    row = tf.math.argmax(oneHot_data[i,j,:],output_type=tf.dtypes.int32).numpy()
                    outputs_j.append(tf.reshape(self.embeddings[row,:],[1,self.embeddings.shape[-1]]))

            array_outputs_j=tf.concat(outputs_j,0)
            array_outputs_j=tf.reshape(array_outputs_j,shape=[1]+(list(tf.concat(outputs_j,0).shape)))
            
            outputs_i.append(tf.concat(array_outputs_j,0))
        return tf.concat(outputs_i,0)
    def get_params(self):
        return [self.embeddings]
    def positional_encoding(self,pos,d):
        def w_k(pos,k, d):
            wk = 1.0/(10000**(((2*(k//2))/(float(d)))))
            return tf.matmul(pos , wk)
        pe=tf.zeros([pos,d])
        wk = w_k(tf.reshape(tf.constant(np.arange(pos),dtype=tf.float32),[-1,1]),tf.reshape(tf.constant(np.arange(d),dtype=tf.float32),[1,-1]),d)
        tmp=wk.numpy()
        tmp[:,0::2]=np.sin(tmp[:,0::2])
        tmp[:,1::2]=np.sin(tmp[:,1::2])
        return tf.reshape(tmp,[1,tmp.shape[0],tmp.shape[1]])



'''
def embedding_lookup(self,oneHot_data):
        
        outputs_i=[]
        ##外层训练是遍历一个批次的所有语句，内层训练是遍历语句的每个one_hot数据
        #for i in range(oneHot_data.shape[0]):
        outputs_j=[]
        for j in range(oneHot_data.shape[0]):
            if(tf.reduce_sum(oneHot_data[j,:]).numpy()==0.0):
                #遇到-填充则返回全零的词向量
                outputs_j.append(tf.zeros([1,self.embeddings.shape[-1]]))
            else:
                #以查找表方式提取对应行的词向量
                row = tf.math.argmax(oneHot_data[j,:],output_type=tf.dtypes.int32).numpy()
                outputs_j.append(tf.reshape(self.embeddings[row,:],[1,self.embeddings.shape[-1]]))

        array_outputs_j=tf.concat(outputs_j,0)
        #print(array_outputs_j,"array_outputs_j,array_outputs_j")
        #array_outputs_j=tf.reshape(array_outputs_j,shape=[1]+(list(tf.concat(outputs_j,0).shape)))
        print(array_outputs_j,"array_outputs_j")
        #outputs_i.append(tf.concat(array_outputs_j,0))
        return array_outputs_j
    def get_params(self):
        return [self.embeddings]
'''
