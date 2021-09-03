#transformer的encoder
import tensorflow as tf
from ATTENTION.ATTENTION import self_attention
from ADD_NORM.ADD_NORM import layer_norm,add
from FEED_FORWARD.FEED_FORWARD import feed_forward
class encoder:
    def __init__(self,input_nums,hidden_nums,output_nums,mask=False,multi_head=1):
        self.self_atten=self_attention(input_nums,hidden_nums,output_nums,mask,multi_head)
        self.fdfd=feed_forward(hidden_nums,output_nums)

    def __call__(self,x,dropout_rate=0.05):
        #print(x,"1x.shape encoder")
        #print(x.shape,"x.shape")
        z_output,K,V=self.self_atten(x)

        z_output=tf.nn.dropout(z_output,dropout_rate)#0.2dropout
        #print(z_output.shape,"z_output.shape encoder")
        #print(z_output,"1z_output.shape encoder")
        z_output=add(z_output,x)
        #print(z_output,"2z_output.shape encoder")
        z_output_self_attentiont=layer_norm(z_output)
        
        
        
        #print(z_output_self_attentiont,"z_output_self_attentiont encoder")
        z_output=self.fdfd(z_output_self_attentiont)

        z_output=tf.nn.dropout(z_output,dropout_rate)#0.2dropout
        #print(z_output,"fdfd encoder")
        z_output=add(z_output_self_attentiont,z_output)
        #print(z_output,"fdfd add encoder")
        z_output_fdfd=layer_norm(z_output)
        #print(z_output_fdfd,"z_output_fdfd")
        
        
        return z_output_fdfd,K,V
    def get_params(self):
        params=[]

        params.extend(self.self_atten.get_params())
        params.extend(self.fdfd.get_params())

        return params

    def get_params_last_layer(self):
        params=[]

        params.extend(self.self_atten.get_k_v_params())
        #params.extend(self.fdfd.get_params())

        return params
    
    '''
    def __init__(self,input_nums,hidden_nums,output_nums,multi_head=1):
        
        #function:
        #    初始化encoder
        #parameter:
        #    input_nums:一般是词向量大小
        
        self.multi_headed=multi_headed
        self.input_nums=input_nums
        
        self.variables={}
        for i in range(multi_headed):
            q_name="q"+str(i)
            k_name="k"+str(i)
            v_name="v"+str(i)
            self.variables[q_name]=tf.Variable(tf.random.truncated_normal([input_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums+hidden_nums))))

            self.variables[k_name]=tf.Variable(tf.random.truncated_normal([input_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums+hidden_nums))))

            self.variables[v_name]=tf.Variable(tf.random.truncated_normal([input_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums+hidden_nums))))


        #self.variables["w0"]=self.w0
        self.w0=tf.Variable(tf.random.truncated_normal([multi_headed*hidden_nums,output_nums],stddev=tf.math.sqrt(2.0/(multi_headed*hidden_nums,output_nums))))

        self.w1=tf.Variable(tf.random.truncated_normal([output_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums,output_nums))))
        self.b1=tf.Variable(tf.random.truncated_normal([hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums,output_nums))))

        self.w2=tf.Variable(tf.random.truncated_normal([hidden_nums,output_nums],stddev=tf.math.sqrt(2.0/(input_nums,output_nums))))
        self.b2=tf.Variable(tf.zeros([output_nums],stddev=tf.math.sqrt(2.0/(input_nums,output_nums))))

    def __call__(self,x):


        #multi_headed attention
        Zs=[]
        for i in range(len(self.variables)):
            q_name="q"+str(i)
            k_name="k"+str(i)
            v_name="v"+str(i)
            Q=tf.matmul(x,self.variables[q_name])
            K=tf.matmul(x,self.variables[k_name])
            V=tf.matmul(x,self.variables[v_name])

            score=tf.matmul(Q,tf.transpose(K,[1,0]))

            score=score/tf.math.sqrt(self.input_nums)

            score=tf.nn.softmax(score,axis=1)

            z=tf.matmul(score,V)

            Zs.append(z)

        z_output=tf.concat(Zs,1)

        z_output=tf.matmul(z_output,self.w0)

        #add
        z_output=z_output+x# Residuals
        
        #norm
        z_output_self_attentiont=self.norm(z_output)
        #temp_z_output=self.norm(z_output)
        
        #mean=tf.reduce_mean(z_output)
        #variance=tf.reduce_mean(z_output-mean)
        #temp_z_output=(z_output-mean)/(tf.math.sqrt(variance)+1e-3)
        
        #feed forward
        z_output=tf.matmul(z_output_self_attentiont,self.w1)+self.b1
        z_output=tf.nn.relu(z_output)
        z_output=tf.matmul(z_output,self.w2)+self.b2

        #add
        z_output=z_output+z_output_self_attentiont

        #norm
        z_output=self.norm(z_output)

        
        #mean=tf.reduce_mean(z_output)
        #variance=tf.reduce_mean(z_output-mean)
        #z_output=(z_output-mean)/(tf.math.sqrt(variance)+1e-3)
        

        return z_output,z_output_self_attentiont
    
    def get_params(self):
        params=list(self.variables.values())+[self.w0,self.w1,self.b1,self.w2,self.b2]
        #for i in range(self.multi_headed):
            #params.append(tf.get_variable(i))
        return params
    
    def norm(self,z):
        
        mean=tf.reduce_mean(z)
        variance=tf.reduce_mean(z-mean)
        output=(z-mean)/(tf.math.sqrt(variance)+1e-3)
        
        return output
    '''    
