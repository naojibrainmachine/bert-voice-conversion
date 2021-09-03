import tensorflow as tf
from BERT_pre_post_fdfd import bert

class phone_classifer:
    def __init__(self,lr,input_nums,hidden_nums,output_nums,max_position_dim,layers_encoder,labels_num,multi_head):
        self.pc=bert(lr=lr,input_nums=input_nums,hidden_nums=hidden_nums,output_nums=output_nums,max_position_dim=max_position_dim,layers_encoder=layers_encoder,labels_num=labels_num,multi_head=multi_head)
        self.opt=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
        
    def __call__(self,x,dropout_rate):

        inputs=x
        
        logits,ppgs,preds=self.pc(inputs,dropout_rate)

        return logits,ppgs,preds

    def loss_ce(self,output,y):
        return -1*tf.reduce_mean(tf.multiply(tf.math.log(output+ 1e-10),y))
    
    def get_params(self):
        params=[]

        #params.extend(self.embed_token.get_params())

        params.extend(self.pc.get_params())
        
        #params.extend([inner_cell for cell in self.encoders for inner_cell in cell.get_params()])

        return params
    def get_params_vc(self):
        params=[]

        params.extend(self.pc.get_params_vc())

        return params

    def update_params(self,grads,params):
        self.opt.apply_gradients(grads_and_vars=zip(grads,params))
        
    def loss_sigmoid(self,output,y):#gamma=5
        return self.pc.loss_sigmoid(output,y)
