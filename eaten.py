import os
import math
import json
import tempfile
import numpy as np
from tqdm import tqdm
from time import  time
import tensorflow as tf
from matplotlib import pyplot

def CNN_ENCODER(H,W,C):

    """
       Instantiates Inception v3 architecture and reshape the encoded image
       Args:
         H: int, indicating the height of the image
         W: int, indicating the widht of the image
         C: int, indicating the channels

    """
    inc_model = tf.keras.applications.InceptionV3(input_shape=(H,W,C),include_top=False)
    reshape = tf.keras.layers.Reshape((-1,2048))
    reshape_out = reshape(inc_model.output) #reshape F to (HxW,C)
    out = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(2048, use_bias=False))(reshape_out) #linear projection of encoded image
    model = tf.keras.Model(inputs=inc_model.input, outputs=out)
    return model

   



class ENTITY_DECODER(tf.keras.Model):
    """
       Builds LSTM based decoder network
       Args:

    """

    def __init__(self,vocab_size,
                 channel,emb_dim=64):
        super(ENTITY_DECODER,self).__init__()
        self.latent_dims = 128
        self.vocab_size = vocab_size
        self.channel = channel
        self.emb_dim = emb_dim

        self.Wc = tf.keras.layers.Dense(self.emb_dim, use_bias=False,name='Wc') #shape(vocab_size+1, emd_dim)
        self.Wct1 = tf.keras.layers.Dense(self.emb_dim, use_bias=False,name="Wct1") #shape(channels, emd_dim)
        self.Wo = tf.keras.layers.Dense(self.vocab_size,use_bias=False,name="Wo") #shape(latent_dims, vocab_size)
        self.Wct2 = tf.keras.layers.Dense(self.vocab_size,use_bias=False,name="Wct2")  #shape(channels, vocab_size)
        self.lstm = tf.keras.layers.LSTMCell(self.latent_dims,use_bias=False,name="lstm")
        
        w_init = tf.random_normal_initializer()
        self.W_h = tf.keras.layers.Dense(self.channel, use_bias=False,name="W_h")
        self.V = tf.Variable(initial_value=w_init(shape=(1,1,self.channel),dtype="float32"),
                         trainable=True,name="V")
        
    def get_config(self):
      return {"latent_dims":self.latent_dims,
              "vocab_size":self.vocab_size,
              "channel":self.channel,
              "emb_dim":self.emb_dim}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    def call(self,inputs,training=True):

       
       
       prev_cont_vec = inputs[0]
       
       F = inputs[1]
       prev_char = inputs[2]

       lstm_inp = self.Wc(prev_char) + self.Wct1(prev_cont_vec) #lstm_inp shape is (batch_size,emb_dim)
       states = (inputs[3],inputs[4])
       state_h, state_c = self.lstm(lstm_inp,states)
       state_c = state_c[1]
       if training:

            state_c = tf.clip_by_value(state_c, -10,10)

       # compute contextual feature

       H_tfm = tf.expand_dims(self.W_h(state_h),axis=1)
       e_t = tf.multiply(self.V,tf.math.tanh(
                 tf.math.add(H_tfm,F)))
       c_t = tf.reduce_sum(tf.multiply(
                       tf.nn.softmax(e_t,axis=1),F),axis=1)

       O_t = self.Wo(state_h) + self.Wct2(c_t)

       O_t = tf.nn.softmax(O_t,axis=1)
       return O_t,c_t, state_h,state_c
    
    
def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.00001)):
     #refrence:https://sthalles.github.io/keras-regularizer/
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


@tf.function
def train_step(cnn_encoder,ent_dict,optimizer,loss_obj,
              channels,batch_data, ent_max_time,ent_vocab_size,
              batch_size,ent_num,latent_dims=128):

    NORM = 2.0

    loss = 0
    true_predicted_count = 0

    #initial states of lstm
    state_h = tf.random.uniform(shape=(batch_size,latent_dims), minval = -10, maxval= 10,seed=42)
    state_c = tf.random.uniform(shape=(batch_size,latent_dims), minval = -10, maxval= 10,seed=42)
    c_t = tf.random.normal(shape=(batch_size,channels),seed=42) #initial context vector
    states = (state_h,state_c)
    O_t=None
    prev_char_one_hot = None
    

    with tf.GradientTape() as tape:
        F = cnn_encoder(batch_data[0])
        #iterate through each entity
        for ent_idx in range(1,ent_num+1):
            ent_data = batch_data[ent_idx]

            true_ans = []
            pred_ans = []
            #each time step in the entity
            for time_step in range(ent_max_time[ent_idx-1]):
                true = tf.reshape(ent_data[:,time_step],(-1,1))
                true_ans.append(true)
                if time_step == 0:
                    inputs = [c_t,F,tf.zeros((batch_size,ent_dict[f"ent_{ent_idx-1}"].vocab_size)),state_h,state_c]
                    O_t,c_t, state_h,state_c= ent_dict[f"ent_{ent_idx-1}"](inputs)
                else:
                    inputs = [c_t,F,prev_char_one_hot,state_h,state_c]
                    O_t,c_t,state_h,state_c = ent_dict[f"ent_{ent_idx-1}"](inputs)


                pred = tf.cast(
                    tf.reshape(tf.argmax(O_t,axis=1),shape=(-1,1)),tf.int32)

                prev_char_one_hot = tf.squeeze(tf.one_hot(pred,depth = ent_dict[f"ent_{ent_idx-1}"].vocab_size,axis=-1))

                loss += loss_obj(true,O_t)
                pred_ans.append(pred)



            true_ans = tf.stack(true_ans,axis=1)
            pred_ans = tf.stack(pred_ans,axis=1)
            true_count = tf.cast(tf.math.equal(true_ans,pred_ans),tf.int16)
            true_predicted_count += int(tf.math.reduce_sum(true_count))
            del true_ans, pred_ans



    trainable_var = cnn_encoder.trainable_variables
    for ent_idx in range(ent_num):
          trainable_var += ent_dict[f"ent_{ent_idx}"].trainable_variables
    model_gradients = tape.gradient(loss, trainable_var)
    grads = [tf.clip_by_norm(g, NORM)
            for g in model_gradients]
    optimizer.apply_gradients(zip(grads, trainable_var))

    return loss, true_predicted_count


def fit(train_ds,ent_vocab_size,epochs,ent_max_time,H=299,W=299,C=3):
    cnn_encoder = CNN_ENCODER(H,W,C)
    cnn_encoder = add_regularization(cnn_encoder)
    epoch_loss = []
    ent_num = len(ent_vocab_size)
    ent_dict = {}
    for i in range(ent_num):
      

      ent_dict[f"ent_{i}"] = ENTITY_DECODER(ent_vocab_size[i]+1,
                                             channel=2048)
      
      #create temperaroy inputs of entity model to be built
      state_h = tf.random.uniform(shape=(32,ent_dict[f"ent_{i}"].latent_dims), minval = -10, maxval= 10,seed=42)
      state_c = tf.random.uniform(shape=(32,ent_dict[f"ent_{i}"].latent_dims), minval = -10, maxval= 10,seed=42)
      c_t = tf.random.normal(shape=(32,ent_dict[f"ent_{i}"].channel)) #initial context vector
      prev_char_one_hot=tf.zeros((32,ent_dict[f"ent_{i}"].vocab_size))
      F = tf.random.normal(shape=(32,64,2048))
      inputs = [c_t,F,prev_char_one_hot,state_h,state_c]

        
      ent_dict[f"ent_{i}"](inputs,training=False)


        

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00004, momentum=0.2)
    for epoch in range(epochs):
        batch_count = 0
        tot_loss = 0
        tot_pos = 0
        if epoch%5 == 0 and epochs != 0:
            optimizer.lr = 0.00004 * (math.pow(0.94,epoch))
        with tqdm(train_ds,unit=" batch") as tbatch:
            for batch_data in tbatch:
              
                batch_size = int(batch_data[0].shape[0])
                batch_loss, batch_true_count  = train_step(cnn_encoder,ent_dict,optimizer,loss_obj,
                                                           2048,batch_data, ent_max_time,ent_vocab_size,
                                                           batch_size,ent_num)
                tbatch.set_description(f"Epochs: {epoch}, loss: {batch_loss/batch_size} entity_acc: {batch_true_count/(batch_size*ent_num)}")
                batch_count += 1
                tot_pos += batch_true_count
                tot_loss += batch_loss
        print(f"After {epoch} loss: {tot_loss/30000}, accuracy: {tot_pos/30000}")
        

        #save model for very 2 epochs
        if epoch%2==0:
          cnn_encoder.save("/root/cnn_encoder")
          for i in ent_dict:
            ent_dict[i].save(f"/root/entity_decoders/{i}")
        
    cnn_encoder.save("/root/cnn_encoder")
    for i in ent_dict:
      ent_dict[i].save(f"/root/entity_decoders/{i}")
    
                
    

def read_ent(path):
  ent = []
  with open(path) as f:
    for line in f:
      ent.append(line.strip("\n"))
  return ent


def ent_vectorise(ent,name):
  text = []
  text.extend(ent)

  ent_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',
                                                        split=" ",
                                                        lower=False)
  ent_tokenizer.fit_on_texts(text)
  tokenizer_json = ent_tokenizer.to_json()
  with open(f'{name}_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

  train_ent_seq = ent_tokenizer.texts_to_sequences(ent)
  train_ent_vect = tf.keras.preprocessing.sequence.pad_sequences(train_ent_seq,
                                                           padding="post")



  return len(ent_tokenizer.word_index),train_ent_vect.shape[1],train_ent_vect


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img



if __name__ == '__main__':

    ent = []
    for i in range(1,6):
        ent.append(read_ent(f"/root/entity_{i}.txt"))

    ent_vect_map = {}
    ent_max_time = []
    ent_vocab_size = []
    for i in range(1,6):
      vocab_size, max_len,ent_vect = ent_vectorise(ent[i-1],f"ent_{i}")
      ent_vect_map[f"ent_{i}"]=ent_vect
      ent_vocab_size.append(vocab_size)
      ent_max_time.append(max_len)

    train_image = tf.data.TextLineDataset("/root/image_order.txt")
    train_image = train_image.map(load_image,
                              num_parallel_calls=tf.data.AUTOTUNE)

    ent1_dataset = tf.data.Dataset.from_tensor_slices(ent_vect_map["ent_1"])
    ent2_dataset = tf.data.Dataset.from_tensor_slices(ent_vect_map["ent_2"])
    ent3_dataset = tf.data.Dataset.from_tensor_slices(ent_vect_map["ent_3"])
    ent4_dataset = tf.data.Dataset.from_tensor_slices(ent_vect_map["ent_4"])
    ent5_dataset = tf.data.Dataset.from_tensor_slices(ent_vect_map["ent_5"])

    train_dataset = tf.data.Dataset.zip((train_image,ent1_dataset,
                                    ent2_dataset,ent3_dataset,
                                    ent4_dataset,ent5_dataset))

    train_dataset = train_dataset.batch(64)
    fit(train_dataset,ent_vocab_size,15,ent_max_time,H=299,W=299,C=3)    
