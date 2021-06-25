import io
import os
import json
import tkinter
import numpy as np
from tkinter import * 
import tensorflow as tf
from PIL import ImageTk, Image
from tkinter import filedialog

class EATEN_OCR_GUI():
    def __init__(self):
        self.root = Tk()
        self.width = 650
        self.height = 600
        self.load_encoder_decoder_model()
        self.display_gui()
    
    def load_encoder_decoder_model(self):
   
        self.cnn_encoder  = tf.keras.models.load_model("cnn_encoder") #inception encoder
        self.ent_dict = {}
        self.ent_token_dict = {}
        for i in range(5):
            
            #load tokenizer to get the vocab
            with open(f"tokenizer/ent_{i+1}_tokenizer.json") as f:
                self.ent_token_dict[f"ent_{i}"] = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))
            
            #load entity decoder
            self.ent_dict[f"ent_{i}"] = tf.keras.models.load_model(f"entity_decoders/ent_{i}")
        
        #build index_word map from word_index map
        self.ent_idx_word = {}
        for i in range(5):
            temp_idx_word = {}
            word_index = self.ent_token_dict[f"ent_{i}"].word_index
            word_index["PAD"]=0
            for word in word_index:
                temp_idx_word[word_index[word]]=word
            self.ent_idx_word[f"ent_{i}"] = temp_idx_word
        
        self.ent_max_len = [] #max time step for each entity
        for i in range(1,6):
            with open(f"train_y/entity_{i}.txt") as f:
                max_len = -1
                for line in f:
                    max_len = max(max_len,len(line.split())-1)
            self.ent_max_len.append(max_len)
        
        #vocab size of each entity 
        self.vocab_size = []
        for i in range(5):
            self.vocab_size.append(len(self.ent_idx_word[f"ent_{i}"]))


        print("ENCODER and ENTITY DECODER LOADED....")
    
    def prepare_image(self,image):
        if image.mode != "RGB":
            image=image.convert("RGB")
        image = image.resize((299,299))
        image = tf.keras.preprocessing.image.img_to_array(image) #PIL image to numpy array
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.inception_v3.preprocess_input(image) 

        return tf.constant(image)
    
    def predict(self,image):
  
        img = self.prepare_image(image)
        F = self.cnn_encoder(img,training=False)

                        

        
        state_h = tf.random.uniform(shape=(1,128), minval = -10, maxval= 10,seed=42)
        state_c = tf.random.uniform(shape=(1,128), minval = -10, maxval= 10,seed=42)
        c_t = tf.random.normal(shape=(1,2048),seed=42) #initial context vector
        prev_char_one_hot=tf.zeros((1,self.vocab_size[0]))

        #warmup for first encoder
        inputs = [c_t,F,prev_char_one_hot,state_h,state_c]
        O_t,c_t, state_h,state_c= self.ent_dict["ent_0"](inputs,training=False)
        pred = tf.cast(tf.reshape(tf.argmax(O_t,axis=1),shape=(-1,1)),tf.int32)
        prev_char_one_hot  = tf.squeeze(tf.one_hot(pred,depth = self.vocab_size[0],axis=-1),axis=0)
        
        entity_pred = []
        for i in range(5):
            temp = []
            pred_char = None
            count = 0
            #iterate till pad or max length reach
            while pred_char != "PAD" and count < self.ent_max_len[i]:
                inputs = [c_t,F,prev_char_one_hot,state_h,state_c]
                O_t,c_t, state_h,state_c= self.ent_dict[f"ent_{i}"](inputs,training=False)
                pred = tf.cast(tf.reshape(tf.argmax(O_t,axis=1),shape=(-1,1)),tf.int32)
                prev_char_one_hot = tf.squeeze(tf.one_hot(pred,depth = self.vocab_size[i],axis=-1),axis=0)
                
                pred_char = self.ent_idx_word[f"ent_{i}"][int(pred[0][0])]
                if pred_char != "PAD":
                    temp.append(pred_char)
                    count += 1
            entity_pred.append(temp)

            if i != 4:
                #warmup for next encoder
                prev_char_one_hot=tf.zeros((1,self.vocab_size[i+1]))
                inputs = [c_t,F,prev_char_one_hot,state_h,state_c]
                O_t,c_t, state_h,state_c= self.ent_dict[f"ent_{i+1}"](inputs,training=False)
                pred = tf.cast(tf.reshape(tf.argmax(O_t,axis=1),shape=(-1,1)),tf.int32)
                prev_char_one_hot  = tf.squeeze(tf.one_hot(pred,depth = self.vocab_size[i+1],axis=-1),axis=0)
                pred_char = None
        
        return entity_pred
    
    def get_ent_clean(self,pred_values):
    
       #pred values is a list of list 
       #containing predicted values of the 5 encoders
    
        entity_names = ["num_entry","sstn_entry","tnum_entry","dstn_entry",
                        "date_entry","trate_entry","sctg_entry","name_entry"]
        pred_entity = dict.fromkeys(entity_names,"Not able to predict") 

        pred_entity["num_entry"] = "".join(pred_values[0]) # 1st  encoder has only one entity
        
        temp = "".join(pred_values[1]).split("<EOI>") # 2nd  encoder has 3 entity
        ent2 = []
        for i in temp:
            if i != "":
                ent2.append([i])

        n = len(ent2)
        if n > 0:
            pred_entity["sstn_entry"]=ent2[0][0]
        if n > 1:
            pred_entity["tnum_entry"]=ent2[1][0]
        if n > 2:
            pred_entity["dstn_entry"]=ent2[2][0]

        temp = "".join(pred_values[2]) # 3rd  encoder has only one entity
        if temp != "":
            pred_entity["date_entry"]=temp

        temp = "".join(pred_values[3]).split("<EOI>") # 4th  encoder has two entity
        ent4 = []
        for i in temp:
            if i != "":
                ent4.append([i])

        n = len(ent4)
        if n > 0:
            pred_entity["trate_entry"]=ent4[0][0]
        if n > 1:
            pred_entity["trate_entry"]=ent4[1][0]

        temp = "".join(pred_values[4]) # 5th encoder has only one entity
        if temp != "":
            pred_entity["name_entry"]=temp
            
        
        return pred_entity

    
    def display_gui(self):
        ws = self.root.winfo_screenwidth() 
        hs = self.root.winfo_screenheight() 
        x = (ws/2) - (self.width/2)
        y = (hs/2) - (self.height/2)
        self.root.geometry('%dx%d+%d+%d' % (self.width,
                                        self.height, x, y))
        self.root.title("EATEN OCR TRAIN TICKET")
        self.img_label =  Label(self.root,bg="gray")
        self.img_label.place(x=0,y=0,height=600, width=400)
        self.entry_dict = {}

        self.num_entry = Text(self.root,state='disabled', font=('Roman',13,'bold'),width=20,height=1.5,relief="sunken")
        self.num_entry.place(x=400,y=25)
        self.entry_dict["num_entry"] = self.num_entry


        self.num_label = Label(self.root,text="Ticket Number",font=('Roman',13,'bold'))
        self.num_label.place(x=400,y=4)

        self.sstn_label = Label(self.root,text="Start station",font=('Roman',13,'bold'))
        self.sstn_label.place(x=400,y=75)

        self.sstn_entry = Text(self.root, state='disabled',font=('Roman',13,'bold'),width=20,height=1.5,relief="sunken")
        self.sstn_entry.place(x=400,y=95)
        self.entry_dict["sstn_entry"] = self.sstn_entry

        self.dstn_label = Label(self.root,text="Destination station",font=('Roman',13,'bold'))
        self.dstn_label.place(x=400,y=145)

        self.dstn_entry = Text(self.root, state='disabled',font=('Roman',13,'bold'),width=20,height=1.5,relief="sunken")
        self.dstn_entry.place(x=400,y=165)
        self.entry_dict["dstn_entry"] = self.dstn_entry

        self.tnum_label = Label(self.root,text="Train Number",font=('Roman',13,'bold'))
        self.tnum_label.place(x=400,y=215)

        self.tnum_entry = Text(self.root,state='disabled', font=('Roman',13,'bold'),width=20,height=1.5,relief="sunken")
        self.tnum_entry.place(x=400,y=235)
        self.entry_dict["tnum_entry"] = self.tnum_entry

        self.trate_label = Label(self.root,text="Ticket rate",font=('Roman',13,'bold'))
        self.trate_label.place(x=400,y=285)

        
        self.trate_entry = Text(self.root,state='disabled', font=('Roman',13,'bold'),width=20,height=1.5,relief="sunken")
        self.trate_entry.place(x=400,y=305)
        self.entry_dict["trate_entry"] = self.trate_entry

        self.sctg_label = Label(self.root,text="Seat Category",font=('Roman',13,'bold'))
        self.sctg_label.place(x=400,y=355)
        

        
        self.sctg_entry = Text(self.root, state='disabled',font=('Roman',13,'bold'),width=20,height=1.5,relief="sunken")
        self.sctg_entry.place(x=400,y=375)
        self.entry_dict["sctg_entry"] = self.sctg_entry

        self.date_label = Label(self.root,text="Date",font=('Roman',13,'bold'))
        self.date_label.place(x=400,y=425)

        
        self.date_entry = Text(self.root, state='disabled',font=('Roman',13,'bold'),width=20,height=1.5,relief="sunken")
        self.date_entry.place(x=400,y=445)
        self.entry_dict["date_entry"] = self.date_entry

        self.name_label = Label(self.root,text="Name",font=('Roman',13,'bold'))
        self.name_label.place(x=400,y=495)

        
        self.name_entry = Text(self.root, state='disabled',font=('Roman',13,'bold'),width=20,height=1.5,relief="sunken")
        self.name_entry.place(x=400,y=515)
        self.entry_dict["name_entry"] = self.name_entry
        
        self.send_f = Button(self.root,fg='white',font=('arial black',12,'bold'),
                        text='Choose Image',bg="green",width=10,command=self.proc_img)
        self.send_f.focus_set()
        self.send_f.place(x=405,y=574)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def proc_img(self):
        
        filename = filedialog.askopenfilename(initialdir=os.path.abspath(os.getcwd()),
                        filetypes =[('Images', '*.png'),('Images','*.jpg'),('Images', '*.jpeg')],title="Choose image")
        if os.path.exists(filename):

            #display image
            dis_image = Image.open(filename)
            dis_image = dis_image.resize((self.img_label.winfo_width(),
                                         self.img_label.winfo_height()),Image.ANTIALIAS)
            dis_image = ImageTk.PhotoImage(dis_image)
            
            self.img_label.configure(image=dis_image)
            self.img_label.image=dis_image


            image = open(filename, "rb").read()
            image = Image.open(io.BytesIO(image)) #read the image
            predicted_entity = self.predict(image) #do ocr
            predicted_entity = self.get_ent_clean(predicted_entity)
            print("PARSED")

            for entity in predicted_entity:
                self.entry_dict[entity].config(state=NORMAL)
                self.entry_dict[entity].delete('1.0','end')
                self.entry_dict[entity].insert('1.0',predicted_entity[entity])
                self.entry_dict[entity].config(state=DISABLED)
    
    def on_closing(self):
        print("\n"*7)
        print("EATEN OCR TRAIN TICKET [CLOSING]")
        self.root.destroy()


            
            

  




if __name__ =="__main__":
    EATEN_OCR_GUI()