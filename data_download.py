import os
import json
import shutil
import random
import pickle
import tensorflow as tf

warm_up = "<WARMUP>"
ent_end = "<EOI>"
def save_train_images():
      global warm_up, ent_end
      counter = 0
      e_1 = []
      e_2 = []
      e_3 = []
      e_4 = []
      e_5 = []
      image_order = []
      for i in range(3):
        with open(f"/root/dataset_trainticket/train/output_{i}/text/text_synth.pkl","rb") as f:
          data = pickle.load(f)
        images = list(data.keys())
        random.shuffle(images)

        for image in images:
          shutil.copy(os.path.join("/root/dataset_trainticket/train", image+".jpg"),
                      f"/root/trainx/image_{counter}.jpg")
          if os.path.exists(f"/root/trainx/image_{counter}.jpg"):
              image_order.append(f"/root/trainx/image_{counter}.jpg")

              date = data[image]["date"]
              dest_stn = data[image]["destination_station"]
              name = data[image]["name"]
              seat_ctg = data[image]["seat_category"]
              start_stn = data[image]["starting_station"]
              tck_num = data[image]["ticket_num"]
              tck_rate = data[image]["ticket_rates"]
              train_num =  data[image]["train_num"]

              temp_e1 = [warm_up] #ticket number
              for char in tck_num:
                temp_e1.append(char)
              e_1.append(" ".join(temp_e1))


              temp_e2 = [warm_up] #start_stn,train_number, destination_stn
              for char in start_stn:
                temp_e2.append(char)
              temp_e2.append(ent_end)

              for char in train_num:
                temp_e2.append(char)
              temp_e2.append(ent_end)

              for char in dest_stn:
                temp_e2.append(char)

              e_2.append(" ".join(temp_e2))

              temp_e3 = [warm_up] #date
              for char in date:
                temp_e3.append(char)
              e_3.append(" ".join(temp_e3))

              temp_e4 = [warm_up] #tck_rate, seat_ctg
              for char in tck_rate:
                temp_e4.append(char)
              temp_e4.append(ent_end)

              for char in seat_ctg:
                temp_e4.append(char)
              e_4.append(" ".join(temp_e4))

              temp_e5 = [warm_up] #name
              for char in name:
                temp_e5.append(char)
              e_5.append(" ".join(temp_e5))
          counter += 1

      with open("image_order.txt","w") as f:
        f.write("\n".join(image_order))

      with open("entity_1.txt","w") as f:
        f.write("\n".join(e_1))

      with open("entity_2.txt","w") as f:
        f.write("\n".join(e_2))

      with open("entity_3.txt","w") as f:
        f.write("\n".join(e_3))

      with open("entity_4.txt","w") as f:
        f.write("\n".join(e_4))

      with open("entity_5.txt","w") as f:
        f.write("\n".join(e_5))

if __name__ == '__main__':
    save_train_images()
