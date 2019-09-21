from utilities import assign_next_frame ,get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from PIL import   ImageFont, ImageDraw
from tqdm import tqdm_notebook as tqdm
import cv2
from datetime import datetime

from yolo import YOLO 
from resnet_occupancy import PREDICT

yolo = YOLO(score =0.1)

def plot_detection(images,index = 0 , figsize=(33,64)):
  photo = cv2.imread(images[index])
  detected = yolo.detect_image(PILImage.fromarray(photo))
  f = plt.figure(figsize=figsize)
  sp = f.add_subplot(1, 2, 1)
  sp.axis('Off')
  plt.imshow(photo)
  sp = f.add_subplot(1, 2, 2)
  sp.axis('Off')
  plt.imshow(detected)

def create_boxes(images):
  data  = pd.DataFrame()#[],columns =["x1","y1","x2","y2", "score","class","file"])
  for i in range(len(images)):
    out_boxes, out_scores, out_classes, labels = yolo.find_objects(PILImage.open(images[i]))
    out_boxes = out_boxes.reshape((len(labels),4))
    df  = pd.DataFrame(out_boxes , columns =["y1","x2","y2","x1"])
    
#     {"y1":out_boxes,"x2","y2","x1"}
    df["score"] = out_scores
    df["class"] =out_classes
    df["frame"] = i
    df["labels"] = np.arange(len(df))
    data = data.append(df, ignore_index=True)
    
  data["xc"] = (data["x1"] + data["x2"])/2
  data["yc"] = (data["y1"] + data["y2"])/2
  data["w"] =  data["x1"] -data["x2"]
  data["b"] = data["y2"] - data["y1"]
  data["a"]= data["w"] *data["b"]
  data["d"] =  np.sqrt(data["b"]*data["b"]  + data["w"]*data["w"] )
  data = data[["labels", 'x1', 'y1', 'x2', 'y2',  'xc', 'yc', 'w',  'd' , 'b', 'score', 'class', 'frame' ,'a']]
#   mask = data["class"].apply(lambda x: x in [2, 5, 7]) 
#   data = data[mask]
  return data



pd.options.mode.chained_assignment = None # Disable warning from pandas




def compute_distance(df, image, th = 0.6, label = "Parking Slots", plot = False):
  print("MERGE TH for combine =", th)
  df.reset_index(drop=True, inplace=True)
  n =  len(df)
  base_col = ['x1', 'y1', 'x2', 'y2',  'xc', 'yc', 'w' , 'b',]
  df.reset_index(drop=True, inplace=True)
  mat, _, _, _ = assign_next_frame(df, df, th = 0.6)
  np.fill_diagonal(mat, -9)
  mat = np.tril(mat)
  count = n
  to_merge = []
  while count > 0:
    r,k = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
    if mat[r,k] > th :
      to_merge.append([r,k])
      x = k
      if df["found"][r] <  df["found"][k] :
        x =r
          mat[x,:] = -9
          mat[:,x] = -9
    else :     
    mat[r,:] = -9
    mat[:,k] = -9
    mat[k,:] = -9
    mat[:,r] = -9
    count = count -1
#   print(to_merge)
  for i in range(len(to_merge)):
    r = to_merge[i][0]
    k = to_merge[i][1]
    if df["found"][r] <  df["found"][k] :
      x =r
      r = k
      k = x
    df.loc[r,base_col] =(df.loc[r,base_col] * df.loc[r,"found"] +  df.loc[r,base_col]* df.loc[k,"found"])/(df.loc[r,"found"]+df.loc[k,"found"])
    df.at[r,"found"] =  df.at[r,"found"]+ df.at[k,"found"]
    df.drop(k, axis=0, inplace = True)
  if plot :
    plt.figure()
    box_img = yolo.draw_rect(PILImage.fromarray(cv2.imread(image)),\
           df[["y1","x2","y2","x1"]].values, df.index.values, df["class"], df["labels"].values)
    f = plt.figure(figsize=(20,40))
    plt.imshow(box_img)
    plt.title(label)
    plt.show()
    plt.close()
  return df

def look_for_slots(data, img=[],PRUNE_TH = 3, plot = True,
                                PRUNE_STEP =  10,
                                MERGE_STEP =  20,
                                MERGE_TH =  0.65):
  
  
  n_fr = data["frame"].nunique()
  cols = ["labels", 'x1', 'y1', 'x2', 'y2',  'xc', 'yc', 'w' , 'b',"class",'a' ]
  base_col = ['x1', 'y1', 'x2', 'y2',  'xc', 'yc', 'w' , 'b']
  slots  = data[data["frame"] == 0 ][cols]
  slots["found"] = 1
  def plot_images():
    df = slots[['x1', 'y1', 'x2', 'y2',"found","labels" ]]
    df["class"] ="empty"
    msk = df["labels"].isin(id_map.values())
    df.loc[msk,"class"] = "occupy"
    nw_df = new[['x1', 'y1', 'x2', 'y2',"labels" ]]
    nw_df["class"] = "new"
    nw_df["found"]=1
    df = df.append(nw_df,  sort=True).reset_index(drop=True)
    df["found"] = df["found"].astype(int)
    plot_frame( img[i], df[["y1","x2","y2","x1"]].values,  df["class"].values, df["found"], df["labels"])
#   out_boxes,  out_classes, found, labels
# "empty":"#4a148c","occupy":"#f44336", "new":"#7cb342","del":"#80deea" 
  print("LOOKING FOR PARKING SLOTS INSIDE IMAGE FRAMES")
  for i in  tqdm(range(1 ,n_fr)) : 
    post =  data[data["frame"]==i].reset_index(drop=True)
    _,iou, id_map, status = assign_next_frame(slots, post, th = MERGE_TH)
    #print(id_map.keys(), status.sum())
    
    ## found again
    mask = post["labels"].isin(id_map.keys())
    slots.loc[status,"found"] = slots.loc[status,"found"] +1
    occupy =  post[mask]
    occupy["labels"] = occupy["labels"].map(id_map)
    slots.sort_values(by =["labels"] , inplace=True)
    slots.reset_index(drop=True, inplace=True)
    occupy.sort_values(by =["labels"], inplace = True)
    occupy.reset_index(drop=True, inplace=True)
    slots.loc[status,base_col] = slots.loc[status,base_col].values *(1 - 1/(i+1)) +  occupy[base_col].values/(i+1)
     
    # clean up
    if i % PRUNE_STEP ==0 :
      slots.drop(slots[slots["found"] < PRUNE_TH+1].index, inplace=True) 
      #print(slots)

    # merge 
    if i % MERGE_STEP ==0 :
      
      slots = compute_distance(slots, img[i-1], th = MERGE_TH, label = "Parking Slots "+ str(i))
       
    # new
    idx = np.logical_not(post["labels"].isin(id_map.keys()))
    new  =  post[idx]
    new["labels"] =  new["labels"] + slots["labels"].max() + 1
    new  = new[cols]
    if len(new ) > 0 :
      new["found"] = 1
    slots = slots.append(new,  sort=True).reset_index(drop=True) 
    if plot and (i % MERGE_STEP*5 ==0):
      plot_images()
  slots = compute_distance(slots, img[0], th = MERGE_TH*0.9, label = "Parking Slots "+ str(MERGE_STEP))  
  slots.drop(slots[slots["found"] < PRUNE_TH*3].index, inplace=True) 
  print(len(slots), "SLOTS FOUND")
  plot_images()
  
  slots = compute_distance(slots, img[0], th = MERGE_TH*0.9, label = "Parking Slots "+ str(MERGE_STEP))  
  print(len(slots), "SLOTS FOUND")
  plot_images()
  
  slots = compute_distance(slots, img[0], th = MERGE_TH*0.9, label = "Parking Slots "+ str(MERGE_STEP))  
  print(len(slots), "SLOTS FOUND")
  plot_images()
  return slots

       
def plot_frame( image, out_boxes,  out_classes, found, labels):
    image =PILImage.fromarray(cv2.imread(image))
    font = ImageFont.truetype("arial.ttf",
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    #colors ={"empty":"#4a148c","occupy":"#f44336", "new":"#7cb342","del":"#80deea"  }
    colors ={"empty":"green","occupy":"red", "new":"blue","del":"grey"  }
    style= {"empty":"--","occupy":"-", "new":"-","del":":"  }
    plt.figure(figsize = (20,20))
    for i, c in list(enumerate(out_classes)):
    # out_classes = [retained, new, deleted]

        box = out_boxes[i]
        label = str(labels[i]) + " (" + str(found[i]) +")"
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])


        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c],)# linestyle = style[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill="white")
        draw.text(text_origin, label, fill=(0, 0, 0), font=font, color="b")
        del draw
#     plt.imshow(image)
#     image = image.resize((image.size[0]//2,image.size[1]//2))
    plt.imshow(image)
    plt.show()
    ts = int(datetime.now().timestamp()*10000)
#     plt.imsave(TMP_MOVIE+str(ts)+".png",image)
    plt.close()
    return image
if __name__ == "__main__": 
  image_data =  get_data()


  for camera  in  image_data["camera"].unique():
      images =  image_data[image_data["camera"] == camera ]["path"].values
      images = np.sort(images)
      img_train = images[:len(images) // 2]
      park_data =  create_boxes(img_train)
      park_slots = look_for_slots(park_data, img= img_train,plot =False,
                                      PRUNE_TH = 1,
                                      PRUNE_STEP =  10,
                                      MERGE_STEP =  50,
                                      MERGE_TH =  0.8)
      park_slots.drop(park_slots[park_slots["found"] < 3].index, inplace=True) 
      park_slots=compute_distance(park_slots, images[20], th=0.2,  label ="20")
      park_slots[['x1', 'y1', 'x2', 'y2',  'xc', 'yc', 'w' , 'b', "found"]] = park_slots[['x1', 'y1', 'x2', 'y2',  'xc', 'yc', 'w' , 'b', "found"]].astype(int)
      #   create_video(path ="./movie/tmp/",file_name="./movie/train/"+ camera+"_train.gif" )
      park_slots= park_slots.reset_index(drop=True)
      park_slots.to_csv("./parkings/"+camera+".csv", index = False)
    
  
  
  
  
  
  
  
def train_and_detect(image_data, camera, pred_fr=120):
  print("PROCESSING", camera)
  images =  image_data[image_data["camera"] == camera ]["path"].values
  images = np.sort(images)

  img_train = images[:len(images) // 2]
  img_pred = images[len(images) // 2:len(images) // 2 +2]
  park_data =  create_boxes(img_train)
  park_slots = look_for_slots(park_data, img= img_train,plot =False,
                                  PRUNE_TH = 1,
                                  PRUNE_STEP =  10,
                                  MERGE_STEP =  50,
                                  MERGE_TH =  0.8)
  park_slots.drop(park_slots[park_slots["found"] < 3].index, inplace=True) 
  park_slots=compute_distance(park_slots, images[20], th=0.2,  label ="20")
  park_slots[['x1', 'y1', 'x2', 'y2',  'xc', 'yc', 'w' , 'b', "found"]] = park_slots[['x1', 'y1', 'x2', 'y2',  'xc', 'yc', 'w' , 'b', "found"]].astype(int)
#   create_video(path ="./movie/tmp/",file_name="./movie/train/"+ camera+"_train.gif" )
  park_slots= park_slots.reset_index(drop=True)
  img_pred_small = img_pred[:pred_fr]
  found = np.zeros((len(img_pred_small),len(park_slots)))
  print("PROCESSING ",pred_fr, "FRAMES TO DETECT OCCUPANCY")
  for i in tqdm(range(len(img_pred_small))):
    image = PILImage.open(img_pred[i]).convert('RGB')
    found[i,:] = find_cars_in_slots(park_slots, image, plot=True, k=i)
#   create_video(path ="./movie/tmp/",file_name="./movie/train/"+ camera+"_detection.gif" ) 
  return park_slots, found.astype(bool)
    
