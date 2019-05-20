from fastai import * #get_transforms, imagenet_stats ,ClassificationInterpretation
#from fastai.models import resnet50
from fastai.vision import * #ImageList , cnn_learner
import pandas as pd
import numpy as np

from utilities import get_data

BASE_PATH = "./model_data"
def train():

# prep data
  df  =  pd.read_csv("./LABELS/all.txt" , names = ["name", "label",'Cond', 'date', 'camid', 'file'], sep =" ")
  df[["Cond","date","camid","file"]] =  df["name"].str.split("/", n = 4,expand = True) 
  df["name"] =  df["name"].apply(lambda x :  "./PATCHES/"+x)
  df_sh = df[["name", "label"]]
  src= (ImageList.from_df( df_sh,".",) #Where to find the data? -> in path and its subfolders
          .split_by_rand_pct()              #How to split in train/valid? -> use the folders
          .label_from_df(cols='label') )             
  data = (src.transform(get_transforms(), size=128)       #Data augmentation? -> use tfms with a size of 64
          .databunch()
          .normalize(imagenet_stats))

  learn = cnn_learner(data, model.resnet50,  model_dir=BASE_PATH).to_fp16()
  learn.lr_find()
  learn.recorder.plot()
  lr = 0.01
  learn.fit_one_cycle(2, slice(lr))
  learn.recorder.plot_losses()
  learn.save(BASE_PATH+"resnet_cars.h5")
  interp = ClassificationInterpretation.from_learner(learn)
  interp.plot_confusion_matrix()
  return learn

class PREDICT:
  def __init__(self):
    exists = os. path. isfile(BASE_PATH+"resnet_cars.h5")
    if not exists :
      train()
    self.learn = cnn_learner(data, resnet50,  model_dir=BASE_PATH).to_fp16()
    self.learn.load(BASE_PATH+"resnet_cars.h5")

  def occupied(self, image):
    im = Image(pil2tensor(image,np.float32 ).div_(255))
    log_preds_single = self.learn.predict(im) # Predict Imag
    return log_preds_single[0].obj

  def find_cars_in_slots(self, park_slots, image, plot=False, k=0) : 
    found = np.zeros(len(park_slots)).astype(int)
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    colors =["green","red"]
    
    for i in range(len(park_slots)) : 
      outbox =park_slots.loc[i,["x2","y1","x1","y2"]].values.astype(int)
      crop = image.crop(outbox)
      found[i] = int(self.occupied(crop))
      if plot :
          draw = ImageDraw.Draw(image)
          label =  str(park_slots.at[i,"labels"])
          label_size = draw.textsize(label, font)
          left, top, right, bottom = outbox
  #         top, left, bottom, right = box
          top = max(0, np.floor(top + 0.5).astype('int32'))
          left = max(0, np.floor(left + 0.5).astype('int32'))
          bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
          right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
          #print(label, (left, top), (right, bottom))

          if top - label_size[1] >= 0:
              text_origin = np.array([left, top - label_size[1]])
          else:
              text_origin = np.array([left, top + 1])
          for it in range(thickness):
              draw.rectangle(
                  [left + it, top + it, right - it, bottom - it], outline=colors[found[i]],)# linestyle = style[c])
          draw.rectangle(
              [tuple(text_origin), tuple(text_origin + label_size)],
              fill="white")
          draw.text(text_origin, label, fill=(0, 0, 0), font=font, color="b")
          del draw
    if plot:
      plt.figure(figsize = (20,20))
      plt.imshow(image)
      plt.show()
      plt.close()
    return found


class DETECTION :
  def __init__(self) :
    self.image_data =  get_data()
    self.camera =  str
    self.predict = PREDICT()

  def process_images(self,camera =  "camera9"):
    image_data =  self.image_data
    images =  image_data[image_data["camera"] == self.camera ]["path"].values
    images = np.sort(images)
    park_slots  =  pd.read_csv("./parkings/"+camera+".csv")
    occpancy =  np.zeros((len(park_slots), len(images)))
    for i, image in enumerate(images ):
      occpancy[:,i] =  self.predict.find_cars_in_slots( park_slots, image, plot=False, k=0) 
    return occpancy