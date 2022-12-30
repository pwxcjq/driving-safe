from keras.models import load_model
import os
import cv2
import glob
from keras.utils import image_utils
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import pandas as pd
import matplotlib as mlp
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import imread
from keras.utils import image_utils
from keras.models import *
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import random

model_name = 'model2_1112'
img_num = 32

loadModel = './model_2/' + model_name + '.h5'
toExcel = model_name + '.xlsx'

model = load_model(loadModel)
dict = {}

def show_test(model):
    for i in [j for j in range(1, img_num+1)]:
        img_path = './test/img (%s).jpg'%i
        img = cv2.imread(img_path)
        img = cv2.resize(img,(480,320))
        status = ["safe driving", "texting-right", "phone-right", "texting-left",  
		"phone-left", "operating radio", "drinking", "reaching behind"] 
		#"hair and makeup", "talking"]

        x = img.copy()
        x.astype(np.float32)
        predictions = model.predict(np.expand_dims(x, axis=0))
        predictions = predictions[0]
        max_idx = np.argmax(predictions)

        dict['img (%s)'%i] = str(status[max_idx])

show_test(model)
# print(dict)
s_dict = pd.Series(dict)
df_s_dict = pd.DataFrame(s_dict, columns=[model_name])
print(df_s_dict)
# df_s_dict.to_excel(toExcel)
