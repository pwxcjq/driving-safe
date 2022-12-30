from keras.models import load_model
import os
import cv2
import glob
# from keras.utils import image_utils
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import matplotlib as mlp
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import imread
# from keras.utils import image_utils
from keras.models import *
from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
import random

model1 = load_model('./11.27/model/4_2.h5')
# print(model1.layers)
# model1.summary()

# font2 = {'family': 'SimHei',
# 		'weight': 'normal',
# 		'size': 5}
mlp.rcParams['font.family'] = 'SimHei'
mlp.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10 

def show_heatmap(model, weight):
    plt.figure(figsize=(10, 10))
    for i in [j for j in range(1,33)]:
        plt.subplot(8, 4, i)
        # img_path = './CAM_testpic/%s.jpg'%i
        img_path = './11.27/testCAM/img (%s).jpg'%i
        # img_path = 'img (%s).jpg'%i
        img = cv2.imread(img_path)
        img = cv2.resize(img,(480,320))
        x = img.copy()
        x.astype(np.float32)
        out, predictions = model.predict(np.expand_dims(x, axis=0))
        predictions = predictions[0]
        out = out[0]
        # print(out.shape)
        
        max_idx = np.argmax(predictions)
        probability = predictions[max_idx]

        status = ["safe driving","texting-right","phone-right","texting-left",
        "phone-left","operating radio","drinking","reaching behind"]
        # "hair and makeup","talking"]

        plt.title('%s: %.2f%%' % (status[max_idx], probability*100))
    
        heat = (probability-0.5) * np.matmul(out, weight)
        heat = heat[:,:,max_idx]
        heat = (heat-heat.min())/(heat.max()-heat.min())
        heat = cv2.resize(heat,(480,320))
        heatmap = cv2.applyColorMap(np.uint8(heat*255), cv2.COLORMAP_JET)
        heatmap[np.where(heat <= 0.25)] = 0
        out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0) #(img, 0.8, heatmap, 0.4, 0)

        plt.axis('off')
        plt.imshow(out[:,:,::-1])
        # plt.imshow(image_utils.load_img(img_path,target_size=(320,480)))
    plt.suptitle('InceptionV3 CAMtest',fontweight ="bold")
    plt.tight_layout()
    plt.show()

# weights1 = model.layers[313].get_weights()[0] 314
# print(weights1.shape)
# print(model1.layers[311]) 311
weights12 = model1.layers[314].get_weights()[0]
# print(weights12.shape)
layer_output1 = model1.layers[311].output
# print(layer_output1)

Model1 = Model(model1.input, [layer_output1, model1.output])
print("layer_output {0}".format(layer_output1))
print("weights shape {0}".format(weights12.shape))
show_heatmap(Model1, weights12)


