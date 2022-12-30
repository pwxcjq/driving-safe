from keras.models import load_model
import os
import cv2
import glob
from keras.utils import image_utils
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib as mlp
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import imread
from keras.utils import image_utils
from keras.models import load_model

model = load_model('model/vgg16_model.h5')

basedir = ''

font2 = {'family': 'SimHei',
		'weight': 'normal',
		'size': 10}
mlp.rcParams['font.family'] = 'SimHei'
mlp.rcParams['axes.unicode_minus'] = False

a = [i for i in range(1, 10)]
fig = plt.figure(figsize=(10,10))
status = ["safe driving", "texting-right", "phone-right", "texting-left",  
		"phone-left", "operating radio", "drinking", "reaching behind", 
		"hair and makeup", "talking"]
for i in a:
	img_name = str(i) + '.jpg'
	img_path = img_name
	img = image_utils.load_img(img_path, target_size=(224,224))
	img = image_utils.img_to_array(img)
	x = np.expand_dims(img,axis=0)
	x = preprocess_input(x)
	# x_vgg = model_vgg.predict(x)
	# x_vgg = x_vgg.reshape(1, 25088)
	result = model.predict(x)
	img_ori = image_utils.load_img(img_name,target_size=(250,250))
	plt.subplot(3,3,i)
	plt.imshow(img_ori)    
	max_idx = np.argmax(result)
	probability = result[0][max_idx]
	print('img',i,': ',max_idx)
	plt.title('%s: %.2f%%' % (status[max_idx], probability*100))
plt.suptitle('VGG16 prediction test',fontweight ="bold")
plt.tight_layout()
plt.show()

# img_name = '6.jpg'
# img_path = img_name
# img = image_utils.load_img(img_path, target_size=(224,224))
# img = image_utils.img_to_array(img)
# x = np.expand_dims(img,axis=0)
# x = preprocess_input(x)
# result = model.predict(x)
# img_ori = image_utils.load_img(img_name,target_size=(250,250))
# plt.plot()
# plt.imshow(img_ori)
# max_idx = np.argmax(result)
# probability = result[0][max_idx]
# # print('img',i,': ',max_idx)
# plt.title('%s: %.2f%%' % (status[max_idx], probability*100))
# plt.tight_layout()
# plt.show()






