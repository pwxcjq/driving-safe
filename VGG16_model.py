import os
import cv2
import glob
import numpy as np
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

model_image_size = 224	# VGG16要求提供图片尺寸224x224

print('-----------load train data-----------')
X_train = list()
y_train = list()
for i in range(10):
    dir = os.path.join('train', 'c%d'%i)
    image_files = glob.glob(os.path.join(dir,'*.jpg'))
    print('load {}, image count={}'.format('train'+'c%d'%i, len(image_files)))
    for image_file in image_files:
        image = cv2.imread(image_file)
        X_train.append(cv2.resize(image, (model_image_size, model_image_size)))
        label = np.zeros(10, dtype=np.uint8)
        label[i]=1
        y_train.append(label)
X_train = np.array(X_train)	# 将X_train向量化
y_train = np.array(y_train)

print('-----------load valid data-----------')
X_valid = list()
y_valid = list()
for i in range(10):
    dir = os.path.join('valid', 'c%d'%i)
    image_files = glob.glob(os.path.join(dir,'*.jpg'))
    print('load {}, image count={}'.format('valid'+'c%d'%i, len(image_files)))
    for image_file in image_files:
        image = cv2.imread(image_file)
        X_valid.append(cv2.resize(image, (model_image_size, model_image_size)))
        label = np.zeros(10, dtype=np.uint8)
        label[i]=1
        y_valid.append(label)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

# print(X_train.shape)	# (20787, 224, 224, 3): 20787个训练数据
# print(y_train.shape)	# (20787, 10): 输出10个可能结果
# print(X_valid.shape)	# (1637, 224, 224, 3)
# print(y_valid.shape)	# (1637, 10)

model_vgg16 = VGG16(input_tensor=Input((model_image_size, model_image_size, 3)), 
            weights='imagenet', include_top=False)

for layers in model_vgg16.layers:
    layers.trainable = False

x = GlobalAveragePooling2D()(model_vgg16.output)
x = Dropout(0.25)(x)
x = Dense(10, activation='softmax')(x)
model = Model(model_vgg16.input, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print('compile is over')

model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid))
print('fit is over')
model.save('./model/vgg16_model.h5')
print('save is over')




