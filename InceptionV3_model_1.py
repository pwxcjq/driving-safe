import os
import cv2
import glob
import numpy as np
import pandas as pd
from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.utils import image_utils
from keras.callbacks import ModelCheckpoint


save_path = './model/_i_add_model_9.28_14_0.9519.h5'

train_gen = ImageDataGenerator( #数据增强
    featurewise_std_normalization=True, #数据标准化(除以数据标准差)
    samplewise_std_normalization=False, 
    rotation_range=10., #随机转动角度10
    width_shift_range=0.05, #随机水平偏移幅度0.05
    height_shift_range=0.05,    #随机垂直偏移幅度0.05
    shear_range=0.1,    #剪切强度0.1
    zoom_range=0.1, #随机缩放幅度0.1
)
valid_gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
)

train_generator = train_gen.flow_from_directory('./train',
                                        target_size=(320,480),shuffle=True,
                                        batch_size=64,class_mode="categorical")
valid_generator = valid_gen.flow_from_directory('./valid',
                                        target_size=(320,480),shuffle=True,
                                        batch_size=64,class_mode="categorical")

input_tensor = Input((320, 480, 3))
x = input_tensor
x = Lambda(inception_v3.preprocess_input)(x)

model_incv3 = InceptionV3(input_tensor=x,
                    weights='imagenet', include_top=False)

x = GlobalAveragePooling2D()(model_incv3.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(8, activation='softmax')(x)
model = Model(model_incv3.input, x)
for i in range(172):
    model.layers[i].trainable = False

steps_train_sample = train_generator.samples // 128 + 1
steps_valid_sample = valid_generator.samples // 128 + 1

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=steps_train_sample, epochs=4, 
            validation_data=valid_generator, validation_steps=steps_valid_sample)

model.compile(optimizer=RMSprop(lr=1*0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
cp = ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
model.fit(train_generator, steps_per_epoch=steps_train_sample, epochs=6, 
            validation_data=valid_generator, validation_steps=steps_valid_sample,callbacks=[cp])

print('fit is over')
print('save best is over')