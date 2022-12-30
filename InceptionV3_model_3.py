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
from keras.callbacks import ModelCheckpoint


originalModelPath = './11.h5'
save_path = './11.27/model/4_2.h5'

train_gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,	
    rotation_range=10.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
)
valid_gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
)

train_generator = train_gen.flow_from_directory('./11.27/train',
										target_size=(320,480),shuffle=True,
										batch_size=64,class_mode="categorical")
valid_generator = valid_gen.flow_from_directory('./11.27/valid',
										target_size=(320,480),shuffle=True,
										batch_size=64,class_mode="categorical")

model = load_model(originalModelPath)

steps_train_sample = train_generator.samples // 128 + 1
steps_valid_sample = valid_generator.samples // 128 + 1

for i in range(172):
    model.layers[i].trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=steps_train_sample, epochs=6, 
			validation_data=valid_generator, validation_steps=steps_valid_sample)

for i in range(172):
    model.layers[i].trainable = True
model.compile(optimizer=RMSprop(lr=1*0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
cp = ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
model.fit(train_generator, steps_per_epoch=steps_train_sample, epochs=15, 
			validation_data=valid_generator, validation_steps=steps_valid_sample,callbacks=[cp])

print('save best is over')