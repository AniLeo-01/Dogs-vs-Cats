import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC

inputs = keras.Input(shape=(128,128,3), name = 'input')
x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
x = layers.Conv2D(64, 3, activation= 'relu', padding= 'same')(x)

batch_normalization = layers.BatchNormalization()(x)
maxpool = layers.MaxPooling2D(strides = (2,2))(batch_normalization)
dropout = layers.Dropout(0.05)(maxpool)

x = layers.Conv2D(128, 3, activation='relu', padding='same')(dropout)
x = layers.Conv2D(128, 3, activation= 'relu', padding= 'same')(x)

batch_normalization = layers.BatchNormalization()(x)
maxpool = layers.MaxPooling2D(strides = (2,2))(batch_normalization)
dropout = layers.Dropout(0.05)(maxpool)

x = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(dropout)
x = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(x)
x = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(x)

batch_normalization = layers.BatchNormalization()(x)
maxpool = layers.MaxPooling2D(strides = (2,2))(batch_normalization)
dropout = layers.Dropout(0.05)(maxpool)

x = layers.Conv2D(512, 3, activation = 'relu', padding= 'same')(dropout)
x = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(x)
x = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(x)

batch_normalization = layers.BatchNormalization()(x)
maxpool = layers.MaxPooling2D(strides = (2,2))(batch_normalization)
dropout = layers.Dropout(0.05)(maxpool)

x = layers.Conv2D(512, 3, activation = 'relu', padding= 'same')(dropout)
x = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(x)
x = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(x)

batch_normalization = layers.BatchNormalization()(x)
maxpool = layers.MaxPooling2D(strides = (2,2))(batch_normalization)
flatten = layers.Flatten()(maxpool)

x = layers.Dense(4096, activation = 'relu')(flatten)
x = layers.Dense(4096, activation = 'relu')(x)
x = layers.Dense(1000, activation = 'relu')(x)

batch_normalization = layers.BatchNormalization()(x)
dropout = layers.Dropout(0.5)(batch_normalization)

outputs = layers.Dense(2, activation='softmax')(dropout)
model = keras.Model(inputs, outputs, name="VGG_custom")
