# In[1]:

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from PIL import ImageFile
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import keras
import os
import requests

# define function to load train, test, and validation datasets


def load_dataset(path):
    data = load_files(path)

    condition_files = np.array(data['filenames'])

    condition_targets = np_utils.to_categorical(np.array(data['target']), 2)
    print(condition_targets)
    return condition_files, condition_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset(
    '/Users/Grampun/Desktop/ISIC-Archive-Downloader-master/data_set/training_data')
valid_files, valid_targets = load_dataset(
    '/Users/Grampun/Desktop/ISIC-Archive-Downloader-master/data_set/valid_data')
test_files, test_targets = load_dataset(
    '/Users/Grampun/Desktop/ISIC-Archive-Downloader-master/data_set/test_data')

# load list of labels
condition_names = [item[58:-1] for item in sorted(
    glob("/Users/Grampun/Desktop/ISIC-Archive-Downloader-master/masked_data_set/training_data/*/"))]
print(condition_names)
# print statistics about the dataset
print('There are %d total categories.' % len(condition_names))
print('There are %s total images.\n' %
      len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d validation images.' % len(valid_files))
print('There are %d test images.' % len(test_files))


# In[2]:

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224, 1))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# (IMPLEMENTATION) Model Architecture

# In[4]:

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(224, 224, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.summary()


#  Compile the Model

# In[ ]:
opt = keras.optimizers.Adam(lr=0.0006, beta_1=0.9,
                            beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy'])


epochs = 15

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.6.hdf5',
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=10, callbacks=[checkpointer], verbose=1)


# In[5]:

model.load_weights('weights.best.from_scratch.6.hdf5')


# ### Test the Model

# In[6]:

# get index of predicted label for each image in test set
condition_predictions = [np.argmax(model.predict(
    np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(condition_predictions) ==
                           np.argmax(test_targets, axis=1))/len(condition_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)  # confusion matrix
