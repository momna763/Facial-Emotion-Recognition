import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from tqdm import tqdm
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
warnings.filterwarnings('ignore')
# %matplotlib inline


TRAIN_DIR = '/Users/momna/Desktop/archive_2/train'
TEST_DIR = '/Users/momna/Desktop/archive_2/test'

def load_dataset(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        if label == ".DS_Store":
            continue
        for filename in os.listdir(os.path.join(directory, label)):
            if filename == ".DS_Store":
                continue
            img = cv2.imread(os.path.join(directory, label, filename))
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels


## convert into dataframe
train = pd.DataFrame()
train['image'], train['label'] = load_dataset(TRAIN_DIR)
# shuffle the dataset
train = train.sample(frac=1).reset_index(drop=True)
train.head()

test = pd.DataFrame()
test['image'], test['label'] = load_dataset(TEST_DIR)
test.head()

sns.countplot(train['label'])
from PIL import Image
img = Image.fromarray(train['image'][0])



# to display grid of images
plt.figure(figsize=(20,20))
files = train.iloc[0:25]

for index, image, label in files.itertuples():
    plt.subplot(5, 5, index+1)
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.axis('off')


def extract_features(images):
    features = []
    for image in tqdm(images):
        img_resized = cv2.resize(image, (48, 48))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_resized = np.expand_dims(img_resized, axis=-1)  # Add a channel dimension
        features.append(img_resized)
    features = np.array(features)
    return features




train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

## normalize the image
x_train = train_features/255.0
x_test = test_features/255.0

## convert label to integer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)


y_train[0]

# config
input_shape = (48, 48, 1)
output_class = 7

model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(output_class, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# train the model
history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()

plt.show()

image_index = random.randint(0, len(test))
print("Original Output:", test['label'][image_index])
pred = model.predict(x_test[image_index].reshape(1, 48, 48, 1))
prediction_label = le.inverse_transform([pred.argmax()])[0]
print("Predicted Output:", prediction_label)
plt.imshow(x_test[image_index].reshape(48, 48), cmap='gray')
model.save('IDS_Project_model.h5')
