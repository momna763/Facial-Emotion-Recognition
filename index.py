import cv2
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('IDS_Project_model.h5')

TRAIN_DIR = '/Users/momna/Desktop/archive_2/train'
TEST_DIR = '/Users/momna/Desktop/archive_2/test'

# Define the mapping of class indices to emotion labels
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Function to predict emotion from an image array
def predict_emotion(img):
    # Preprocess the image
    img_resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_resized = np.expand_dims(img_resized, axis=-1)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict the emotion
    pred = model.predict(img_input)
    emotion_index = np.argmax(pred)
    emotion_label = emotion_labels[emotion_index]
    return emotion_label

# Get the list of subdirectories in the test directory
subdirectories = [subdir for subdir in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, subdir))]

# Randomly select a subdirectory
selected_subdir = random.choice(subdirectories)

# Get the list of images in the selected subdirectory
image_files = os.listdir(os.path.join(TEST_DIR, selected_subdir))

# Randomly select an image from the selected subdirectory
selected_image_file = random.choice(image_files)

# Load the selected image
# Load the selected image
image_path = os.path.join(TEST_DIR, selected_subdir, selected_image_file)
img = cv2.imread(image_path)

# Check if img is None or not
if img is None:
    print(f"Error: Unable to load image from path: {image_path}")
else:
    print(f"Image loaded successfully, shape: {img.shape}, dtype: {img.dtype}")

# Predict the emotion
predicted_emotion = predict_emotion(img)

# Display the image along with the predicted emotion
plt.figure()
if img is not None:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Emotion: {predicted_emotion}\n")
    plt.axis('off')
    plt.show()


TRAIN_DIR = '/Users/momna/Desktop/archive_2/train'
TEST_DIR = '/Users/momna/Desktop/archive_2/train'

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



# Display the image along with the predicted emotion
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Emotion: {predicted_emotion}\nImage Path: {image_path}")
plt.axis('off')
plt.show()
