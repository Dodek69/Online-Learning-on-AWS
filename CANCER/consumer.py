from kafka import KafkaConsumer
from torch import manual_seed
import torch
import torch.nn.functional as F
import torch.nn as nn
import kagglehub
from river import metrics, compose, stream, preprocessing
from deep_river import classification
import numpy as np
import cv2
import os
import pandas as pd

PRETRAINING = True
PRETRAINED_ACC = True
IMAGE_SIZE = 128
N_TRAIN_IMAGES = 800
N_TEST_IMAGES = 50
N_EPOCHS = 2

_ = manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = metrics.Accuracy()

def create_dataset(type: str, labels: [str], dataset_path: str, n_images=100):
    X_train = []
    Y_train = []
    for label_idx, label in enumerate(labels):
        tmp = 0
        label_path = os.path.join(dataset_path, type, label)
        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)
            print(f"Reading file: {file_path}")
            try:
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                image_data = np.frombuffer(image_data, dtype=np.uint8)
                image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image_data is None:
                    print(f"Error decoding image: {file_path}")
                    continue
                image_data = (image_data / 255.0).astype(np.float32)
                image_data = cv2.resize(image_data, (IMAGE_SIZE, IMAGE_SIZE))
                X_train.append(image_data.flatten())
                Y_train.append(label_idx)
                tmp += 1
                if tmp >= n_images:
                    break
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return np.array(X_train), np.array(Y_train)

# KAFKA CONSUMER SETUP
consumer = KafkaConsumer('image_topic', bootstrap_servers='localhost:9092')

# BUILDING MODEL
class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = None
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        if x.numel() == 0:
            raise ValueError("Input tensor is empty.")
        print(f"Input shape before reshape: {x.shape}")
        x = x.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 64).to(x.device)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

model = classification.Classifier(
    module=MyModule,
    loss_fn='binary_cross_entropy',
    optimizer_fn='adam',
    lr=0.001,
    is_class_incremental=False,
    device=device
)

# PRETRAINING
if PRETRAINING:
    dataset_path = kagglehub.dataset_download("hayder17/breast-cancer-detection")

    labels = ['0', '1']
    X_train, Y_train = create_dataset('train', labels, dataset_path, N_TRAIN_IMAGES)

    print("Learning")
    n_batches = N_TRAIN_IMAGES // 8
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1} / {N_EPOCHS}")
        shuffled_indices = np.random.permutation(N_TRAIN_IMAGES)
        shuffled_x = X_train[shuffled_indices]
        shuffled_y = Y_train[shuffled_indices]
        for i in range(n_batches):
            batch_x = shuffled_x[i * 8:(i + 1) * 8]
            batch_y = shuffled_y[i * 8:(i + 1) * 8]
            if batch_x.size == 0 or batch_y.size == 0:
                print("Empty batch encountered, skipping.")
                continue
            model.learn_many(pd.DataFrame(batch_x), pd.Series(batch_y))

    if PRETRAINED_ACC:
        X_test, Y_test = create_dataset('test', labels, dataset_path, N_TEST_IMAGES)
        print("Evaluating")
        for x, y in stream.iter_array(X_test, Y_test):
            y_pred = model.predict_one(x)
            metric.update(y_pred, y)
        print(f"Accuracy: {metric.get():.2f}")

print("Waiting")

tmp = 0
for msg in consumer:
    print("Analyzing new message")
    image_data = np.frombuffer(msg.value, dtype=np.uint8)
    image_data = (cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)

    image_data = cv2.resize(image_data, (IMAGE_SIZE, IMAGE_SIZE))
    image_data = image_data.reshape(-1)

    x = stream.iter_array(np.expand_dims(image_data, 0), np.array([tmp%2]))

    for i, j in x:
        y_pred = model.predict_one(i)
        metric.update(j, y_pred)
        model.learn_one(i, j)

    print(f"Accuracy: {metric.get():.4f}")
    tmp += 1


