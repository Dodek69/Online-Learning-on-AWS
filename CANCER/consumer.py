from kafka import KafkaConsumer
from torch import manual_seed
import torch
import torch.nn as nn
import kagglehub
from river import metrics, compose, stream, preprocessing
from deep_river import classification
import numpy as np
import cv2
import os



PRETRAINING = True
PRETRAINED_ACC = True
IMAGE_SIZE = 320
N_TRAIN_IMAGES = 200
N_TEST_IMAGES = 50

_ = manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = metrics.Accuracy()

def create_dataset(type: str, labels: [str], dataset_path: str, n_images=100):
    l = 0
    X_train = np.array([])
    Y_train = np.array([])
    for label in labels:
        tmp = 0
        for filename in os.listdir(dataset_path + f'/{type}/' + label):
            file_path = dataset_path + f'/{type}/' + label + '/' + filename
            with open(file_path, 'rb') as f:
                image_data = f.read()
            image_data = np.frombuffer(image_data, dtype=np.uint8)
            image_data = (cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)

            image_data = cv2.resize(image_data, (320, 320))
            image_data = image_data.reshape(-1)
            if tmp == 0 and l == 0:
                X_train = image_data
                Y_train = l
            else:
                X_train = np.vstack((X_train, image_data))
                Y_train = np.hstack((Y_train, l))
            tmp += 1
            if tmp >= n_images:
                break
        l += 1
    return X_train, Y_train


# KAFKA CONSUMER SETUP
consumer = KafkaConsumer('image_topic', bootstrap_servers='localhost:9092')

# BUILDING MODEL
class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(IMAGE_SIZE*IMAGE_SIZE, 1024)
        self.dense1 = nn.Linear(1024, 512)
        self.dense2 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, n_features)
        self.dropout = nn.Dropout(0.4)

    def forward(self, X):
        X = nn.functional.relu(self.dense0(X))
        X = nn.functional.relu(self.dense1(X))
        X = self.dropout(X)
        X = nn.functional.relu(self.dense2(X))
        X = self.dropout(X)
        X = nn.functional.relu(self.dense4(X))
        X = nn.functional.softmax(self.dense5(X))
        return X

model_pipeline = compose.Pipeline(
    preprocessing.StandardScaler(),
    classification.Classifier(module=MyModule, loss_fn="binary_cross_entropy", optimizer_fn='sgd')
)

# PRETRAINING

if PRETRAINING:
    # DATASET
    dataset_path = kagglehub.dataset_download("hayder17/breast-cancer-detection")

    labels = ['0', '1']
    X_train, Y_train = create_dataset('train', labels, dataset_path, N_TRAIN_IMAGES)

    train_dataset = stream.iter_array(
        X_train, Y_train, shuffle=True
    )
    print(f"Learning")
    for x, y in train_dataset:
        model_pipeline.learn_one(x, y)

    if PRETRAINED_ACC:
        X_test, Y_test = create_dataset('test', labels, dataset_path, N_TEST_IMAGES)

        test_dataset = stream.iter_array(
            X_test, Y_test, shuffle=True
        )

        print(f"Evaluating")
        for x, y in test_dataset:
            y_pred = model_pipeline.predict_one(x)
            metric.update(y_pred, y)
            # print(model_pipeline.predict_proba_one(x))


print("Waiting")

tmp = 0
for msg in consumer:
    image_data = np.frombuffer(msg.value, dtype=np.uint8)
    image_data = (cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)

    image_data = cv2.resize(image_data, (IMAGE_SIZE, IMAGE_SIZE))
    image_data = image_data.reshape(-1)

    x = stream.iter_array(np.expand_dims(image_data, 0), np.array([tmp%2]))

    for i, j in x:
        y_pred = model_pipeline.predict_one(i)
        metric.update(j, y_pred)
        # print(f"{j} {y_pred}")
        # print(model_pipeline.predict_proba_one(i))
        model_pipeline.learn_one(i, j)

    print(f"Accuracy: {metric.get():.4f}")
    tmp += 1


