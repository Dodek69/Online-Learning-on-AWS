from kafka import KafkaConsumer
from river import preprocessing, compose
import numpy as np
import pandas as pd
from river import metrics, datasets, preprocessing, compose, stream
from sklearn import datasets as sk_datasets
from deep_river import classification
from torch import nn
from torch import manual_seed
import numpy as np
import io

pretraining = True
_ = manual_seed(42)

# Kafka consumer setup
consumer = KafkaConsumer('image_topic', bootstrap_servers='localhost:9092')

# Preprocessing pipeline for image classification (example: using standard scaler)
preprocessor = compose.TransformerUnion(('scaler', preprocessing.StandardScaler()))


#CHANGE THIS NAIVE BAYES CLASSIFIER WITH RESNET FROM PYTLARZ ET AL.
class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()
        self.dense0 = nn.Linear(n_features, 128)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 3)

    def forward(self, X, **kwargs):
        X = self.relu(self.dense0(X))
        X = self.relu(self.dense1(X))
        X = self.dense2(X)
        return X

model_pipeline = compose.Pipeline(
    preprocessing.StandardScaler(),
    classification.Classifier(module=MyModule, loss_fn=nn.CrossEntropyLoss(), optimizer_fn='adam')
)

dataset = stream.iter_sklearn_dataset(
    dataset=sk_datasets.load_iris(),
    shuffle=True,
    seed=42
)

if pretraining:
    # Pretraining
    metric = metrics.Accuracy()
    for x, y in dataset:
        y_pred = model_pipeline.predict_one(x)
        print(f"{y} {y_pred}")
        metric.update(y, y_pred)
        model_pipeline.learn_one(x, y)

    print(f"Accuracy: {metric.get():.4f}")


# Consume messages and classify images
iris_data = sk_datasets.load_iris()
for msg in consumer:
    image_data = msg.value
    binary_file = io.BytesIO(msg.value)
    image_data = np.load(binary_file)

    # Preprocess image data (example: scale)
    x = stream.iter_array(np.expand_dims(image_data, 0), np.array([0]), feature_names=iris_data.feature_names)
    # metric = metrics.Accuracy()

    for i, j in x:
        print(f"{j} {i}")
        y_pred = model_pipeline.predict_one(i)
        metric.update(j, y_pred)
        model_pipeline.learn_one(i, j)

    print(f"Accuracy: {metric.get():.4f}")


