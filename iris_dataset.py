from river import metrics, datasets, preprocessing, compose, stream
from sklearn import datasets as sk_datasets
from deep_river import classification
from torch import nn
from torch import optim
from torch import manual_seed
import numpy as np


save_as_file = False
pretraining = True
_ = manual_seed(42)

# Save as file
if save_as_file:
    iris_data = sk_datasets.load_iris()
    print(iris_data.feature_names)
    for i in range(10):
       with open(f'img{i}.npy', 'wb') as f:
           np.save(f, iris_data.data[i])
    with open(f'y.npy', 'wb') as f:
        np.save(f, iris_data.target[:10])

if not save_as_file:
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

    # dataset = datasets.Phishing()

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

    # Import elements in file
    test = np.load(f'img0.npy')
    iris_data = sk_datasets.load_iris()

    for i in range(1, 10):
        test1 = np.load(f'img{i}.npy')
        test = np.vstack((test, test1))

    test_y = np.load('y.npy')
    x = stream.iter_array(test, test_y, feature_names=iris_data.feature_names)
    metric = metrics.Accuracy()

    for i, j in x:
        print(f"{j} {i}")
        y_pred = model_pipeline.predict_one(i)
        metric.update(j, y_pred)
        model_pipeline.learn_one(i, j)

    print(f"Accuracy: {metric.get():.4f}")

