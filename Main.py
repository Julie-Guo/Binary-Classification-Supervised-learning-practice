import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Reading the CSV file
df = pd.read_csv(
    "/Users/zhipingguo/Desktop/Binary Classification Supervised learning practice/telescope_data.csv"
)

# Dropping the first column
df = df.drop(df.columns[0], axis=1)

# Creating a separate Series for the 'class' column transformed to 0s and 1s
class_series = (df["class"] == "g").astype(int)

# Iterating over columns except the 'class' column
for label in df.columns[:-1]:
    plt.hist(
        df[df["class"] == "g"][label],
        color="blue",
        label="gamma",
        alpha=0.7,
        density=True,
    )
    plt.hist(
        df[df["class"] != "g"][label],
        color="red",
        label="hardron",
        alpha=0.7,
        density=True,
    )
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


# split dataset into train, validation, test datasets
train, validation, test = np.split(
    df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))]
)


# scale dataset so each feature value are not too far off from each other, also over sample train data for "hadron" class as it's lenth is much smaller vs "gamma" class
def scale_dataset(dataframe, feature_names, target_name, oversample=False):
    x = dataframe[feature_names].values
    y = dataframe[target_name].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y



feature_names = df.columns[:-1]  # All column names except the target
target_name = df.columns[-1]  # The target column name

train, x_train, y_train = scale_dataset(
    train, feature_names, target_name, oversample=True
)
print(x_train.dtype)



validation, x_validation, y_validation = scale_dataset(
    validation, feature_names, target_name, oversample=False
)
test, x_test, y_test = scale_dataset(test, feature_names, target_name, oversample=False)


# K- Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)

print(classification_report(y_test, y_pred))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model = nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_test)
print(classification_report(y_test, y_pred))


# Logistic Regression ( Sigmoid function)

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train, y_train)

y_pred = lg_model.predict(x_test)
print(classification_report(y_test, y_pred))


# Support Vector Machine ( find hyperplane between 0 and 1, binary classification. Use kernal trick to re-arrange 0 and 1 in order to find hyperplane)

from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)
print(classification_report(y_test, y_pred))


# Neural Net

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history,history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary crossentropy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(history):
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history,history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

nn_model = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)), tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(1, activation="sigmoid")])

nn_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

history = nn_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_validation, y_validation), verbose=0)

plot_loss(history)
plot_accuracy(history)








def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history["loss"], label="loss")
    ax1.plot(history.history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary crossentropy")
    ax1.grid(True)

    ax2.plot(history.history["accuracy"], label="accuracy")
    ax2.plot(history.history["val_accuracy"], label="val_accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.show()


def train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    # Create the model
    nn_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                num_nodes, activation="relu", input_shape=(x_train.shape[1],)
            ),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(num_nodes, activation="relu"),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the model
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    history = nn_model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_validation, y_validation)
        verbose=0,
    )

    return nn_model, history


least_val_loss = float("inf")
least_loss_model = None
epochs = 100
for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.001, 0.005, 0.01]:
            for batch_size in [32, 64, 128]:
                print(
                    f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch_size {batch_size}"
                )
                model, history = train_model(
                    x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs
                )
                plot_history(history)
                val_loss = model.evaluate(x_validation, y_validation)[0]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model


# couldn't debug the above last code block ( something to do with incompatibility with tendor string vs tensor float)
