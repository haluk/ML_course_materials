#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

plt.style.use("ggplot")

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

image_shape = train_images.shape[1:]
image_shape

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

plt.figure(figsize=(5, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.savefig("sample_images.png")
plt.clf()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=image_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))


model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)
)


plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.savefig("train_val_acc.png")
plt.clf()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


logits = model.predict(test_images)
preds = tf.math.argmax(tf.nn.softmax(logits), axis=1)


conf_mat = confusion_matrix(test_labels, preds)


conf_mat[4]
conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]


def plot_confusion_matrix(
    cm, classes, fname, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    Plots the confusion matrix.
    """
    import itertools

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "2.2f"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(fname)


plt.figure(figsize=(20, 7))
plot_confusion_matrix(conf_mat, class_names, "confusion_matrix.png")
plt.clf()


def create_model(
    filter_nos=[32, 32, 32],
    filter_size=(3, 3),
    dense_layers=[64],
    input_shape=image_shape,
):
    """
    Args:
        filter_nos (tuple): number of filters in each Conv2D layer
        filter_size (tuple): size of Conv2D filters. Filters are assumed to have uniform sizes across layers.
        input_shape (tuple): shape of the input of the first layer
        dense_layer (tuple): number of hidden units in each Dense layer except the output layer. Output layer is hardcoded to the number of classes.
    """

    model = models.Sequential()

    for i, filter_no in enumerate(filter_nos):
        if i == 0:
            # first Conv2D requires input_shape
            model.add(
                layers.Conv2D(
                    filter_no, filter_size, activation="relu", input_shape=input_shape
                )
            )
        else:
            model.add(layers.Conv2D(filter_no, filter_size, activation="relu"))

        if i + 1 < len(filter_nos):
            model.add(
                layers.MaxPooling2D((2, 2))
            )  # add MaxPooling layer to all but the last convolutional stack

    model.add(layers.Flatten())

    for dense_layer in dense_layers:
        model.add(layers.Dense(dense_layer, activation="relu"))

    model.add(layers.Dense(10))  # output layer

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


model = create_model(filter_nos=(32, 64, 64), filter_size=(3, 3), dense_layers=(64,))
model.summary()


model = KerasClassifier(
    create_model,
    filter_nos=(32, 64, 64),
    filter_size=(3, 3),
    dense_layers=(64,),
    epochs=10,
    batch_size=32,
)


num_folds = 5
kfold = StratifiedKFold(n_splits=num_folds)
fold_accuracies = cross_val_score(
    estimator=model, X=train_images, y=train_labels, cv=kfold
)

print(
    "Accuracy: {:.3f} (+/- {:.3f})".format(
        fold_accuracies.mean(), fold_accuracies.std() * 2
    )
)
