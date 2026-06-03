import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

X = np.load(
    "models/X.npy"
)

y = np.load(
    "models/y.npy"
)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = tf.keras.Sequential([

tf.keras.layers.Conv2D(
32,
(3,3),
activation="relu",
input_shape=(128,128,3)
),

tf.keras.layers.MaxPooling2D(),

tf.keras.layers.Conv2D(
64,
(3,3),
activation="relu"
),

tf.keras.layers.MaxPooling2D(),

tf.keras.layers.Conv2D(
128,
(3,3),
activation="relu"
),

tf.keras.layers.MaxPooling2D(),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(
128,
activation="relu"
),

tf.keras.layers.Dropout(0.5),

tf.keras.layers.Dense(
len(np.unique(y)),
activation="softmax"
)

])

model.compile(
optimizer="adam",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"]
)

model.fit(
X_train,
y_train,
epochs=20,
batch_size=16,
validation_data=(X_test,y_test)
)

model.save(
"models/person_classifier.h5"
)

print("Model Saved")