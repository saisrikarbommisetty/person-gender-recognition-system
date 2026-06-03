import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

images = []
labels = []

for person in os.listdir("dataset"):

    person_path = os.path.join(
        "dataset",
        person
    )

    for img_name in os.listdir(
        person_path
    ):

        img_path = os.path.join(
            person_path,
            img_name
        )

        img = cv2.imread(img_path)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = img / 255.0

        images.append(img)

        labels.append(person)

X = np.array(images)

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

joblib.dump(
    encoder,
    "models/person_encoder.pkl"
)

np.save(
    "models/X.npy",
    X
)

np.save(
    "models/y.npy",
    y
)

print("Preprocessing Completed")