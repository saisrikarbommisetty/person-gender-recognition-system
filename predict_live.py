import cv2
import numpy as np
import tensorflow as tf
import joblib

# ==========================

# Load Person Recognition Model

# ==========================

person_model = tf.keras.models.load_model(
"models/person_classifier.h5"
)

person_encoder = joblib.load(
"models/person_encoder.pkl"
)

# ==========================

# Load Gender Model

# ==========================

gender_net = cv2.dnn.readNet(
"models/gender_net.caffemodel",
"models/gender_deploy.prototxt"
)

gender_list = ["Male", "Female"]

MODEL_MEAN_VALUES = (
78.4263377603,
87.7689143744,
114.895847746
)

# ==========================

# Face Detector

# ==========================

face_detector = cv2.CascadeClassifier(
cv2.data.haarcascades +
"haarcascade_frontalface_default.xml"
)

# ==========================

# Unknown Threshold

# ==========================

UNKNOWN_THRESHOLD = 97

# ==========================

# Start Webcam

# ==========================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        person_face = cv2.resize(
            face,
            (128, 128)
        )

        person_face = person_face.astype(
            "float32"
        ) / 255.0

        person_face = np.expand_dims(
            person_face,
            axis=0
        )

        person_pred = person_model.predict(
            person_face,
            verbose=0
        )

        person_idx = np.argmax(person_pred)

        person_conf = float(
            np.max(person_pred) * 100
        )

        if person_conf >= UNKNOWN_THRESHOLD:
            name = person_encoder.inverse_transform(
                [person_idx]
            )[0]
        else:
            name = "Unknown"

        blob = cv2.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )

        gender_net.setInput(blob)

        gender_preds = gender_net.forward()

        gender_idx = gender_preds[0].argmax()

        gender = gender_list[gender_idx]

        gender_conf = (
            gender_preds[0].max() * 100
        )

        text = (
            f"{name} | "
            f"{gender} | "
            f"Face:{person_conf:.1f}% | "
            f"Gender:{gender_conf:.1f}%"
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Person & Gender Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()