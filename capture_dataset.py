import cv2
import os
import pandas as pd

name = input("Enter Name: ")

folder = f"dataset/{name}"

os.makedirs(folder, exist_ok=True)

csv_file = "data/person_details.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=["Name"])

if name not in df["Name"].values:
    df.loc[len(df)] = [name]
    df.to_csv(csv_file, index=False)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x,y,w,h) in faces:

        face = frame[y:y+h,x:x+w]

        face = cv2.resize(
            face,
            (128,128)
        )

        count += 1

        cv2.imwrite(
            f"{folder}/{count}.jpg",
            face
        )

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        print(
            f"Captured {count}/50"
        )

    cv2.imshow(
        "Capture Dataset",
        frame
    )

    if count >= 50:
        break

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset Created")