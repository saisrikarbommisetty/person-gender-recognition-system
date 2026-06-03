import cv2

net = cv2.dnn.readNet(
    "models/gender_net.caffemodel",
    "models/gender_deploy.prototxt"
)

print("Empty:", net.empty())