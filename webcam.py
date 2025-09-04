import cv2
import numpy as np

# Load model files
ageProto = r"C:\Users\Srilekha\Downloads\webcam\age_deploy (5).prototxt"
ageModel = r"C:\Users\Srilekha\Downloads\webcam\age_net (4).caffemodel"
genderProto = r"C:\Users\Srilekha\Downloads\webcam\gender_deploy (4).prototxt"
genderModel = r"C:\Users\Srilekha\Downloads\webcam\gender_net (5).caffemodel"

# Load the models
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.426, 87.768, 114.895)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Initialize webcam
cap = cv2.VideoCapture(0)
padding = 20

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                     MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        genderNet.setInput(blob)
        gender = genderList[genderNet.forward().argmax()]

        # Predict Age
        ageNet.setInput(blob)
        age = ageList[ageNet.forward().argmax()]

        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Age-Gender Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
