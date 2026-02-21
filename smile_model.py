from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
from imutils import paths
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout


def build_model(width=96, height=96, depth=3):

    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3,3), padding="same",
                     activation="relu",
                     input_shape=(height, width, depth)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Block 2
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Block 3
    model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    return model

data = []
labels = []
dataset_path = "C:/Users/Sharif/PycharmProjects/PythonProject1/kaggle-genki4k"
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
for ImagePath in sorted(list(paths.list_images(dataset_path))):

    image = cv2.imread(ImagePath)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        continue   # اگر صورت نداشت ردش کن

    # انتخاب بزرگ‌ترین صورت (حرفه‌ای‌تر)
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    (x, y, w, h) = faces[0]

    face = image[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (96, 96))
    face = img_to_array(face)

    data.append(face)

    label = ImagePath.split(os.path.sep)[-2]
    label = 'smiling' if label == 'smile' else 'non_smile'
    labels.append(label)


data = np.array(data, dtype = 'float') / 255
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = le.transform(labels)

classTotals = np.bincount(labels)
classWeight = {}
for i in range(len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

(trainX , testX, trainY ,testY) = train_test_split(data , labels, test_size=0.2 , stratify=labels , random_state=42)
print ('compiling model ...')
model = build_model()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
M = model.fit(trainX,trainY, validation_data= (testX,testY),class_weight=classWeight, batch_size=64, epochs=10)
print('evaluating network...')
predictions = model.predict(testX , batch_size=64)
predictions = (predictions > 0.5).astype("int32")
print(classification_report(testY, predictions))
model.save('smile_model.keras')

#plt.style.use('ggplot')
#plt.figure()
#plt.plot(np.arange(0,15),M.history['loss'],label='train_loss')
#plt.plot(np.arange(0,15),M.history['val_loss'],label='val_loss')
#plt.plot(np.arange(0,15),M.history['accuracy'],label='accuracy')
#plt.plot(np.arange(0,15),M.history['val_accuracy'],label='val_accuracy')
#plt.show()

from tensorflow.keras.models import load_model

print("[INFO] loading model...")
model = load_model("smile_model.keras")

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
cv2.CascadeClassifier()
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        # کراپ صورت
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (96, 96))
        face_resized = face_resized.astype("float") / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        # پیش‌بینی
        prediction = model.predict(face_resized)[0][0]

        label = "Smiling" if prediction > 0.5 else "Not Smiling"
        color = (0,255,0) if prediction > 0.5 else (0,0,255)

        # رسم مستطیل دور صورت
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

        # نمایش متن بالای صورت
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

    cv2.imshow("Smile Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()