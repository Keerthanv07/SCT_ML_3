import zipfile
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage.feature import hog
import joblib 
zip_path = 'dogs-vs-cats2.zip'
extract_path = 'dogs-vs-cats2'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

data_folder = os.path.join(extract_path, 'dogs-vs-cats2', 'train2')

images = []
labels = []

print("Loading and processing images...")
for filename in tqdm(os.listdir(data_folder)):
    if filename.endswith('.jpg'):
        label = 0 if 'cat' in filename else 1 
        img_path = os.path.join(data_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0 
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        images.append(hog_features)
        labels.append(label)

X = np.array(images)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
model_filename = 'svm_cat_dog_model.joblib'
joblib.dump(svm, model_filename)
print(f"\nModel saved as '{model_filename}'")