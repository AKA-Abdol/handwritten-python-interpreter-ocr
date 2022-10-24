import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


# read the image
img = cv2.imread("dataset/digits.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# break up the image into individual digits
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]


# compute the HOG features for each digit and make a dataset
dataset = []
for i in range(50):
    for j in range(100):
        feature = hog(cells[i][j],
                      pixels_per_cell=(10, 10),
                      cells_per_block=(1, 1))
        dataset.append(feature)
dataset = np.array(dataset, 'float64')


# make labels for all 5000 digits
labels = []
for i in range(10):
    labels = labels + [i] * 500


# use the KNN algorithm for training
clf = KNeighborsClassifier()
x_train, x_test, y_train, y_test = tts(dataset, labels, test_size=0.3)

# fit the data into the classifier
clf.fit(x_train, y_train)


# print the accuracy score
print("Accuracy:", accuracy_score(y_test, clf.predict(x_test)))

# save the classifier in a file called digit_classifier_knn.model
joblib.dump(clf, 'digit_classifier_knn.model')
