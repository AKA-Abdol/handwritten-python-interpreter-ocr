import idx2numpy
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

x_train = idx2numpy.convert_from_file('./emnist-letters-train-images-idx3-ubyte')
x_train_labels = idx2numpy.convert_from_file('./emnist-letters-train-labels-idx1-ubyte')
x_test = idx2numpy.convert_from_file('./emnist-letters-test-images-idx3-ubyte')
x_test_labels = idx2numpy.convert_from_file('./emnist-letters-test-labels-idx1-ubyte')
#cv2.imshow('demo', cv2.rotate(cv2.resize(cv2.flip(x_train[a], 0), (100, 100)), cv2.cv2.ROTATE_90_CLOCKWISE))
#cv2.waitKey()
#print(x_train[0].shape)

PPC = 5
x_train_feature = []
for img in x_train:
    revised_img = cv2.rotate(cv2.flip(img, 0), cv2.cv2.ROTATE_90_CLOCKWISE)
    feature = hog(revised_img,
                  pixels_per_cell = (PPC, PPC),
                  cells_per_block = (1, 1))
    x_train_feature.append(feature)
x_train_feature = np.array(x_train_feature, 'float64')

x_test_feature = []
for img in x_test:
    revised_img = cv2.rotate(cv2.flip(img, 0), cv2.cv2.ROTATE_90_CLOCKWISE)
    feature = hog(revised_img,
                  pixels_per_cell = (PPC, PPC),
                  cells_per_block = (1, 1))
    x_test_feature.append(feature)
x_test_feature = np.array(x_test_feature, 'float64')

clf = KNeighborsClassifier()
clf.fit(x_train_feature, x_train_labels)


print("Accuracy:", accuracy_score(x_test_labels, clf.predict(x_test_feature)))

joblib.dump(clf, 'word_classifier_knn3.model')