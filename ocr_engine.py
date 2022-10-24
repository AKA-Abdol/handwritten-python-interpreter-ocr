import cv2
import numpy as np
from skimage.feature import hog
import joblib
#from sklearn.externals import joblib



def load(model):
    return joblib.load(model)


def get_bounding_boxes(cnts):

    # get the bounding rectangles
    bounding_boxes = [cv2.boundingRect(c)
                      for c in cnts if cv2.contourArea(c) > 300]

    # sort them left to right
    return sorted(bounding_boxes, key=lambda box: box[0])


def predict_digit(rect, img_thrsh, clf):

    x, y, w, h = rect

    # expand the rectangle slightly for better recognition
    x = x - 5
    y = y - 5
    w = w + 10
    h = h + 10

    roi = img_thrsh[y: y + h, x: x + w]

    # resize the region of interest to 20x20
    roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # calculate the HOG features to predict the number
    features = hog(roi,
                   pixels_per_cell=(10, 10),
                   cells_per_block=(1, 1))
    # print (features)

    # predict the number
    digit = clf.predict([features])
    return digit[0]


def perform_ocr(img, clf, write=True):

    # read the file and perform some image processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thrsh = cv2.threshold(img_gray_blur, 90, 255, cv2.THRESH_BINARY_INV)

    # find the contours of the digits
    contours, _ = cv2.findContours(
        img_thrsh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get the bounding rectangle for each box
    boxes = get_bounding_boxes(contours)

    recognized = []
    for box in boxes:
        x, y, w, h = box
        if write:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # predict the digit inside each box
        digit = str(predict_digit(box, img_thrsh, clf))
        recognized.append(digit)

        # write the recognized digits on the image
        if write:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, digit, (x, y), font, 0.8, (255, 0, 0), 2)

    return recognized
