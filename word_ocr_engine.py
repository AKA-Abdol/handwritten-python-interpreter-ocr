import cv2
import numpy as np
from skimage.feature import hog
import joblib


def load(model):
    return joblib.load(model)


def get_bounding_boxes(cnts):

    # get the bounding recangles
    bounding_boxes = [cv2.boundingRect(c)
                      for c in cnts if cv2.contourArea(c) > 300]

    # sort them left to right
    return sorted(bounding_boxes, key=lambda box: box[0])


def predict_digit(rect, img_thrsh, clf):
    PPC = 5

    x, y, w, h = rect

    # expand the rectangle slightly for better recognition
    
    x -= 5
    w += 10
    y -= 5
    h += 10

    roi = img_thrsh[y: y + h, x: x + w]

    # resize the region of interest to 20x20
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # calculate the HOG features to predict the number
    features = hog(roi,
                   pixels_per_cell=(PPC, PPC),
                   cells_per_block=(1, 1))
    # print (features)

    # predict the number
    digit = clf.predict([features])
    return chr(96 + digit[0])


def perform_ocr(img, clf, write=True):

    # read the file and perform some image processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thrsh = cv2.threshold(img_gray_blur, 120, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('bebin', img_thrsh)
    cv2.waitKey()
    # find the contours of the digits
    contours, _ = cv2.findContours(img_thrsh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get the bounding rectangle for each box
    boxes = get_bounding_boxes(contours)

    recognized = []
    for box in boxes:
        x, y, w, h = box
        x -= 5
        w += 10
        y -= 5
        h += 10
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