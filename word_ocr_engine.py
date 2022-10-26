from random import gauss
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from distance import levenshtein as diff
from statistics import pvariance, mean


def load(model):
    return joblib.load(model)


def get_bounding_boxes(cnts):

    # get the bounding recangles
    bounding_boxes = [cv2.boundingRect(c)
                      for c in cnts if cv2.contourArea(c) > 300]

    # check the lines
    y_sorted = sorted(bounding_boxes, key=lambda box: box[1])
    lines = []
    line_flag = False
    for i in range(max([(box[1] + box[3]) for box in y_sorted]) + 2):
        flag = False
        for box in y_sorted:
            if (not flag) and (0 <= i - box[1] <= box[3]):
                flag = True
                line_flag = True
                break

        if (not flag) and (line_flag):
            line_flag = False
            new_line = []
            for box in y_sorted:
                if box[1] + box[3] < i:
                    new_line.append(box)

            # removing the line
            for box in new_line:
                y_sorted.remove(box)

            lines.append(sorted(new_line, key = lambda box : box[0]))

    # sort them left to right
    return lines


def predict_digit(rect, img_thrsh, clf):
    PPC = 5

    x, y, w, h = rect

    # expand the rectangle slightly for better recognition

    x -= 5
    w += 10
    y -= 5
    h += 10

    roi = img_thrsh[y: y + h, x: x + w]

    # resize the region of interest to 28x28
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # calculate the HOG features to predict the number
    features = hog(roi,
                   pixels_per_cell=(PPC, PPC),
                   cells_per_block=(1, 1))

    # predict the number
    char = clf.predict([features])
    return chr(96 + char[0])


def perform_ocr(img, clf, write=True):

    # read the file and perform some image processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thrsh = cv2.threshold(
        img_gray_blur, 120, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('thresholded image', cv2.resize(img_thrsh, (800, 800), interpolation = cv2.INTER_AREA))
    cv2.waitKey()
    # find the contours of the digits
    contours, _ = cv2.findContours(
        img_thrsh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get the bounding rectangle for each box
    lines = get_bounding_boxes(contours)

    recognized = []
    for line in lines:
        initial_len = len(recognized)
        for box in line:
            x, y, w, h = box
            x -= 5
            w += 10
            y -= 5
            h += 10
            if write:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # predict the digit inside each box
            char = str(predict_digit(box, img_thrsh, clf))
            recognized.append(char)

            # write the recognized di# sorted(bounding_boxes, key=lambda box: box[0])gits on the image
            if write:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, char, (x, y), font, 0.8, (255, 0, 0), 2)
        
        var_coeff = 1.25
        diff_line = []
        for i in range(1, len(line)):
            diff_line.append(line[i][0] - line[i - 1][0])
        print('line_diff:', diff_line, 'thresh:', mean(diff_line) + var_coeff * (pvariance(diff_line) ** 0.5))
        
        gaussian_thresh = mean(diff_line) + var_coeff * (pvariance(diff_line) ** 0.5)
        for i in range(len(diff_line)):
            if diff_line[i] > gaussian_thresh:
                recognized.insert(initial_len + i + 1, ' ')
                initial_len += 1

        recognized.append('\n')

    return recognized


def get_difference_ratio(reference, candidate):
    diff_count = diff(reference.strip(' \n'), candidate.strip(' \n'))
    return 100 * diff_count / len(reference)


DIFFERENCE_RATIO_THRESHOLD = 60


def get_python_syntax(string):
    db = [x for x in open('./commands.txt').read().split()]
    ratios = []
    for word in db:
        ratio = get_difference_ratio(word, string)
        if (ratio < DIFFERENCE_RATIO_THRESHOLD):
            ratios.append([word, ratio])

    ratios.sort(key=lambda x: x[1])
    return ratios[0][0] if len(ratios) else string
