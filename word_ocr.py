import word_ocr_engine as engine
import cv2

# load the classifier
clf = engine.load('word_classifier_knn2.model')

name = "./images/cmd5.jpg"
img = cv2.imread(name)
chars = engine.perform_ocr(img, clf)

# make up the number from the individual digits
# and compute its square
word = ''.join(chars)
python_instance = engine.get_python_syntax(word)
if(len(python_instance)):
    word = python_instance[0]

# display the information
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, word, (0, 40), font, 1, (0, 255, 255), 2)

cv2.imshow("words", img)
cv2.waitKey(0)