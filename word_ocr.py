import word_ocr_engine as engine
import cv2

# load the classifier
clf = engine.load('word_classifier_knn2.model')

name = "./images/multiline3.jpg"
img = cv2.imread(name)
chars = engine.perform_ocr(img, clf)

# make up the number from the individual digits
# and compute its square
word = ''.join(chars)
word = '\n'.join([engine.get_python_syntax(sub_word) for sub_word in word.split('\n')])
print(word)

# display the information
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, word, (0, 40), font, 1, (255, 0, 0), 2)

cv2.imshow("words", img)
cv2.waitKey(0)