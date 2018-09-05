import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import cv2

#Global variables
#=====================
pictureSize = 28

#----------Aid Functions--------------
#=====================================
def caluclateMoment(image):
    return cv2.moments(image)

def caluclateCentralMoment(moment):
    m1 = 'mu11'
    m2 = 'mu02'
    return (moment[m1] / moment[m2])

def getMaxArea(area):
    return np.argmax(area)

#Functio which resize picture to 28x28 for MNIST dataset
def resizeImage(image):
    global pictureSize
    return cv2.resize(image, (pictureSize,pictureSize), interpolation=cv2.INTER_AREA)

#============== Main Functions ======================

#Model preuzet sa interneta
#Source: https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
def neural_network_model(shape, class_number=10):

    model = Sequential()

    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=shape))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_number, activation='softmax'))

    return model


#Function which crop and deskew image
def srediSliku(number):

    thr = 128
    maxValue = 255

    #Deo koji se tice secenja broja
    ret, thresh = cv2.threshold(number, thr, maxValue, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv2.contourArea)

    pointX, pointY, width, height = cv2.boundingRect(contour)

    xLeft = pointX
    yLeft = pointY
    xRight = pointX + width
    yRight = pointY + height

    #Secemo sliku
    cropPicture = number[yLeft:yRight, xLeft:xRight]

    #Menjamo velicinu slike na 28x29 zbog uporedjivanja sa MNIST dataset-om
    cropPicture = resizeImage(cropPicture)

    return cropPicture

