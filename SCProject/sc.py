import numpy as np
import math
import cv2
import vector
from nn_model import srediSliku, neural_network_model

import sys

#Global variables
#=====================
i = 0
j = 0
ranges = [([230, 230, 230], [255, 255, 255])]
maxLength = 0
pictureSize = 28
allNumbers = []
distance = 20
jezgro = np.ones((2, 2), np.uint8)

#Main funkcija za pokretanje programa
def main():

    global jezgro
    name = 'video-' + str(sys.argv[1]) + '.avi'
    video = cv2.VideoCapture(name)
    #video = cv2.VideoCapture('video-0.avi')
    ret, frame = video.read()

    shape = (28, 28, 1)
    class_number = 10

    klasifikator = neural_network_model(shape , class_number)
    klasifikator.load_weights(''
                              'neuralModel.h5')
    crvena = [6, 19, 216]
    zelena = (0, 255, 0)
    plava = (255, 153, 0)

    frameNum = 0
    allNum = [] # svi brojevi koji su bili na sceni
    sum = 0
    presao = True
    nijePresao = False

    kernel = jezgro
    linija = detektujLiniju(cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel=kernel))

    while ret:
        ret, frame = video.read()

        if not ret:
            break

        minTacka = linija[0]
        maxTacka = linija[1]

        konture = contoursOfNumbers(frame)

        for kontura in konture:

            cx, cy = centerOfPoints(kontura)

            element = {'point': (cx, cy), 'brFrame': frameNum, 'history': []}

            founded = detectNumbers(allNum, element)

            if len(founded) == 0:
                element['value'] = recognizeNumber(frame, kontura, klasifikator)
                element['presaoLiniju'] = nijePresao
                allNum.append(element)

            elif len(founded) == 1:
                i = founded[0]
                histo = {'brFrame': frameNum, 'point': element['point']}

                allNum[i]['history'].append(histo)

                allNum[i]['brFrame'] = frameNum

                allNum[i]['point'] = element['point']

            #ispitujemo da li je linija predjena
        for element in allNum:

            subb = frameNum - element['brFrame']
            rast = 3

            if (subb > rast):
                continue
            if not element['presaoLiniju']:

                distanca, _, r = vector.pnt2line(element['point'], minTacka, maxTacka)

                if r == 1 and distanca < 11.0:
                    #saberi brojeve
                    brojevi = element['value']
                    sum += int(brojevi)
                    element['presaoLiniju'] = presao


            cv2.circle(frame, element['point'], 18,crvena, 2)

            cv2.putText(frame, str(element['value']), (element['point'][0] + 12, element['point'][1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,zelena, 3)


            #Ispis teksta na ekranu
            #===============================================================================
            cv2.putText(frame,"Stefan Milovic RA164/2014", (15, 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, plava, 1)

            text = 'Sum: '
            cv2.putText(frame, text + str(sum), (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,plava, 1)

            text1 = 'Number of curent frame: '
            cv2.putText(frame, text1 + str(frameNum), (15, 42), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,plava, 1)


            for history in element['history']:
                sub = frameNum - history['brFrame']
                dis = 70
                if (sub < dis):
                    cv2.circle(frame, history['point'], 1, (200, 200, 200), 1)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == 13:
            break
        counter = 1
        frameNum += counter

    cv2.destroyAllWindows()
    video.release()

    print ("Ukupan broj frejmova:", (frameNum))
    print ("Ukupan zbir  brojeva:", str(sum))

    f = open('out.txt', 'a')
    f.write('\n' + name + '\t' + str(sum))
    f.close()

#----------Aid Functions--------------
#=====================================
#Function which convert RGB to HSV
def convertImageToHSV(picture):
    return cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)

#Function which convert RGB to Gray
def convertImageToGray(picture):
    return cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

#Function which smoothing image
def blurImage(picture):
    kernelSize = 5
    return cv2.GaussianBlur(picture, (kernelSize, kernelSize), 0)

#Function for threshold image
def treshImage(frame,thr,maxValue):

    ret,image=cv2.threshold(frame, thr, maxValue, cv2.THRESH_BINARY);

    return ret, image;

#Function which calculate Euclidian folds
def euclidianDistance(x,x1,y,y1):
    return math.sqrt((x-x1)**2 + (y-y1)**2)

#function which calculate center of point
def centerOfPoints(kontura):

    half = 2
    (pointx, pointy, width, height) = kontura
    centarX = int(pointx + width / half)
    centarY = int(pointy + height / half)
    #print ("Ispisi koordinate x: ", centarX)
    #print ("Ispisi koordinate y: ", centarY)
    centerX = int(centarX)
    centerY = int(centarY)

    return centerX, centerY

#Function for copy
def kopiraj(parameter):
    return parameter.copy()

#Dilate image
def dilacija(picture):
    global jezgro
    return cv2.dilate(picture, jezgro, iterations=1)

#Erode image
def erozija(picture):
    global jezgro
    return cv2.erode(picture, jezgro, iterations=1)

#Functio which resize picture to 28x28 for MNIST dataset
def resizeImage(image):
    global pictureSize
    return cv2.resize(image, (pictureSize,pictureSize), interpolation=cv2.INTER_AREA)

def caluclateMoment(image):
    return cv2.moments(image)

#Function which reshape image
def preoblikuj(image):
    imageSize = 28
    return image.reshape(1, imageSize, imageSize, 1)

#Function which read video
def ucitaj_video(putanja):
    video = cv2.VideoCapture(putanja)
    return video

#============ Test Functions ========================================================

#Funkcija koja vraca koordinate linije samo za prvi frame
def getLineCoordinates():

    cap = ucitaj_video('video-0.avi')
    # setuje CV_CAP_PROP_POS_FRAMES na taj frame
    cap.set(1, 0)
    _, frame = cap.read()

    linija = detektujLiniju(frame)
    print (linija)

    return linija

#Funkcija kojom crtamo konture
def drawingConturs(image):
    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        return cv2.drawContours(image, contour, -1, (0, 255, 0), 3)

#Funkcija kojom ispitujemo da li je element presao liniju
def presaoLiniju(number):

    if(number['presaoLiniju']==True):

            print ("Broj je presao liniju")

            return True
    else:

        return False

#========= Main Functions ============================================================

def detektujLiniju(picture):

    #global  i
    #global  j
    global maxLength

    lower_range = [100, 50, 50]
    upper_range = [130, 255, 255]

    #Dobijamo koordinate linije za prvi frame
    #cap = cv2.VideoCapture('video-0.avi')
    #cap.set(1, 0)
    #_, frame = cap.read()

    # Temena koja sacinjavaju koordinate trazene linije - A(x1,y1) i B(x2,y2)
    dobijena_temena = [(0, 0), (0, 0)]

    # Konvertujemo BGR(Blue-Green-Read) u HSV(Hue-Saturation-Value)
    hsv = convertImageToHSV(picture)

    # definisemo opseg plave boje u HSV-u
    donja_granica_plave = np.array(lower_range, dtype="uint8")
    gornja_granica_plave = np.array(upper_range, dtype="uint8")

    # funkcija putem koje isticemo plavi element slike - plavu liniju
    mask = cv2.inRange(hsv, donja_granica_plave, gornja_granica_plave)

    minValue = 50  # 100
    maxValue = 150  # 200
    L2gradient = None
    aperture_size = 3

    # Canny transformaciju koristimo za dobijanje ivica linije
    ivice = cv2.Canny(mask, minValue, maxValue)

    rho = 1
    theta = np.pi / 180
    treshold = 50
    l2grad = None
    minLineLength = 50
    maxLineGap = 10

    # primenom Hough transformacije docicemo do temena linije
    # HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
    # temena_linije = cv2.HoughLinesP(ivice,rho,theta,treshold,minLineLength)

    temena_linije = cv2.HoughLinesP(ivice, rho, theta, treshold, l2grad, minLineLength, maxLineGap)

    if temena_linije is not None:
        for i in range(0, len(temena_linije)):

            x1,x2,y1,y2 = temena_linije[i][0]

            # duzinu linije racunamo na osnovu formule Euklidskog prostora[sqrt((X-X1)^2 +  (Y-Y1)^2))]
            lineLength = euclidianDistance(y1,x1,y2,x2)

            if maxLength < lineLength:

                dobijena_temena[0] = (x1, x2)  # teme X(x1,x2)
                dobijena_temena[1] = (y1, y2)  # teme Y(y1,y2)
                maxLength = lineLength

                #cv2.line(picture, dobijena_temena[0], dobijena_temena[1], (0, 0, 255), 2)

    return dobijena_temena


#Contours of numbers
def contoursOfNumbers(picture):

    number_contours = []  # niz u koji cemo smestiti dobijene konture

    #Opseg bele boje kojom su pretstavljeni brojevi,cije konture trazimo
    global ranges

    (lower_range, upper_range) = ranges[0]

    # definisemo opseg bele boje kojom su ispisani brojevi
    lower_range = np.array(lower_range)  #donja granica bele
    upper_range = np.array(upper_range)  #gornja granica bele

    # funkcija putem koje isticemo beli element slike - broj
    mask = cv2.inRange(picture, lower_range, upper_range)

    # Bitwise-AND(erosion) za uklanjanje sumova
    input_image = cv2.bitwise_and(picture, picture, mask=mask)

    #Konvertujemo sliku u sivu radi lakse manipulacije
    gray = convertImageToGray(input_image)

    #Uglacavamo povrsinu - pokusavamo da smanjimo sumove sto vise
    blur = blurImage(gray)

    image_original = kopiraj(blur)

    #funkcija putem koje pronalazimo konture na slici
    _, contours, _ = cv2.findContours(image_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        #cv2.drawContours(picture, contour, -1, (0, 255, 0), 3)

        (pointX, pointY, width, height) = cv2.boundingRect(contour) #x,y-the top-left coordinate of rectangle , w-width , h-height

        points = (pointX, pointY, width, height)

        number_contours.append(points)

        #Prikaz frejma i maske
        #cv2.imshow("Frame", picture)
        #cv2.imshow("Mask", mask)
        #print(number_contours)

    return number_contours


#Recognize the number using MNIST dataset which contains 60000 sort of numbers
def recognizeNumber(picture, contour, classifier):

    thresh = 128
    maxValue = 255
    size = 12

    cx, cy = centerOfPoints(contour)

    gray = convertImageToGray(picture)

    number = gray[cy-size:cy+size, cx-size:cx+size]

    ret, number = cv2.threshold(number, thresh, maxValue, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    opening = cv2.morphologyEx(number, cv2.MORPH_OPEN,  np.ones((1, 1), np.uint8))
    dilation = dilacija(opening)
    erosion = erozija(opening)

    number = srediSliku(number)

    predictedNumber = classifier.predict_classes(preoblikuj(number))

    num = int(predictedNumber)

    return num


#Function which use to detect numbers which get across line
def detectNumbers(numbers ,number):

    global distance
    allNumbers = []
    #(x, y) = number.center  # koristice nam za formulu Euklidskog rastojanja
    (x, y) = number['point'] #centar konture - broja

    for rbr, num in enumerate(numbers):
        #(x1, y1) = num.center
        (x1, y1) = num['point']

        #euclidianFolds = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)  # formula za dobijanje Euklidskog rastojanja
        euclidianFolds = euclidianDistance(x,x1,y,y1)

        if euclidianFolds < distance:
            allNumbers.append(rbr)

    return allNumbers


if __name__ == "__main__":
    main()
