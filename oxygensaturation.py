import numpy as np
import math
import cv2
import face_recognition
def decodeYUV420SPtoRedBlueGreenSum(yuv420sp, width, height, type):
        frameSize = width * height

        sum=0
        sumr = 0
        sumg = 0
        sumb = 0
        for i in range(0,height):
            for j in range(0,width):
                  sumr+=yuv420sp[j,i,0]
                  sumg+=yuv420sp[j,i,1]
                  sumb+=yuv420sp[j,i,2]
        if type==1:
            sum = sumr/frameSize
        elif type==2:
            sum=sumg/frameSize
        else:
            sum=sumb/frameSize  
        return sum



'''from ctypes.wintypes import RGB
import cv2
 
# path
path = r'D:\\istockphoto-692423386-170667a.jpg'
 
# Using cv2.imread() method
# Using 0 to read image in grayscale mode
img = cv2.imread(path,1)

def RGB2YUV( rgb ):
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv
im= RGB2YUV(img)
print(img[20,20,0])
height = img.shape[0]
width = img.shape[1]
print(img.shape)
red =decodeYUV420SPtoRedBlueGreenSum(img, height,width,2)
print(red)'''



##########################


def onPreviewFrame():

    size =cv2.VideoCapture(0,cv2.CAP_DSHOW)
    RedAvg=0
    RedAvgList=[]
    BlueAvgList=[]
    sumred=0
    sumblue=0
    counter=0
    while True:
        success, frame = size.read()
        cv2.imshow('frame', frame)

        
           #//put width + height of the camera inside the variables'''
        width = int(size.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(size.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        RedAvg = decodeYUV420SPtoRedBlueGreenSum(frame, height, width, 1)
        sumred = sumred + RedAvg
        BlueAvg = decodeYUV420SPtoRedBlueGreenSum(frame, height, width, 2)
        sumblue = sumblue + BlueAvg

        RedAvgList.append(RedAvg)
        BlueAvgList.append(BlueAvg)

        counter+=1

        Red = np.array(RedAvgList)
        Blue = np.array(BlueAvgList)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

   
    meanr = sumred / counter
    meanb = sumblue / counter
    Stdb=0
    Stdr=0
    for i in range(counter-1): 
        bufferb = Blue[i]

        Stdb =Stdb+((bufferb - meanb) * (bufferb - meanb))

        bufferr = Red[i]

        Stdr = Stdr + ((bufferr - meanr) * (bufferr - meanr))

    varr = math.sqrt(Stdr / (counter - 1))
    varb = math.sqrt(Stdb / (counter - 1))

    R = (varr / meanr) / (varb / meanb)

         
          

    return  100 - 5 * R
a=onPreviewFrame()
print(a)
