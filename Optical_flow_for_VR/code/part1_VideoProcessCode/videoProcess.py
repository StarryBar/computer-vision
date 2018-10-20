import numpy as np 
import cv2

def videoProcess(fileName,sampleInterval):
    cap = cv2.VideoCapture(fileName)  
    flag = False
    '''while cv2.waitKey(30)!=ord('q'):
        retval, image = cap.read()
        cv2.imshow("video",image)
        cap.release()'''
    #while(cap.isOpened()):  
    '''while cv2.waitKey(30)!=ord('q'):
        ret, frame = cap.read()
        if flag == False:
            print(type(frame))
            #cv2.imshow('image', frame)
        flag = True
    cap.release()'''
    success,image = cap.read()
    count = 0
    index = 0
    while success:
        if count%sampleInterval == 0:
            cv2.imwrite('/Users/KunyuYe/Desktop/Frame/%d.jpg' % index,image)
            index += 1
        success,image = cap.read()
        count = count + 1
    '''
    while(cap.isOpened()):
        ret, frame = cap.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(type(frame))
        cv2.imshow('image',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()'''

videoProcess('/Users/KunyuYe/Desktop/11.mp4',5)