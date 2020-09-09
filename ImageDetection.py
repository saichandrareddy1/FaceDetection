# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:12:07 2020

@author: SUPERMAN
"""
import cv2
import os

def Image(img):
    
    face_cascade = cv2.CascadeClassifier('face_read.xml') 
    
    # reads frames from a camera 
    img = cv2.imread(img)  
      
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
      
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3,10)
      
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
            
        #Saving the detected face in the Live Stream
        cv2.imwrite(os.getcwd()+"\imagedetected.jpg", img[y:y+h, x:x+w])
          
    # Display an image in a window 
    cv2.imshow('img',img) 
    
    cv2.waitKey(0)
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    Image(os.getcwd()+"/image.jpg")