#################################################################### MQTT Communication ###############################################################
import os
import subprocess
import paho.mqtt.publish as publish
import picamera
import time
import urllib
import re
import requests
import RPi.GPIO as GPIO  #Importing all the required libraries
import serial
import time
import sys
#import Adafruit_DHT
import numpy as np
import cv2
while(1):
	#os.system('webcam1/./webcam1.sh')
	#a=cv2.imread('still.jpg')           #reading an image from file
 	#print("image captured")
	#a=cv2.resize(im,None,fx=0.15,fy=0.15,interpolation=cv2.INTER_AREA)  #Reducing the size of an image
	#hsv=cv2.cvtColor(a,cv2.COLOR_BGR2HSV) #converting image to hsv
	#image_mask=cv2.inRange(hsv,np.array([20,50,50]),np.array([100,255,255])) # applying masking to detect the green colour 
	#out=cv2.bitwise_and(a,a,mask=image_mask) # performing and operation with image and mask to show green area of image
	#kernel = np.ones((5,5),np.uint8)
	#dilation = cv2.dilate(image_mask,kernel,iterations = 2) #dilating the making image
	#dil=dilation.ravel() # to convert the multidimensional array to the 1d array
	#count = 0
	#for x in np.nditer(dil):
	#	if (x==255):
        #        	count=count+1  # number of white pixels
	#total=float(np.size(dil)) # total number of pixels 
	#c=float(count) 
	#p=c/total 
	#per=p*100
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(11,GPIO.OUT)
	GPIO.output(11,True)
	time.sleep(1)
	GPIO.output(11,False)
	ser=serial.Serial('/dev/ttyACM0')                 # to connect arduino with raspberry pi
	a=ser.readline()
	print(a)
	per = 63.2556476
	wind=requests.get("https://api.thingspeak.com/apps/thinghttp/send_request?api_key=RVRML8PD68G61SIB")
	list1=str("1")+str(",")+str(wind.text)+","+str(per)+","+str(a)
	#list1 = []
	#list1.append(per)
	#list1.append(a)
	#data = list(str(per),str(a))
	print(list1)
	#print('pergreen'+str(per))
	#print('moisture'+str(a))
	#time.sleep(4)
        ser.write("end")
        MQTT_SERVER = "192.168.43.177"
        MQTT_PATH1 = "green_per3"
        #MQTT_PATH2 = "moisture/waterlevel"

        publish.single(MQTT_PATH1, str(list1), hostname=MQTT_SERVER)
        #publish.single(MQTT_PATH2, str(a), hostname=MQTT_SERVER)
	print("###############data sent to main node################")
	#for k in range(0,5):
	#	print("waiting for"+' '+str(k)+' '+'second')
	#	time.sleep(1)
GPIO.cleanup()

