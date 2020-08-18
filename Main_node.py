##############################################Code to Record Dataset#############################
import paho.mqtt.client as mqtt
import os
import json
import subprocess
import time
import urllib
import re
import requests
import RPi.GPIO as GPIO  #Importing all the required libraries
import serial
import time
import sys
import Adafruit_DHT
import numpy as np
import cv2
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
column=['time','temp','humidity','relay','rain','green_per','water','moisture','node','wind']
df=pd.read_csv(r'/var/www/html/dht.csv',header=None,names=column)
df.drop(['time'],1,inplace=True)
df.drop(['node'],1,inplace=True)
df2=df.mask(df.astype(object).eq('None')).dropna()
df2 = shuffle(df2)
df2[df2.relay != 'relay']
X=np.array(df2.drop(['relay'],axis=1))
y=np.array(df2['relay'])
X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2)
#print('test'+str(y_test.shape))
print(X_train)
print(y_train)
#print('train'+str(y_train.shape))
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
predict1=clf.predict(X_train)
while(1):
	MQTT_SERVER1 = "192.168.43.177"
	#MQTT_SERVER2 = "192.168.43.56"                         #######connecting to node 1#########
	MQTT_PATH1 = "green_per3"
	MQTT_PATH2 = "green_per2"
        #MQTT_PATH2 = "moisture/waterlevel"
        #MQTT_PATH2 = "moisture/waterlevel"
        def on_connect(client, userdata, flags, rc):
            print("Connected with result code "+str(rc))
            client.subscribe(MQTT_PATH1)
	    client.subscribe(MQTT_PATH2)
            #client.subscribe(MQTT_PATH2)
            print("connected")
	#elapsedTime = time.time()
	os.system('./webcam.sh')
	a=cv2.imread('still.jpg')           #reading an image from file
	print("image captured")
	#a=cv2.resize(im,None,fx=0.15,fy=0.15,interpolation=cv2.INTER_AREA)  #Reducing the size of an image
	hsv=cv2.cvtColor(a,cv2.COLOR_BGR2HSV) #converting image to hsv
	image_mask=cv2.inRange(hsv,np.array([20,50,50]),np.array([100,255,255])) # applying masking to detect the green colour 
	out=cv2.bitwise_and(a,a,mask=image_mask) # performing and operation with image and mask to show green area of image
	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(image_mask,kernel,iterations = 2) #dilating the making image
	dil=dilation.ravel() # to convert the multidimensional array to the 1d array
	count = 0
	for x in np.nditer(dil):
		if (x==255):
                	count=count+1  # number of white pixels
	total=float(np.size(dil)) # total number of pixels 
	c=float(count) 
	p=c/total 
	per=p*100  #calculating the percentage of the green pixels
	print(per)
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	ser=serial.Serial('/dev/ttyACM0')
	GPIO.setup(13,GPIO.IN) 
	GPIO.setup(5,GPIO.IN)
	GPIO.setup(11,GPIO.OUT)
	GPIO.setup(9,GPIO.OUT)
	GPIO.setup(10,GPIO.OUT)
	RAIN=GPIO.input(5)                                #Variable to store the Raining Status
	relay=GPIO.input(13)                              #Variable to store the on/off of relay
	t=time.asctime(time.localtime(time.time()))       #Variable stores the current time
	humidity,temperature = Adafruit_DHT.read_retry(11,2)
	a=ser.readline()
	wind=requests.get("https://api.thingspeak.com/apps/thinghttp/send_request?api_key=RVRML8PD68G61SIB")
	file=open("/var/www/html/dht.csv","a")                          #Opening File in the Append Mode
	GPIO.output(11,True)
	time.sleep(1)
	#GPIO.output(11,False)

	#file.write(str(t))
	#file.write(",")
	print(str(t))
	#file.write(str(temperature))
	#file.write(",")
	print('temp'+str(temperature))
	#file.write(str(humidity))
	#file.write(",")
	print('humidity'+str(humidity))
	#file.write(str(relay))
	#file.write(",")
	print('relay'+str(relay))
	#file.write(str(RAIN))
	#file.write(",")
	print('rain'+str(RAIN))
	#file.write("0")
	#file.write(",")
	#file.write(str(wind.text))
	print('wind'+str(wind.text))
	#file.write(",")
	#file.write(str(per))
	#file.write(",")
	#file.write(str(a))
	print("water/mois"+str(a))
	#file.close()
	time.sleep(2)
	GPIO.output(11,False)
	ser.write('end')
	print("end")
	#GPIO.output(11,False)
	for k in range(0,5):
		print("waiting for"+' '+str(k)+' '+'second')
		time.sleep(1)
	GPIO.output(11,False)
	aa = str(a).replace("\r\n", "")
	m,n = aa.split(",")
	print(type(aa))
	test = np.array([temperature, humidity, RAIN, per, float(m), float(n), float(wind.text)])
	print(test)
	arr = test.reshape(1, -1)
	predict1=clf.predict(arr)
	print("Status of the relay of main node should be:")
	print(predict1)
        #print(accuracy_score(y_train, predict1))
	time.sleep(3)
	node1=0
	node2=0
    	#client.subscribe(MQTT_PATH2)
	def on_message(client, userdata, msg):
	    mois=msg.payload
	    print(str(mois))
	    global node1
	    global node2
	    print(mois[0])
	    if mois[0]=='1':
		node1 = str(mois)
		print(node1)
		print("recieved from node 1")
	    if mois[0]=='2':
		node2= str(mois)
		print(node2)
		print("data recieved from node 2")
	print("##########data recieved from node 1#############")
	client = mqtt.Client()
	client.on_connect = on_connect
	client.on_message = on_message
	client.connect(MQTT_SERVER1, 1883, 60)
	#client.loop_forever()
	client.loop_start()
	time.sleep(20)
	#print(node1)
	GPIO.output(10,True)
	client.loop_stop()
	print("final values to be write")
	print(node1)
	print(node2)
	#file.write(str(t))
        #file.write(",")
        #print(str(t))
        #file.write(str(temperature))
        #file.write(",")
        #print('temp'+str(temperature))
        #file.write(str(humidity))
        #file.write(",")
        #print('humidity'+str(humidity))
        #file.write(str(relay))
        #file.write(",")
        #print('relay'+str(relay))
        #file.write(str(RAIN))
        #file.write(",")
        #print('rain'+str(RAIN))
        #file.write(str(node1))
	#print("data of node1 is written")
	####global node1
	####global node2
	nod, wi, perc, moi, wt = node1.split(",")
	test2 = np.array([temperature, humidity, RAIN, perc, float(moi), float(wt), float(wi)])
        arr2 = test2.reshape(1, -1)
        predict2=clf.predict(arr2)
        print("Status of the relay of 1st node should be:")
        print(predict2)
        #print(accuracy_score(y_train, predict1))
        #time.sleep(3)

	time.sleep(3)
	#file.write("\n")
	#file.close()
	GPIO.output(10,False)
	GPIO.output(9,True)
	#file.write(str(t))
        #file.write(",")
        #print(str(t))
        #file.write(str(temperature))
        #file.write(",")
        #print('temp'+str(temperature))
        #file.write(str(humidity))
        #file.write(",")
        #print('humidity'+str(humidity))
        #file.write(str(relay))
        #file.write(",")
        #print('relay'+str(relay))
        #file.write(str(RAIN))
        #file.write(",")
        #print('rain'+str(RAIN))
	#file.write(str(node2))
        #print("data of node2 is written")
	#file.write("\n")
	time.sleep(3)
	nod2, wi2, perc2, moi2, wt2 = node2.split(",")
        #test2 = np.array([temperature, humidity, RAIN, perc, moi, wt, wi])
	test3 = np.array([temperature, humidity, RAIN, perc2, float(moi2), float(wt2), float(wi2)])
        arr3 = test3.reshape(1, -1)
        predict3=clf.predict(arr3)
        print("Status of the relay of 2nd node should be:")
        print(predict3)

	GPIO.output(9,False)
	file.close()





GPIO.cleanup()

