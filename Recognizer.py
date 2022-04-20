import cv2
import numpy
import pyttsx

engine = pyttsx.init() #initializing engine
newVoiceRate = 105
engine.setProperty('rate',newVoiceRate)
load = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer/TraningData.yml"); #loading the training data
font = cv2.FONT_HERSHEY_SIMPLEX

names = {
    '89': 'Rishi',
    '111': 'Ram',
    '107': 'Priyanka',
    '002': 'SenthilSelvi'
}

f = open("datatext.txt","r")
user = {}
for x  in f:
        y,z = x.split(" ")
        user[y] = z.replace("\n","")
print(user)
while(1):
    status,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = load.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w]) #returns id and confidence level
        
        if(conf>85):   #VERY IMP confidence level is checked
            name = f"Hello {user[str(id)]}"
            
        else:
            name = ""

        engine.say(name)
        engine.runAndWait()
        
        cv2.putText(img,str(name),(x,y+h),font, 0.5, (255, 0, 0))
        
    cv2.imshow('FaceDetect',img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

f.close()
cap.release()
cv2.destroyAllWindows()
    
