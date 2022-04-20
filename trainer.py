import os,cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create();
path = 'Data'

def img(path):
    # print(os.listdir(path))
    imgpaths = []
    for subimg in os.listdir(path):
        imgpaths.append(path+f'/{subimg}')
    # imgpaths = [os.path.join(path,f) for f in os.listdir(path)] #getting images path
    # print(imgpaths)
    # return
    #print imgpaths
    
    faces=[]
    users=[]
    for imgpath in imgpaths:
        faceimg = Image.open(imgpath).convert('L') #converting to grayscale
        facenp = np.array(faceimg,'uint8')
        user=int(os.path.split(imgpath)[-1].split('.')[0]) #getting image ids
        faces.append(facenp)
        print (user)
        users.append(user)
        cv2.imshow("Traning",facenp)
        cv2.waitKey(10)
    return users,faces


users , faces = img(path)
recognizer.train(faces,np.array(users)) #training
recognizer.write('recognizer/TraningData.yml') #saving histogram data
cv2.destroyAllWindows()
