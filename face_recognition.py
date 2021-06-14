
import numpy as np
import cv2
import os
import sklearn.neighbors as sk
from PIL import Image, ImageTk

## Knn
def distance(v1,v2):
    #euclidean
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test, k=8):
    dist =[]

    for i in range(train.shape[0]):
        #get the vector and label
        ix = train[i,:1]
        iy = train[i,-1]
        #compute distance
        d = distance(test,ix)
        dist.append([d,iy])
    #sort based on distance and get top k
    dk = sorted(dist,key=lambda x:x[0])[:k]
    #retrieve only the labels
    labels = np.array(dk)[:,-1]

    #get frequencies of each label
    output = np.unique(labels,return_counts=True)
    #Find max frequency and corresponding label
    
    index = np.argmax(output[1])
    
    return output[0][index]

###########

def recognize():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    dataset_path ="./face_dataset/"

    face_data =[]
    labels =[]
    class_id =0
    names={}
    names_id=[]
    #Dataset preparation
    for fx in os.listdir(dataset_path):
        if fx.endswith('.jpeg'):
            names[class_id]=fx[:-5]
            path = os.path.join(dataset_path,fx)
            img = Image.open(path).convert('L')
            data_item = np.array(img,'uint8')
            face_data.append(data_item)
            names_id.append(class_id)
            class_id +=1
            #target = class_id*np.ones((data_item.shape[0],))
            #names_id.append(class_id)
            #class_id+=1;
            
            #labels.append(target[0])
    #print(labels)

    #face_dataset = np.concatenate(face_data,axis=0)
    #face_labels = np.array(labels).astype(np.int32)
    #print(face_labels,names)
    #print(face_dataset.shape)


    #trainset = np.concatenate((face_dataset,face_labels),axis=1)
    #print(trainset,trainset.shape)q
    #print(type(names))

    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.train(face_data,np.array(names_id))
    recognizer.save("dataset_path\Trainner.yml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret ,frame = cap.read()
        if ret == False:
            continue

        # convert to gray
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #detect multi faces in the images
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        l = []
        for face in faces:
            x,y,w,h = face

            #get the face roi

            offset = 5
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section = cv2.resize(face_section,(100,100))
            face_section = cv2.cvtColor(face_section,cv2.COLOR_BGR2GRAY)
            #out = knn(trainset,face_section.flatten())
            
            #model = sk.KNeighborsClassifier(n_neighbors =5)
            #model.fit(trainset[:,1].reshape(-1,1),trainset[:,-1])
            #out = model.predict(face_section.flatten().reshape(-1,1))

            #index = np.argmax(out[1])
            #print(accuracy(names,names[int(out[index])]))
            i, conf = recognizer.predict(face_section)
            #print(i)
            
            
            #int(out[index])
                
            cv2.putText(frame,names[i],(x,y-10),font,1,(255,0,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("faces",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

            


        
    
