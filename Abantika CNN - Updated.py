import numpy as np
import scipy as sc
import matplotlib as plt
import librosa
from librosa import display
import sklearn
import os
import pylab
import cv2
import random
import keras
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras.layers import Dense, LSTM,Conv2D, Dropout,Input, Flatten,Activation,MaxPooling2D,AveragePooling2D,BatchNormalization


def train_model():

    
    model = Sequential()
    model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(480,640,3)))
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(AveragePooling2D(pool_size = (2,2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64,(4,4),padding='same',activation='relu',strides=(2,2)))
    model.add(Conv2D(64,(4,4),activation='relu',strides=(2,2)))
    model.add(AveragePooling2D(pool_size = (2,2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64,(4,4),padding='same',activation='relu'))
    model.add(Conv2D(64,(4,4),activation='relu'))
    model.add(AveragePooling2D(pool_size = (2,2)))
    #model.add(Dropout(0.25))
    #model.add(Conv2D(64,(10,10),padding='same',activation='relu',strides=(2,2)))
    #model.add(Conv2D(64,(10,10),activation='relu',strides=(2,2)))
    #model.add(MaxPooling2D(pool_size = (2,2)))
    #model.add(Dropout(0.25))

    #LSTM


    #model.add(LSTM(96,input_shape=(7,704)))
    """
    lstm1,h,c = LSTM(96, return_sequences=True,return_state=True)([lstm1,h,c])
    lstm1,h,c = LSTM(96, return_sequences=False,return_state=True)([lstm1,h,c])
    #y = Lambda(lambda x: tf.keras.backend.concatenate([h,c],0))([lstm1,h,c])
    y = Concatenate()([h,c])
    model_language = Model(inputs=inputs1, outputs=y)
    # combined model
    conc = keras.layers.Multiply()([model_language.output,model.output])
    """



    model.add(Flatten())
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))

    model.add(Dense(10,activation='softmax'))

    model.summary()
    return(model)

def getcategories(categories,list_of_files,path):
    for files in os.listdir(path):
        if ".mf" not in files:
            categories.append(files)
            list_of_files[files] = list()
            newPath = os.path.join(path,files)
            for f in os.listdir(newPath):
                list_of_files[files].append(f)
    #print(list_of_files)
    return(categories,list_of_files)

def calculate_fft(filename,dr,count,traindirectory):
    fullpath = "C:/Users/16022/Desktop/Arizona/1stSem/Statistical Machine Learning/SML Project/genres/" + dr +"/"+filename
    waveform,sr=librosa.load(fullpath)
    #print("Sampling Rate: "+str(sr))
    #waveform is a 1 dimensional numpy array that stores the values of an audio signal wrt time
    #sr is the sampling rate of the audio signal
    frame_size = int(0.5 * sr)
    #print("Frame Size: "+str(frame_size))
    #Each 10 second audio is divided into frames each of 0.5 seconds each
    stride = int(0.1 * sr)
    #print("Stride: "+str(stride))
    #Here stride is taken to be 0.1 seconds
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    mfccs=librosa.feature.mfcc(waveform,sr=sr,n_mfcc=13,hop_length=stride,n_fft=frame_size)
    #Here, number of features = 13
    
    #mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    #print(mfccs.mean(axis=1))
    #print(mfccs.var(axis=1))
    
    #print(mfccs.shape)
    
    img = librosa.display.specshow(mfccs,sr=sr,x_axis='time')
    pylab.savefig(train_directory+"/"+dr+"/"+str(count)+".jpg", bbox_inches=None, pad_inches=0)
    pylab.close()
    #print(type(img))
    
    #img.figure.savefig(train_directory+dr+"/"+str(count))
    #fig = plot.get_figure()
    #fig.savefig(str(count1)+".jpg")
    
    
    return mfccs,img
    
    
def generateImages(list_of_files):
    
    for dr in list_of_files.keys():   #Iterating through the genres
        print(dr)
        count1=1 #Image files in each genre are named as 1,2,3,....100.jpg
        print(list_of_files[dr])
        for filename in list_of_files[dr]:
            
            mfccs,img=calculate_fft(filename,dr,count1,train_directory) #Image formed and saved
            #print(type(img))

            count1+=1 
            
        

        print("Done.")

def get_index(category,categories):
    print(category)
    for i in range(0,len(categories)):
        if(categories[i]==category):
            break
    return(i)

def genTrainTest(training_data,test_data,list_of_files,categories,directory):
    list1 = [x for x in range(100)] #Since we have 100 files in each genre, I take a list which contains values like [0,1,2,3,4,.....,100].
    list2 = list1 #create a backup of list1
    for category in list_of_files.keys(): #For each music genre
        category1 = category + "/"
        path = os.path.join(directory,category1) #helps to navigate through the required genre directory
        #print(path)
        class_label = get_index(category,categories) #we have to represent each category by a number which is nothing but the position of the genre name in the categories list. 
        test_index=[]
        for direc,_,filenames in os.walk(path):
              #print("Length of filenames"+str(len(filenames)))
             # print(filenames)
              random.shuffle(list1) #shuffling the list 
              train_index = list1[:70] #take the first 70 random indices as train indices
              #print("List1: "+str(list1))
              #print("Training index: "+str(train_index))
              #print(len(train_index))
              #count=0
              #count1 = 0
              #count2=0
              #print(len(list1))
              for i in range(0,len(list1)):
                if(list1[i] not in train_index):
                    test_index.append(list1[i]) #remaining 30 for test indices
              #print(count,count1,count2)
              #print(train1_index==train_index)
              #test_index = list(set(list1)-set(train_index))
              #print(len(test_index))
              #print("Test index: "+str(test_index))


              for i in train_index:
                    full_path=os.path.join(path,filenames[i]) #navigate to each genre subfolder
                    img_arr = cv2.imread(full_path) #Read the image
                    #print(img_arr)
                    im_rgb = cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB) #Convert into RGB as colour is important
                    #print(im_rgb.shape)


                    #plt.imshow(im_rgb)
                    #plt.show()
                    training_data.append([im_rgb,class_label]) #each row in the training data contains the rgb numpy matrix and the class label 



              for i in test_index:
                    full_path=os.path.join(path,filenames[i])
                    img_arr = cv2.imread(full_path)
                    #print(img_arr.shape)
                    im_rgb = cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)
                    #print(im_rgb.shape)
                    #plt.imshow(im_rgb)
                    #plt.show()
                    test_data.append([im_rgb,class_label]) #similarly for test data
                    
                    
    return training_data,test_data

    
def convert_to_numpy(training_data,test_data):
    X=[]
    Y=[]
    for features,class_label in training_data:
        X.append(features)
        Y.append(class_label)
        
    #X has to be a numpy array
    Y=np.array(Y)
    #X=np.asarray(X)
    #print(len(X[0]))
    #print(len(X[0][0]))
    #print(len(X[0][0][0]))
    #X=np.array(X)
    X=np.array(X)
    #print(X.shape)
    #print(Y.shape)
    np.save("X.npy",X)
    np.save("Y.npy",Y)


    X2 = []
    Y2 = []
    for features, class_label in test_data:
        X2.append(features)
        Y2.append(class_label)
    X2 = np.array(X2)
    Y2 = np.array(Y2)
    np.save("X2.npy",X2)
    np.save("Y2.npy",Y2)


    return(X,Y,X2,Y2)
    
if __name__=="__main__":
    categories=[] # a list of all categories ['pop','rock',.....,'jazz']
    list_of_files = {} # a dictionary with the genre as the key value. For every key value, we have a list which contains the names of the audio files contained within that genre folder
    path =  "genres"
    categories,list_of_files = getcategories(categories,list_of_files,path)
    #print(list_of_files['jazz'])
    train_directory='train/'
    #Next, we create a directory 'train' which contains subfolders named after each genre. In each subfolder, there will be 100 images corresponding to the genre that subfolder is representing.
    if not os.path.exists(train_directory):
        os.mkdir(train_directory)  #command for making the directory
    for dr in list_of_files.keys():
        if not os.path.exists(train_directory+dr):
            os.mkdir(train_directory+dr) #command for making the subfolder by appending the genre to the directory name
    #generateImages(list_of_files) #Since we have already generated images, so no need to use this function call
    training_data=[]
    test_data=[]
    #generating training and test data
    training_data, test_data = genTrainTest(training_data,test_data,list_of_files,categories,train_directory)
    print(len(training_data))
    print(len(test_data))

    
    X,Y,X2,Y2 = convert_to_numpy(training_data,test_data) #since training_data and test_data are lists, we need to convert them into numpy arrays for further processing.
    print(X.shape)
    print(Y.shape)
    print(X2.shape)
    print(Y2.shape)

    X = X.astype('float32') #Convert all the pixel values to float type as we are going to normalise them for training features
    X2 = X2.astype('float32') #Convert all the pixel values to float type as we are going to normalise them for test features
    X/=255 #Normalising the training features
    X2/=255 #Normalising the test features

    Y1=to_categorical(Y,10) #use one-hot encoding for training class labels (required for cnn)
    Y3 = to_categorical(Y2,10) #use one-hot encoding for test class labels (required for cnn)

    model = train_model()
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X,Y1,batch_size =32 ,epochs=30,verbose=1,validation_data=(X2,Y3))
    

    

    

    

    
        

            
    
    
    


        
