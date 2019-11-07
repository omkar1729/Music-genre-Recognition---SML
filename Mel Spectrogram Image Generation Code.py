import numpy as np
import scipy as sc
import matplotlib as plt
import librosa
from librosa import display
import sklearn
import os
import pylab
import cv2


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
    S = librosa.feature.melspectrogram(y=waveform, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    #Here, number of features = 13
    
    #mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    #print(mfccs.mean(axis=1))
    #print(mfccs.var(axis=1))
    
    #print(mfccs.shape)
    
    #img = librosa.display.specshow(mfccs,sr=sr,x_axis='time')
    pylab.savefig(train_directory+"/"+dr+"/"+str(count), bbox_inches=None, pad_inches=0)
    pylab.close()
    #print(type(img))
    
    #img.figure.savefig(train_directory+dr+"/"+str(count))
    #fig = plot.get_figure()
    #fig.savefig(str(count1)+".jpg")
    
    
    
    
    
def generateImages(list_of_files):
    
    for dr in list_of_files.keys():   #Iterating through the genres
        print(dr)
        count1=1 #Image files in each genre are named as 1,2,3,....100.jpg
        print(list_of_files[dr])
        for filename in list_of_files[dr]:
            
            calculate_fft(filename,dr,count1,train_directory) #Image formed and saved
            #print(type(img))

            count1+=1 
            
        

        print("Done.")






















if __name__=="__main__":
    categories=[] # a list of all categories ['pop','rock',.....,'jazz']
    list_of_files = {} # a dictionary with the genre as the key value. For every key value, we have a list which contains the names of the audio files contained within that genre folder
    path =  "genres"
    categories,list_of_files = getcategories(categories,list_of_files,path)
    #print(list_of_files['jazz'])
    train_directory='train1/'
    #Next, we create a directory 'train' which contains subfolders named after each genre. In each subfolder, there will be 100 images corresponding to the genre that subfolder is representing.
    if not os.path.exists(train_directory):
        os.mkdir(train_directory)  #command for making the directory
    for dr in list_of_files.keys():
        if not os.path.exists(train_directory+dr):
            os.mkdir(train_directory+dr) #command for making the subfolder by appending the genre to the directory name
    generateImages(list_of_files)
    
    
