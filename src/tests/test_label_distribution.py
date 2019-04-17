import os
import numpy as np

directory= "/import/c4dm-datasets/nsynth/nsynth-train/audio"


word_list = ['brass','flute','guitar','keyboard','mallet','string','vocal']
count = np.zeros(7)
for files in os.listdir(directory):
    if(files.endswith(".wav") and any(word in files for word in word_list)):
            if(files.find("brass")!=-1):
                label = 0
                count[0] +=1
            elif(files.find("flute")!=-1):
                label = 1
                count[1] +=1
            elif(files.find("guitar")!=-1):
                label = 2
                count[2] +=1
            elif(files.find("keyboard")!=-1):
                label = 3
                count[3] +=1
            elif(files.find("mallet")!=-1):
                label = 4
                count[4] +=1
            #elif(files.find("reed")!=-1):
            #    label = 5
            #    count[5] +=1
            elif(files.find("string")!=-1):
                label = 5
                count[5] +=1
            elif(files.find("vocal")!=-1): 
                label = 6
                count[6] +=1
 
print(count,np.sum(count))
