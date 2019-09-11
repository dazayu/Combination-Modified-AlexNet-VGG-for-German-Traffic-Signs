# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:35:25 2019

@author: jiaoyi2
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:56:25 2019

@author: jiaoyi2
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:10:30 2019

@author: jiaoyi2
"""
import scipy.io
import numpy as np
import skimage
from imgaug import augmenters as iaa
##################################
# Input Dataset
Train = scipy.io.loadmat('OriTrainData.mat')
Test = scipy.io.loadmat('OriTestData.mat')
#test_mat = scipy.io.loadmat('cts_testing_data.mat')
trainData = Train["images"]
testData = Test["images"]
trainLabels = Train["labels"]
testLabels = Test["labels"]

if trainLabels.shape[0] < trainLabels.shape[1]:
    trainLabels = np.transpose(trainLabels)
if testLabels.shape[0] < testLabels.shape[1]:
    testLabels = np.transpose(testLabels)

numTrain = trainLabels.shape[0]
numTest = testLabels.shape[0]

height = np.size(trainData,1)
width = np.size(trainData,2)
depth = np.size(trainData,3)
sizeOfIn = np.array([height,width,depth])

'''
the size of the images we want:
'''
#trainData = trainData.reshape((numTrain,28,28))
#testData = testData.reshape((numTest,28,28))
#testData = testData.reshape((testData.shape[0], 80, 80, 3))

#######################
#Test of the Code
#trainData = trainData[0:200,:,:,:]
#testData = testData[0:20,:,:,:]
#trainLabels = trainLabels[0:200]
#testLabels = testLabels[0:20]
#numTrain = 200
#numTest = 20
####################################
# the Code of Data Augmentation
def Trans(img):
    img = np.array(img)
    img = img.astype(np.uint8)
#    height = np.size(img,0)
#    width = np.size(img,1)      
    xPixel = np.random.randint(-5,6,dtype='int8')
    yPixel = np.random.randint(-5,6,dtype='int8')
    tempImgTrans = img.copy()    
    if (xPixel > 0):
        for i in range(0, width+1, 1):
            for j in range(height):
                if (i < width-xPixel):
                    tempImgTrans[j][i] = tempImgTrans[j][i+xPixel]
                elif (i < width):
                    tempImgTrans[j][i] = tempImgTrans[j][2*(width-1-xPixel)-i]
    elif (xPixel == 0):
        pass
    else:
        for i in range(width-1, -1, -1):
            for j in range(height):
                if (i > -xPixel-1):
                    tempImgTrans[j][i] = tempImgTrans[j][i+xPixel]
                elif (i >= 0):
                    tempImgTrans[j][i] = tempImgTrans[j][-2*xPixel-i]      

    ImgTrans = tempImgTrans.copy()
    # xPixel is positive, shift down 
    # xPixel is negative, shift up
    if(yPixel > 0):
        for j in range(height-1, -1, -1):
            for i in range(width):
                if (j > yPixel-1):
                    ImgTrans[j][i] = ImgTrans[j-yPixel][i]
                elif (j >= 0):
                    ImgTrans[j][i] = ImgTrans[2*yPixel-j][i]
    elif(yPixel == 0):
        pass
    else:
        for j in range(0, height+1, 1):
            for i in range(width):
                if(j < height + yPixel):
                    ImgTrans[j][i] = ImgTrans[j-yPixel][i]
                elif(j < height):
                    ImgTrans[j][i] = ImgTrans[2*(height-1+yPixel)-j][i]
    return ImgTrans
            
def GNois(img):
    img = np.array(img)
    img = img.astype(np.uint8)
    noise = np.random.randint(0,200,size=sizeOfIn)
    noise = 0 * (noise<80) + (noise-80) * (noise==199) 
    noise = noise.astype(np.uint8)
    ImgNois = img + noise
    ImgNois = ImgNois.astype(np.uint8)   
    return ImgNois

def Rot(img):
    img = np.array(img)
    img = img.astype(np.uint8)
    ImgRot = skimage.transform.rotate(img, angle=np.random.randint(-30,31), 
                                      mode='reflect',preserve_range=1)
    ImgRot = ImgRot.astype(np.uint8)
    return ImgRot
    ################
def CInt(img):
    img = np.array(img)
    img = img.astype(np.uint8)
    ImgCInt = img.astype(np.uint16)
    iValue = np.ones(sizeOfIn,dtype='uint16') * np.random.randint(0,100) 
    ImgCInt += iValue
    np.clip(ImgCInt,0,255,out=ImgCInt)
    ImgCInt = ImgCInt.astype(np.uint8)    
    ################
    return ImgCInt

def Blur(img):
    img = np.array(img)
    seq = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0,1.0))
            ])
    ImgBlur = seq(images=img)
    ImgBlur = np.uint8(ImgBlur)
    return ImgBlur

def GammaContrast(img):
    img = np.array(img)
    seq = iaa.Sequential([iaa.GammaContrast(np.random.uniform(0.5,2))])
    ImgGC = seq(images=img)
    ImgGC = np.uint8(ImgGC)
    return ImgGC
    
def Shuffle(images,labels):
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)
    return images,labels
# randomly rearrange the dataset

tempTrain = np.zeros((numTrain,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
tempTest = np.zeros((numTest,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')

#combinedTrain = 5 * numTrain
#combinedTest = 5 * numTest

#fullTrainImages = np.zeros((combinedTrain,32,32),dtype='uint8')
#fullTrainLabels = np.zeros((combinedTrain,1),dtype='uint8')

transTrain = np.zeros((numTrain,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
noisTrain = np.zeros((numTrain,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
rotTrain = np.zeros((numTrain,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
intCTrain = np.zeros((numTrain,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
blurTrain = np.zeros((numTrain,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
GCTrain = np.zeros((numTrain,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')


transTest = np.zeros((numTest,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
noisTest = np.zeros((numTest,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
rotTest = np.zeros((numTest,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
intCTest = np.zeros((numTest,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
blurTest = np.zeros((numTest,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')
GCTest =  np.zeros((numTest,sizeOfIn[0],sizeOfIn[1],sizeOfIn[2]),dtype='uint8')


(trainData,trainLabels) = Shuffle(trainData,trainLabels)


#####   Train
for iterTrain in range(0,numTrain):
    tempTrain[iterTrain] = Trans(trainData[iterTrain])
    transTrain[iterTrain] = Trans(trainData[iterTrain])
    
    tempTrain[iterTrain] = GNois(tempTrain[iterTrain])
    noisTrain[iterTrain] = GNois(trainData[iterTrain])
    
    tempTrain[iterTrain] = Rot(tempTrain[iterTrain])
    rotTrain[iterTrain] = Rot(trainData[iterTrain])
    
    tempTrain[iterTrain] = CInt(tempTrain[iterTrain])
    intCTrain[iterTrain] = CInt(trainData[iterTrain])
    
    tempTrain[iterTrain] = Blur(tempTrain[iterTrain])
    blurTrain[iterTrain] = Blur(trainData[iterTrain])
    
    tempTrain[iterTrain] = GammaContrast(tempTrain[iterTrain])
    GCTrain[iterTrain] = GammaContrast(trainData[iterTrain])

    
    print("[INFO] processing the training dataset",iterTrain,"/",numTrain)

#fullTrainLabels = np.concatenate((trainLabels,trainLabels,trainLabels,trainLabels,trainLabels),axis=0) 
#fullTrainImages = np.concatenate((transTrain,rotTrain,noisTrain,intCTrain,trainData),axis=0)



#(tempTrain,trainLabels) = Shuffle(tempTrain,trainLabels)
#(trainData,trainLabels) = Shuffle(trainData,trainLabels)
#(transTrain,trainLabels) = Shuffle(transTrain,trainLabels)
#(rotTrain,trainLabels) = Shuffle(rotTrain,trainLabels)
#(noisTrain,trainLabels) = Shuffle(noisTrain,trainLabels)
#(intCTrain,trainLabels) = Shuffle(intCTrain,trainLabels)
    
scipy.io.savemat('SeriallyAugmentedTrainData.mat',{
    'images':tempTrain,
    'labels':trainLabels
    })
##scipy.io.savemat('AugmentedTrainData.mat',{
#    'images':fullTrainImages,
#    'labels':fullTrainLabels
#    })
scipy.io.savemat('TrainOri.mat',{
    'images':trainData,
    'labels':trainLabels
    })    
scipy.io.savemat('TrainTrans.mat',{
    'images':transTrain,
    'labels':trainLabels
    })
scipy.io.savemat('TrainRot.mat',{
    'images':rotTrain,
    'labels':trainLabels
    })    
scipy.io.savemat('TrainNois.mat',{
    'images':noisTrain,
    'labels':trainLabels
    })
scipy.io.savemat('TrainIntChange.mat',{
    'images':intCTrain,
    'labels':trainLabels
    })
scipy.io.savemat('TrainBlur.mat',{
    'images':blurTrain,
    'labels':trainLabels
    })
scipy.io.savemat('TrainGammaContrast.mat',{
    'images':GCTrain,
    'labels':trainLabels
    })    

    
(tempTest,testLabels) = Shuffle(tempTest,testLabels)
#####   Test
for iterTest in range(0,numTest):
    tempTest[iterTest] = Trans(testData[iterTest])
    transTest[iterTest] = Trans(testData[iterTest])
    
    tempTest[iterTest] = GNois(tempTest[iterTest])
    noisTest[iterTest] = GNois(testData[iterTest])
    
    tempTest[iterTest] = Rot(tempTest[iterTest])
    rotTest[iterTest] = Rot(testData[iterTest])
    
    tempTest[iterTest] = CInt(tempTest[iterTest])
    intCTest[iterTest] = CInt(testData[iterTest])
    
    tempTest[iterTest] = Blur(tempTest[iterTest])
    blurTest[iterTest] = Blur(testData[iterTest])
    
    tempTrain[iterTest] = GammaContrast(tempTest[iterTest])
    GCTest[iterTest] = GammaContrast(testData[iterTest])
    print("[INFO] processing the test dataset",iterTest,"/",numTest)

(tempTest,testLabels) = Shuffle(tempTest,testLabels)
(testData,testLabels) = Shuffle(testData,testLabels)
(transTest,testLabels) = Shuffle(transTest,testLabels)
(rotTest,testLabels) = Shuffle(rotTest,testLabels)
(noisTest,testLabels) = Shuffle(noisTest,testLabels)
(intCTest,testLabels) = Shuffle(intCTest,testLabels)

scipy.io.savemat('SeriallyAugmentedTestData.mat',{
    'images':tempTest,
    'labels':testLabels
    })
scipy.io.savemat('TestOri.mat',{
    'images':testData,
    'labels':testLabels
    })    
scipy.io.savemat('TestTrans.mat',{
    'images':transTest,
    'labels':testLabels
    })
scipy.io.savemat('TestRot.mat',{
    'images':rotTest,
    'labels':testLabels
    })    
scipy.io.savemat('TestNois.mat',{
    'images':noisTest,
    'labels':testLabels
    })
scipy.io.savemat('TestIntChange.mat',{
    'images':intCTest,
    'labels':testLabels
    })
scipy.io.savemat('TestBlur.mat',{
    'images':blurTest,
    'labels':testLabels
    })
scipy.io.savemat('TestGammaContrast.mat',{
    'images':GCTest,
    'labels':testLabels
    })