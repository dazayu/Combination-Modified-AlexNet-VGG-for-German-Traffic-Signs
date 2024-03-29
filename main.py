##############################################
#   jiaoyifei Sep.2019 HELLA intern
##############################################
from Architecture import DeepConvNetwork
from keras.optimizers import SGD
from keras.utils import np_utils
#from keras import backend as K
import numpy as np
#import argparse
import scipy.io
#from matplotlib import pyplot as plt
#import h5py
import time
import xlsxwriter

##############################################
#   setting the parameters and hyperparameters
##############################################
dataPath = 'C:/Users/jiaoyi2/Documents/Clas a DataAug of GTS/simpVGG/'
# the rootpath of the folder
isEvaluation = True
# when True: only do the evaluation of the existing weights with the testDataset
# when false: train an architecture and test the architecture 
isFromScratch = False
# when true: the code loads the weights and starts the training on the basis of 
#            the existing weights
# when false: the architecture will be trainied from scratch
isExpDecreasing = True
initialLearningRate = 0.03
paraOfLRDescreasing = 0.8
# the learning rate of the training processes, it can decreasing in exponential function
initialGroup = 1
# additional parameter with the isFromScratch
# the number is the ending group number of the loading weight.
# for example, the last training process ended at 25 epochs, then the 
# initialGroup = 5
listOfWeights = [
# elements are all string,
# the name of the weights data, is necessary when the isEvaluation is true
        'GTSResizedTrain_Epochs1to5__20190910123338',
        'GTSResizedTrain_Epochs6to10__20190910133050',
        'GTSResizedTrain_Epochs11to15__20190910142537',
        'GTSResizedTrain_Epochs16to20__20190910152032',
        'GTSResizedTrain_Epochs21to25__20190910161750',
        'GTSResizedTrain_Epochs26to30__20190910171126',
        'GTSResizedTrain_Epochs31to35__20190910180526',
        'GTSResizedTrain_Epochs36to40__20190910185619',
        'GTSResizedTrain_Epochs41to45__20190910194711',
        'GTSResizedTrain_Epochs46to50__20190910203843'
        ]
groupOfEpochs = 9         
# One Group is five epochs
batchSize = 128
# the number of images in every batch
trainDataset = 'AugTrain'    
# String, name of the training set
testDataset = [                 
# The elements of the test data list should be all string
#         'AAAAATestTest'
        'TestRot',
        'TestBlur',
        'TestGammaContrast',
        'TestIntChange',
        'TestNois',
        'TestTrans',
        'TestOri',
        'AugTest'
        ]
sizeOfImg = [48,48,3]
# size of the input images [height,width,channel]
numOfClasses = 43
# number of the classes of the training/test dataset
dropoutRate = 0.5
# the rate of the dropout layer
regularizationParameter = 0.01


##############################################
# assistent Functions and global variables
##############################################
def Print():
    print(" ")
    print("###########################")
    print(" ")
def Evaluation(num,iterGroup):
    nameOfTestData = testDataset[num]
    pathOfTestData = dataPath+nameOfTestData
    test_mat = scipy.io.loadmat(pathOfTestData)
    testData = test_mat["images"]
    testLabels = test_mat["labels"]
    testLabels = np_utils.to_categorical(testLabels, numOfClasses)
    print("[INFO] evaluating of the epochs",iterGroup*5+1,
          "to",(iterGroup+1)*5,"with the Test Dataset",nameOfTestData,"...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=batchSize, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
    Print()       
    return loss,accuracy
learningRateMat = np.zeros((groupOfEpochs,1))
#def PlotTheParas(history):
#    # Plot training & validation accuracy values
#    plt.figure(1)
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('Model accuracy')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc='upper left')
#    plt.show()
#    
#    # Plot training & validation loss values
#    plt.figure(2)
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('Model loss')
#    plt.ylabel('Loss')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'Test'], loc='upper left')
#    plt.show()
#    return
##############################################
# assertions
##############################################
if isEvaluation == True:
    assert len(listOfWeights) != 0,'listOfWeights should not be empty'
else:
    if isFromScratch == True:
        assert initialGroup == 0,'The initialGroup should be 0 if the isFromScratch is False'
        assert len(listOfWeights) == 0,'listOfWeights should be empty'
        weightDocName = None    # initial weight file
    else:
        assert initialGroup != 0,'The initialGroup should not be 0 if the isFromScratch is True'
        assert len(listOfWeights) == 1,'listOfWeights should only have one element'
        weightDocName = dataPath+listOfWeights[0]+'.hdf5' # initial weight file

#historyList = []   
##############################################
# the code of the training process
##############################################
if isEvaluation == False:
    nameOfTrainingData = dataPath+trainDataset
    train_mat = scipy.io.loadmat(nameOfTrainingData)
    trainData = train_mat["images"]
    trainLabels = train_mat["labels"]
    trainLabels = np_utils.to_categorical(trainLabels, numOfClasses)
    accuracyMatTrain = np.zeros((groupOfEpochs,5))
    accuracyMatTest = np.zeros((groupOfEpochs,len(testDataset)))
    lossMatTest = np.zeros((groupOfEpochs,len(testDataset)))
    
    for iterGroup in range (initialGroup,initialGroup+groupOfEpochs):
        if isExpDecreasing == True:
            learningRate = initialLearningRate*(pow(paraOfLRDescreasing,iterGroup))
            learningRateMat[iterGroup-initialGroup] = learningRate
        if isFromScratch == False:
            print("[INFO] compiling model of the epochs",iterGroup*5+1,
                  "to",(iterGroup+1)*5,"...")
            print("[INFO] the learning rate is {}".format(learningRate))
            print("[INFO] the loaded weight file is",weightDocName)
            Print()
            opt = SGD(lr=learningRate)
            model = DeepConvNetwork.build(sizeOfInput=sizeOfImg,numOfClass=numOfClasses,
                                dropRate=dropoutRate,
                                regPara=regularizationParameter,
                                weightsPath=weightDocName)
        else:
            print("[INFO] compiling model of the epochs",iterGroup*5+1,
                      "to",(iterGroup+1)*5,"...")
            print("[INFO] the learning rate is {}".format(learningRate))
            print("[INFO] the loaded weight file is",weightDocName)
            Print()
            opt = SGD(lr=learningRate)
            model = DeepConvNetwork.build(sizeOfInput=sizeOfImg,numOfClass=numOfClasses,
                                    dropRate=dropoutRate,
                                    regPara=regularizationParameter,
                                    weightsPath=weightDocName)
        
        model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
                #if args["load_model"] < 0:
        print("[INFO] training of the epochs",iterGroup*5+1,
              "to",(iterGroup+1)*5,"...")
        history = model.fit(trainData,trainLabels,batch_size=batchSize,
                            initial_epoch=iterGroup*5,epochs=(iterGroup+1)*5,
                            verbose=1)
#        listOfRealtimeData = listOfRealtimeData 
#        PlotTheParas(history)
        # show the accuracy on the testing set
#        accuracyMatTrain[iterGroup,:] = np.round(
#                        np.asarray(history.history['acc']),4)
        Print()
        for iterOfTestDataset in range(0,len(testDataset)):
            (loss,acc) = Evaluation(iterOfTestDataset,iterGroup)     
            accuracyMatTest[iterGroup-initialGroup,iterOfTestDataset]=np.round(acc,4) 
            lossMatTest[iterGroup-initialGroup,iterOfTestDataset]=np.round(loss,4)
        print("[INFO] dumping weights to file ...")
        weightDocName = dataPath+trainDataset+"_Epochs{}to{}".format(iterGroup*5+1,(iterGroup+1)*5)+\
            time.strftime("_%Y%m%d%H%M%S",time.localtime())+'.hdf5'
        print("    ",weightDocName)
        model.save_weights(weightDocName, overwrite=True)
        Print()

##############################################
# the code of testing the existing weights
##############################################
elif isEvaluation == True: 
    assert listOfWeights != [], "forgot the name of weights"
    accuracyMatTest = np.zeros((len(listOfWeights),len(testDataset)))
    lossMatTest = np.zeros((len(listOfWeights),len(testDataset)))

    for iterOfWeights in range(0,len(listOfWeights)):
        if isExpDecreasing == True:
            learningRate = initialLearningRate*(pow(paraOfLRDescreasing,iterGroup))
            learningRateMat[iterGroup-initialGroup] = learningRate
        else:
            learningRate = initialLearningRate
            
        weightDocName = dataPath+listOfWeights[iterOfWeights]+'.hdf5'
        print("[INFO] compiling model with the weight",listOfWeights[iterOfWeights],
              "...")
        opt = SGD(lr=learningRate)
        model = DeepConvNetwork.build(sizeOfInput=sizeOfImg,numOfClass=numOfClasses,
                            weightsPath=weightDocName,dropRate=dropoutRate,
                            regPara=regularizationParameter)
        model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
        
        # evaluate the training dataset
        for iterOfTestDataset in range(0,len(testDataset)):
            (loss,acc) = Evaluation(iterOfTestDataset,iterGroup)     
            accuracyMatTest[iterOfWeights,iterOfTestDataset]=np.round(acc,4) 
            lossMatTest[iterOfWeights,iterOfTestDataset]=np.round(loss,4)

else:
    raise Exception('The isEvaluation should be boolean value!')

##############################################
# the code of saving as excel table
############################################## 
print('Table of AccuracyMatTrain')
print(accuracyMatTrain)
Print()
print('Table of AccuracyMatTest')
print(accuracyMatTest)
Print()
print('Table of lossMatTest')
print(lossMatTest)
Print()
print('Table of learningRateMat')
print(learningRateMat)
Print()
#print("the variable history")
#print(history.history.keys())
#Print()
print("[INFO] Processing the Excel file")
Print()

workbookName = trainDataset+'_groups of epochs_'+str(groupOfEpochs) \
               +time.strftime("_%Y%m%d%H%M%S",time.localtime())+'.xlsx'
workbook = xlsxwriter.Workbook(workbookName)
sheetTest = workbook.add_worksheet()
percentageFormal = workbook.add_format()
percentageFormal.set_num_format('0.00%')
decimalFormal = workbook.add_format()
decimalFormal.set_num_format('0.000')
sheetTest.set_column('B:J',13)
sheetTest.set_column('A:A',30)

for col in range(0,len(testDataset)):
    sheetTest.write(1,col+1,testDataset[col])
    sheetTest.write(1+groupOfEpochs+3,col+1,testDataset[col])
# the first row, the name of the test dataset

sheetTest.write(0,0,'the accuracy')

if isEvaluation == False:
    for row in range(0,groupOfEpochs):
        sheetTest.write(row+2,0,'the epoch {}'.format((row+1+initialGroup)*5))
        for col in range(0,len(testDataset)):
            sheetTest.write(row+2,col+1,accuracyMatTest[row,col],percentageFormal)
else:
    for row in range(0,len(listOfWeights)):
        sheetTest.write(row+2,0,'weight\'name'+listOfWeights[row])
        for col in range(0,len(testDataset)):
                sheetTest.write(row+2,col+1,accuracyMatTest[row,col],percentageFormal)

# the first column, the number of epochs or the name of the weight

sheetTest.write(3+groupOfEpochs,0,'the loss')
if isEvaluation == False:
    for row in range(0,groupOfEpochs):
        sheetTest.write(row+5+groupOfEpochs,0,'the epoch {}'.format((row+1+initialGroup)*5))
        for col in range(0,len(testDataset)):
            sheetTest.write(row+5+groupOfEpochs,col+1,lossMatTest[row,col],decimalFormal)
else:
    for row in range(0,len(listOfWeights)):
        sheetTest.write(row+5+groupOfEpochs,0,'weight\'name'+listOfWeights[row])
        for col in range(0,len(testDataset)):
                sheetTest.write(row+5+groupOfEpochs,col+1,lossMatTest[row,col],decimalFormal)
    
workbook.close()       

##############################################
# the code of plotting
############################################## 
#print("[INFO] Plot generating") 
#iterEpoch = range(0,groupOfEpochs)
#fig = plt.figure()
#acc = fig.add_subplot(1,2,1)
#with plt.style.context('Solarize_Light2'):
#    for iterTestDataset in range(0,len(testDataset)):
#        acc.plot(iterEpoch,accuracyMatTest[iterEpoch-initialGroup,iterTestDataset])
#    acc.set_xticks(range(initialGroup,initialGroup+groupOfEpochs,5))
#    plt.xlabel('Epoch')
#    plt.ylabel('Accuracy')
#loss = fig.add_subplot(1,2,2)
#with plt.style.context('Solarize_Light2'):
#    for iterTestDataset in range(0,len(testDataset)):
#        loss.plot(iterEpoch,lossMatTest[iterEpoch-initialGroup,iterTestDataset])    
#    loss.set_xticks(range(initialGroup,initialGroup+groupOfEpochs,5))
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')



