# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
import scipy.io

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def ReadTrainData(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    imagesTrain = [] # images
    labelsTrain = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            imagesTrain.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labelsTrain.append(row[7]) # the 8th column is the label
        gtFile.close()
        print("reading the folder %d",c)
    return imagesTrain,labelsTrain

def ReadTestData(rootpath):
    images = []
    labels = []
    gtFile = open(rootpath+'GT-final_test.csv')
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader)
    for row in gtReader:
        images.append(plt.imread(rootpath+row[0]))
        labels.append(row[7])
    gtFile.close()
    return images,labels
    
## training dataset
#rootPath = 'C:/Users/jiaoyi2/Documents/Dataset/GTSRB/Final_Training/Images'
#(imagesTrain,imagesTrain) = ReadTrainData(rootPath)

# test dataset
rootPath = 'C:/Users/jiaoyi2/Documents/Dataset/GTSRB/Final_Test/Images/'
(imagesTest,labelsTest) = ReadTestData(rootPath)

#scipy.io.savemat('OriGTS.mat',{
#        'images':imagesTrain,
#        'labels':labelsTrain
#        })
scipy.io.savemat('OriGTSTest.mat',{
        'images':imagesTest,
        'labels':labelsTest
        })