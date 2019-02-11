# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:13:44 2018

@author: ilhamksyuriadi
"""

import xlrd
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import math
import time
import csv
start_time = time.time()


def LoadDataset(FileLoc):
    data = []
    label = []
    workbook = xlrd.open_workbook(FileLoc)
    sheet = workbook.sheet_by_index(0)
    count = 0
    for i in range(2,sheet.nrows):
        data.append(sheet.cell_value(i,0))
        label.append([int(sheet.cell_value(i,1)),int(sheet.cell_value(i,2)),int(sheet.cell_value(i,3))])
        count += 1
        print(count, "data inserted")
    return data,label

def Preprocessing(data):
    cleanData = []
    tokenizer = RegexpTokenizer(r'\w+')
    factory_stopwords = StopWordRemoverFactory()
    stopwords = factory_stopwords.get_stop_words()
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    count = 0
    for i in range(len(data)):
        lowerText = data[i].lower()#Case folding
        tokenizedText = tokenizer.tokenize(lowerText)#Punctual removal and tokenization
        swRemovedText = []#Stopwords removal
        for j in range(len(tokenizedText)):
            if tokenizedText[j] not in stopwords:
                swRemovedText.append(tokenizedText[j])
        stemmedText = []
        for k in range(len(swRemovedText)):#Stemming
            stemmedText.append(stemmer.stem(swRemovedText[k]))
        cleanData.append(stemmedText)
        count += 1
        print(count, "data cleaned")
    return cleanData

def TransformToBigram(data):
    bigramData = []
    count = 0
    for i in range(len(data)):
        bigramPerData = []
        for j in range(len(data[i])-1):
            temp = data[i][j] + " " + data[i][j+1]
            bigramPerData.append(temp)
        bigramData.append(bigramPerData)
        count += 1
        print(count, "Data's feature transformed to bigram")
    return bigramData

def CreateBigram(data):
    bigram = []
    count = 0
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            tempBigram = data[i][j] + " " + data[i][j+1]
            if tempBigram not in bigram:
                bigram.append(tempBigram)
                count += 1
                print(count, "bigram created")
    return bigram

def CreateDF(data,doc):
    df = {}
    deletedDf = []
    count = 0
    for i in range(len(data)):
        for j in range(len(doc)):
            if data[i] in doc[j]:
                if data[i] in df:
                    df[data[i]] += 1
                else:
                    df[data[i]] = 1
        count += 1
        print(count, "df created")
        countTreshold = 0
    for i in data:
        if df[i] <= 2 or df[i] > 20: 
            deletedDf.append(i)
            del df[i]
            countTreshold += 1
            print(countTreshold, "treshold applied")
    return df,deletedDf

def CreateTFIDF(data,df,unigram,deletedDf):
    tfidf = []
    count = 0
    for i in range(len(data)):
        tempTfidf = []
        for j in range(len(unigram)):
            if unigram[j] in data[i] and unigram[j] not in deletedDf:
                tf = 0
                for k in range(len(data[i])):
                    if data[i][k] == unigram[j]:
                        tf += 1
                idf = math.log10(len(data)/df[unigram[j]])
                tempTfidf.append(idf*tf)
            else:
                tempTfidf.append(0)
        count += 1
        print(count, "tf-idf created")
        tfidf.append(tempTfidf)
    return tfidf

def Euclidean(a,b):
    distance = 0
    for i in range(len(a)):
        distance = distance + ((a[i]-b[i])**2)
    return math.sqrt(distance)

def KnnClassifier(K,train,test,actualLabel,data,cls):
    predict = []
    for i in range(len(test)):
        yes,no = 0,0
        distance = []
        for j in range(len(train)):
            tempDistance = Euclidean(test[i],train[j])
            distance.append([tempDistance,int(actualLabel[j])])
        distance.sort()
        for k in range(K):
            if distance[k][1] == 0:
                no += 1
            elif distance[k][1] == 1:
                yes += 1
        temp = 2
        if yes > no:
            predict.append(1)
            temp = 1
        else:
            predict.append(0)
            temp = 0
#        predict.append(label.index(max(label)))
#        predict.append(distance[0])
        print("K:",K,", split:",data,", class:",cls,", data ke:",i,", result:",temp)
        print("yes:",yes,"no:",no)
    return predict

def HammingLoss(actual,predict):
    value = 0
    for i in range(len(predict)):
        a,b,c,x,y,z = 0,0,0,0,0,0
        if predict[i][0] == 1 and actual[i][0] == 1:
            a = 1
        else:
            a = 0
        if predict[i][1] == 1 and actual[i][1] == 1:
            b = 1
        else:
            b = 0
        if predict[i][2] == 1 and actual[i][2] == 1:
            c = 1
        else:
            c = 0
        if predict[i][0] == 1 or actual[i][0] == 1:
            x = 1
        else:
            x = 0
        if predict[i][1] == 1 or actual[i][1] == 1:
            y = 1
        else:
            y = 0
        if predict[i][2] == 1 or actual[i][2] == 1:
            z = 1
        else:
            z = 0
        value = value + ((a + b + c) / (x + y +z))
    hammingLoss = 1-(1/len(predict)*value)
    return round(hammingLoss,6)

FileLoc = "data.xlsx"
rawData,actualLabel = LoadDataset(FileLoc)
cleanData = Preprocessing(rawData)
bigram = CreateBigram(cleanData)
bigramData = TransformToBigram(cleanData)
df,deletedDf = CreateDF(bigram,bigramData)
dataTfidf = CreateTFIDF(bigramData,df,bigram,deletedDf)

#spliting data
dataTrain1, dataTest1 = dataTfidf[0:798], dataTfidf[798:1064]
labelTrain1, labelTest1 = np.array(actualLabel[0:798]), np.array(actualLabel[798:1064])

dataTrain2, dataTest2 = dataTfidf[0:532]+dataTfidf[798:1064], dataTfidf[532:798]
labelTrain2, labelTest2 = np.array(actualLabel[0:532]+actualLabel[798:1064]), np.array(actualLabel[532:798])

dataTrain3, dataTest3 = dataTfidf[0:266]+dataTfidf[532:1064], dataTfidf[266:532]
labelTrain3, labelTest3 = np.array(actualLabel[0:266]+actualLabel[532:1064]), np.array(actualLabel[266:532])

dataTrain4, dataTest4 = dataTfidf[266:1064], dataTfidf[0:266]
labelTrain4, labelTest4 = np.array(actualLabel[266:1064]), np.array(actualLabel[0:266])

predict1A,predict1B,predict1C = [],[],[]
predict2A,predict2B,predict2C = [],[],[]
predict3A,predict3B,predict3C = [],[],[]
predict4A,predict4B,predict4C = [],[],[]

K = 3
result = []
while K < 33:
#iteration for classify the data
    tempResult = []
    for i in range(4):
        if i == 0:
            predict1A = KnnClassifier(K,dataTrain1,dataTest1,labelTrain1[:,0],1,"anjuran")
            predict1B = KnnClassifier(K,dataTrain1,dataTest1,labelTrain1[:,1],1,"larangan")
            predict1C = KnnClassifier(K,dataTrain1,dataTest1,labelTrain1[:,2],1,"informasi")
            predict = []
            for i in range(len(predict1A)):
                predict.append([predict1A[i],predict1B[i],predict1C[i]])
            hammingLoss = HammingLoss(labelTest1,predict)
            tempResult.append(hammingLoss)
        elif i == 1:
            predict2A = KnnClassifier(K,dataTrain2,dataTest2,labelTrain2[:,0],2,"anjuran")
            predict2B = KnnClassifier(K,dataTrain2,dataTest2,labelTrain2[:,1],2,"larangan")
            predict2C = KnnClassifier(K,dataTrain2,dataTest2,labelTrain2[:,2],2,"informasi")
            predict = []
            for i in range(len(predict2A)):
                predict.append([predict2A[i],predict2B[i],predict2C[i]])
            hammingLoss = HammingLoss(labelTest2,predict)
            tempResult.append(hammingLoss)
        elif i == 2:
            predict3A = KnnClassifier(K,dataTrain3,dataTest3,labelTrain3[:,0],3,"anjuran")
            predict3B = KnnClassifier(K,dataTrain3,dataTest3,labelTrain3[:,1],3,"larangan")
            predict3C = KnnClassifier(K,dataTrain3,dataTest3,labelTrain3[:,2],3,"informasi")
            predict = []
            for i in range(len(predict3A)):
                predict.append([predict3A[i],predict3B[i],predict3C[i]])
            hammingLoss = HammingLoss(labelTest3,predict)
            tempResult.append(hammingLoss)
        else:
            predict4A = KnnClassifier(K,dataTrain4,dataTest4,labelTrain4[:,0],4,"anjuran")
            predict4B = KnnClassifier(K,dataTrain4,dataTest4,labelTrain4[:,1],4,"larangan")
            predict4C = KnnClassifier(K,dataTrain4,dataTest4,labelTrain4[:,2],4,"informasi")
            predict = []
            for i in range(len(predict4A)):
                predict.append([predict3A[i],predict3B[i],predict3C[i]])
            hammingLoss = HammingLoss(labelTest4,predict)
            tempResult.append(hammingLoss)
    meanAcc = sum(tempResult) / len(tempResult)
    result.append([K,tempResult[0],tempResult[1],tempResult[2],tempResult[3],meanAcc])
    with open('result_bigram.csv', mode='w') as resultFile:
        resultWriter = csv.writer(resultFile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        resultWriter.writerow(['K value','Fold 1','Fold 2','Fold 3','Fold 4', 'Mean'])
        for i in range(len(result)):
            resultWriter.writerow([result[i][0],result[i][1],result[i][2],result[i][3],result[i][4],result[i][5]])
    K += 1

#predictA = predict4A + predict3A + predict2A + predict1A
#predictB = predict4B + predict3B + predict2B + predict1B
#predictC = predict4C + predict3C + predict2C + predict1C
#predict = []
#
#for i in range(len(predictA)):
#    predict.append([predictA[i],predictB[i],predictC[i]])
#
#hammingLost = HammingLoss(actualLabel,predict)
#print(round(hammingLost*100,2),"%")
print("--- %s seconds ---" % (time.time() - start_time))

#code below for count actual and predict each class
anjuranActual0, anjuranActual1 = 0,0
laranganActual0, laranganActual1 = 0,0
informasiActual0, informasiActual1 = 0,0
anjuranPredict0, anjuranPredict1 = 0,0
laranganPredict0, laranganPredict1 = 0,0
informasiPredict0, informasiPredict1 = 0,0
for i in range(len(predict)):
    if actualLabel[i][0] == 0:
        anjuranActual0 += 1
    if actualLabel[i][0] == 1:
        anjuranActual1 += 1
    if actualLabel[i][1] == 0:
        laranganActual0 += 1
    if actualLabel[i][1] == 1:
        laranganActual1 += 1
    if actualLabel[i][2] == 0:
        informasiActual0 += 1
    if actualLabel[i][2] == 1:
        informasiActual1 += 1
    if predict[i][0] == 0:
        anjuranPredict0 += 1
    if predict[i][0] == 1:
        anjuranPredict1 += 1
    if predict[i][1] == 0:
        laranganPredict0 += 1
    if predict[i][1] == 1:
        laranganPredict1 += 1
    if predict[i][2] == 0:
        informasiPredict0 += 1
    if predict[i][2] == 1:
        informasiPredict1 += 1
    













    
    
    
    
    
    
    

    