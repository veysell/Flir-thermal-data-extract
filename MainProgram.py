from math import degrees
import matplotlib.pyplot as plt
from posixpath import basename
from sre_constants import RANGE
from tkinter import filedialog
from FlirLib import flir
import numpy as np
import sys
from PIL import Image

#region
# def calistir(fileName,asdf):
#     thermogram=flir.unpack(fileName)
#     thermal=thermogram.celsius # 0-255 arası cast et
#     print(thermal)
#     print(asdf)

# if __name__=="__main__":
#     asdf=str(sys.argv[1])
#     name=str(sys.argv[2])
#     calistir(name,asdf)
#endregion


    
    

# def threshold(FileName):
#     therm=flir.unpack(FileName)
#     thermal=therm.celsius
#     threshold_1=np.array(thermal.max()*0.2)
#     threshold_2=np.array(thermal.max()*0.5)
#     thermalDat=np.array(thermal)
#     thermalRoot=np.array(thermal)
#     row,col=thermalDat.shape
#     for i in range(0,row):
#         for j in range(0,col):
#             if(thermalDat[i][j]<threshold_1):
#                 thermalRoot[i][j]=0
#             elif(thermalDat[i][j]<threshold_2):
#                 thermalRoot[i][j]=threshold_1
#             else:
#                 thermalRoot[i][j]= threshold_2
#     image=Image.fromarray(thermalRoot)
#     image.show()
        
def saveCSV(FileName):
    therm=flir.unpack(FileName)
    thermal=therm.celsius
    name=basename(FileName) 
    np.savetxt("{}.csv".format(name),thermal,delimiter=",")




def linear(FileName)->Image:
    therm=flir.unpack(FileName)
    thermal=therm.celsius
    themin = np.array(thermal.min())
    themax = np.array(thermal.max())
    therange = themax - themin
    thermallinear = 255 * (thermal - themin) / therange
    image=Image.fromarray(thermallinear)
    image.show()
    return thermallinear

def differenceThreshold(FileName):
    therm=flir.unpack(FileName)
    thermal=therm.celsius
    themin=np.min(thermal) 
    if(themin<50):
         themin=themin
    else:
       themin=50
    difference=np.max(thermal)-themin
    factor=1/np.std(thermal)
    for i in range(len(thermal)):
        for j in range(len(thermal[i])):
            if(thermal[i][j]<difference):
                thermal[i][j]=thermal[i][j]*factor
            else:
                thermal[i][j]=thermal[i][j]*(difference/np.std(thermal)) 
    image=Image.fromarray(thermal)
    image.show()





# SAVECSV="1"
# LINEAR="2"
# DIFFERENCE_TRESHOLD="3"
# if __name__ == "__main__":
#     _function=str(sys.argv[1])
#     name=str(sys.argv[2])
#     if(_function=="1"):
#         saveCSV(name)
#     elif(_function=="2"):
#         linear(name)
#     elif(_function=="3"):
#         differenceThreshold(name)
#     else:
#         pass


def minimalFalse(fileName):
    thermal=flir.unpack(fileName)
    thermalData=thermal.celsius
    ıAvg=np.average(thermalData)
    ıStd=np.std(thermalData)
    ıMin = np.min(thermalData)
    tHeat=ıAvg+2*ıStd
    ıMax=np.max(thermalData)
    ıLmax=179
    ıYmax=235
    tHot=0
    if(ıMax<ıLmax):
        tHot=0.5*(ıAvg+3*ıStd)+0.5*ıMax
    else:
        tHot=0.5*(ıAvg+3*ıStd)+0.5*ıYmax
    
    difference=np.max(thermalData)-ıMin
    factor=1/np.std(thermalData)
    thermalData1=thermalData
    thermalData2=thermalData
    for i in range(len(thermalData2)):
        for j in range(len(thermalData2[i])):
            if(thermalData2[i][j]<tHot):
                thermalData2[i][j]=thermalData2[i][j]*factor
            else:
                thermalData2[i][j]=thermalData2[i][j]*(difference/np.std(thermalData2)) 
    image=Image.fromarray(thermalData2)
    image.show()
    for i in range(len(thermalData1)):
        for j in range(len(thermalData1[i])):
            if(thermalData1[i][j]<tHeat):
                thermalData1[i][j]=thermalData1[i][j]*factor
            else:
                thermalData1[i][j]=thermalData1[i][j]*(difference/np.std(thermalData1))
    image2=Image.fromarray(thermalData1)
    image2.show()


fileName=filedialog.askopenfilename()
minimalFalse(fileName)


# buraya np.array.savetxt nesnesine path verilebiliyor mu ona bakılacak
# 

