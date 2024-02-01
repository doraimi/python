from ClassTestA import ClassA
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random


def runTester():
    a=1
    T_a=torch.tensor(a)
    print(T_a)

    #b=['a','b','c']
    b=[1,2,3]
    T_b=torch.tensor(b)
    print(f"T_b : {T_b}")
    print(f"T_b.shape : {T_b.shape}")
    print(f"T_b.ndim : {T_b.ndim}")
    print(f"T_b.size : {T_b.size}")
    print(f"T_b.dtype : {T_b.dtype}")


    c=[[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]],[[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]]
    print(c)
    T_c=torch.tensor(c)
    #this is error : print(f"c.shape : {c.shape}")
    print(f"T_c.shape : {T_c.shape}")
    #this is error : print(f"c.ndim : {c.ndim}")
    print(f"T_c.ndim : {T_c.ndim}")

def testNN():
    ker=torch.ones(5,5,dtype=torch.float32)
    ker[2,2]=24
    ker=ker.reshape((1,1,5,5))
    picA=Image.open("testCatPic.jpg")
    picAarray=np.array(picA.convert("L"),dtype=np.float32)
    hh,ww=picAarray.shape
    imgA=torch.from_numpy(picAarray.reshape((1,1,hh,ww)))

    #pool2dA=nn.maxPool2(picA)
    conv2d= nn.Conv2d(1,2,(5,5),bias= False)
    conv2d.weight.data=ker
    
    imgC2=conv2d(imgA)
    imconv2dout_im = imgC2.data.squeeze()
    print(imgC2)
    plt.subplot(1,3,1)
    plt.imshow(picA)
    #print(f"picA.shape is {picA.shape}")
    plt.subplot(1,3,2)
    plt.imshow(imconv2dout_im)
    print(f"imconv2dout_im.shape is {imconv2dout_im.shape}")
    #Image.save("./testCatPic_output.jpg")
    plt.subplot(1,3,3)
    plt.imshow(picA)
    plt.show()

def randomTester():
    i=1
    while i<100:
        print(f"  {random.randint(0, 200)}")
        i+=1

def findMax():
    prediction = torch.tensor([0.1, 0.8, 0.3, 0.5 ,3,6,7,8,29,32,12])

    # Find the index of the maximum value
    max_index = torch.argmax(prediction).item()

    print("Index of maximum value:", max_index)
    print("Index of maximum value:", prediction[max_index])

if __name__=="__main__":

    p=257%255
    print (p)
    print (f"runTester __name__{__name__}")
    #runTester()

    #testNN()
    #randomTester()
    findMax()
