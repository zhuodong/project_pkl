import pickle    
import os    
import json  
import pylab
import numpy  
from PIL import Image

i = 0
import os
imageFile=r'E:\tensorflow\PKL\data'
num = (sum([len(x) for _, _, x in os.walk(imageFile)]))
#r'data\Expression 1 文件一共含有165张，每张大小40*40.
#olivettifaces 则保存的就是这165张图片的信息。
#olivettifaces_label 中包含的是165张图片的标签信息

olivettifaces=numpy.empty((num,784))
olivettifaces_label=numpy.empty(num)

#下面这函数是列出文件夹中所有的文件，到filename中
for filename in os.listdir(r'data\0'):
    #print(filename)
    if(filename!='Thumbs.db'):   
        basedir = 'data/0/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)
        # 标签要从0开始，不然在cnn训练时会有错误
        olivettifaces_label[i]=0
        i = i + 1



for filename in os.listdir(r'data\1'):
    #print(filename)
    if(filename!='Thumbs.db'):   
        basedir = 'data/1/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)  

        olivettifaces_label[i]=1     
        i = i + 1


for filename in os.listdir(r'data\2'):
    #print(filename)
    if(filename!='Thumbs.db'):   
        basedir = 'data/2/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=2
        i = i + 1

olivettifaces_label=olivettifaces_label.astype(numpy.int)   
#下面是生成pkl格式的文件，保存数据。 
write_file=open('olivettifaces.pkl','wb')
pickle.dump([[olivettifaces[0:164:3],olivettifaces_label[0:164:3]],
             [olivettifaces[0:164:5],olivettifaces_label[0:164:5]],
             [olivettifaces[0:164:7],olivettifaces_label[0:164:7]]],
             write_file,-1)  
write_file.close()

read_file=open('olivettifaces.pkl','rb')  
train_set,valid_set,test_set = pickle.load(read_file)
read_file.close() 
train,label = train_set
img0=train[4].reshape(28,28)
pylab.imshow(img0)
pylab.gray()
print(label[0:100]) 
pylab.show()