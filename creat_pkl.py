'''
将data文件夹下面的子文件夹0,1,2中的图片保存为.pkl
'''
import pickle    
import os    
import json  
import pylab
import numpy  
from PIL import Image

i = 0
'''
// r'data 数据保存路径。文件一共含有165张，每张大小40*40.
// data 则保存的就是这165张图片的信息。不需要人为新建文件，在程序中只是变量
// data_label 中包含的是165张图片的标签信息。不需要人为新建文件，在程序中只是变量
'''
data=numpy.empty((33,784))
data_label=numpy.empty(33)

#下面这函数是列出文件夹中所有的文件，到filename中
for filename in os.listdir(r'E:\tensorflow\data\0'):
    print(filename)
    if(filename!='Thumbs.db'):   
        basedir = 'E:/tensorflow/data/0/'
        imgage = Image.open(basedir + filename)
        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        data[i]=numpy.ndarray.flatten(img_ndarray)
        #标签要从0开始，不然在cnn训练时会有错误
        data_label[i]=0
        i = i + 1



for filename in os.listdir(r'E:\tensorflow\data\1'):
    print(filename)
    if(filename!='Thumbs.db'):   
        basedir = 'E:/tensorflow/data/1/'
        imgage = Image.open(basedir + filename)
        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        data[i]=numpy.ndarray.flatten(img_ndarray)  
        data_label[i]=1     
        i = i + 1


for filename in os.listdir(r'E:\tensorflow\data\2'):
    print(filename)
    if(filename!='Thumbs.db'):   
        basedir = 'E:/tensorflow/data/2/'
        imgage = Image.open(basedir + filename)
        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        data[i]=numpy.ndarray.flatten(img_ndarray)
        data_label[i]=2
        i = i + 1


'''
for filename in os.listdir(r'data\Expression 1\Happiness (HA)'):
    print filename
    if(filename!='Thumbs.db'):   
        basedir = 'data\Expression 1\Happiness (HA)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=3  
        i = i + 1

for filename in os.listdir(r'data\Expression 1\Sadness (SA)'):
    print filename
    if(filename!='Thumbs.db'):   
        basedir = 'data\Expression 1\Sadness (SA)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=4  
        i = i + 1

for filename in os.listdir(r'data\Expression 1\Surprise (SU)'):
    print filename
    if(filename!='Thumbs.db'):   
        basedir = 'data\Expression 1\Surprise (SU)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=5    
        i = i + 1
'''
data_label=data_label.astype(numpy.int)   
#下面是生成pkl格式的文件，保存数据。 
write_file=open('creat_pkl_data.pkl','wb')
pickle.dump(data,write_file,-1)  
pickle.dump(data_label,write_file,-1)
write_file.close()

#从pkl文件中读取数据显示图像和标签。
read_file=open('creat_pkl_data.pkl','rb')  
data=pickle.load(read_file)
label=pickle.load(read_file)
read_file.close() 
img0=data[1].reshape(28,28)
pylab.imshow(img0)
pylab.gray()
print(label[0:33])
pylab.show()