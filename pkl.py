'''
将data文件夹下面的子文件夹0,1,2中的图片保存为.pkl
imageFile 数据保存路径。
data 则保存图片的信息。不需要人为新建文件，在程序中只是变量
data_label 中包含图片的标签信息。不需要人为新建文件，在程序中只是变量
'''
import pickle    
import os    
import json  
import pylab
import numpy  
from PIL import Image
import numpy as np

"""
对数据进行打包
"""
def trans_data(imageFile,img_num,label_num):
    data = np.empty((img_num,784))
    data_label = np.empty(img_num)
    #数据的分类标签
    classes = ['0','1','2']
    i = 0
    j = 0
    for num in range(label_num):
        classfile = imageFile + '\\' + classes[j]
        classfile = classfile.replace('\\','/' )
        
        for filename in os.listdir(classfile):
        #显示单个文件夹中的所有图片名称
            if(filename!='Thumbs.db'):            
                basedir = classfile + '/'            
                imgage = Image.open(basedir + filename)            
                img_ndarray = numpy.asarray(imgage, dtype='float64')/256
                img_ndarray.reshape(28,28)
                data[i] = numpy.ndarray.flatten(img_ndarray)
                #标签要从0开始，不然在cnn训练时会有错误
                data_label[i] = j 
                i = i+1 
        j = j+1
    data_label = np.array(data_label)
    return data,data_label
"""
#将数据和标签压缩为一个文件
#下面是生成pkl格式的文件，保存数据。 
"""
def pic_pkl(data,data_label):

    # 将图片与标签保存为一个文件，加载
    #压缩文件时，需要给两个变量分别保存这两个参数，数据和标签
    '''
    write_file=open('creat_pkl_data.pkl','wb')
    pickle.dump([data,data_label],write_file,-1)  
    write_file.close()
    '''
    #将数据压缩为三个部分，训练集，验证集，测试集，
    #每个变量包含两个部分 数据和标签,确定三个文件夹中数据的个数

    write_file=open('data.pkl','wb')
    pickle.dump([[data[0:2499:3],data_label[0:2499:3]],
             [data[0:2499:5],data_label[0:2499:5]],
             [data[0:2499:7],data_label[0:2499:7]]],
             write_file,-1)  
    write_file.close()

if __name__ == "__main__":
   
    #数据保存路径
    imageFile=r'E:\tensorflow\PKL\data'
    import os
    img_num = (sum([len(x) for _, _, x in os.walk(imageFile)]))#输出目标文件夹下的文件数目
    label_num = (sum([len(x) for x in os.listdir(imageFile)]))
    print('数据的总数: %d'%img_num)
    print('数据的种类: %d'%label_num)
    #将数据便形成所需形式，数据存储在data里表中，标签存储在data_label中
    data,data_label = trans_data(imageFile,img_num,label_num)
    #选择压缩形式，需要依据样本数目确定三种数据的个数
    pic_pkl(data,data_label)
   
    '''
    #对单个数据进行测试,从pkl文件中读取数据显示图像和标签
    read_file = open('creat_pkl_data.pkl','rb')  
    data,label = pickle.load(read_file)
    read_file.close() 

    img0=data[50].reshape(28,28)
    print(label[12])
    pylab.imshow(img0)
    pylab.gray()
    pylab.show()
    '''
    #对单个数据进行测试
    read_file=open('data.pkl','rb')  
    train_set,valid_set,test_set=pickle.load(read_file)
    read_file.close() 

    train,label = test_set
    
    img0=train[4].reshape(28,28)
    pylab.imshow(img0)
    print('目标图片种类: %d'%label[4])
    #显示各个数据集的数目
    print('训练数据的总数: %d'%len(train_set[1]))
    print('验证数据的总数: %d'%len(valid_set[1]))
    print('测试数据的总数: %d'%len(test_set[1]))
    pylab.gray()
    pylab.show()


