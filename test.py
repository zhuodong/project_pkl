"""
使用CNN检测数据集‘data.pkl’
修改程序时需要注意：载入数据，分类种类,在应用x和y_时，应当注意数据结构

运行结果：能够顺利运行
"""

import tensorflow as tf
import pickle
import numpy as np

sess=tf.InteractiveSession()
def load_data(data_file):
    read_file=open(data_file,'rb')  
    train,valid,test = pickle.load(read_file)
    read_file.close() 
    return train,valid,test
#定义权重和偏差的初始化函数
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义卷积层和池化层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def cov_label(data):
    """
    对数据标签进行one_hot编码
    """
    from numpy import argmax
    #标签的种类
    alphabet = '012'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [int(char) for char in data]
    #print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
       letter = [0 for _ in range(len(alphabet))]
       letter[value] = 1
       onehot_encoded.append(letter)
    #print(onehot_encoded)
    return onehot_encoded

    # 解码标签
    #inverted = int_to_char[argmax(onehot_encoded[0])]
   #print(inverted)
#每次随机提取数据集样本
"""
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
"""

def next_batch(data, label, batch_size, shuffle=True):

    data_size = len(label)
    num_batches_per_epoch = int(data_size / batch_size) 
    #for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        shuffled_label = label[shuffle_indices]
    else:
        shuffled_data = data
        shuffled_label = label

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        yield shuffled_data[start_index:end_index],shuffled_label[start_index:end_index]

#定义输入的placeholder，x是特征，y_是真实的label。
# 因为卷积神经网络是会用到2D的空间信息，所以要把784维的数据恢复成28*28的结构
x = tf.placeholder(tf.float32,[None,784])#28*28
y_ = tf.placeholder(tf.float32,[None,3])#3类
x_image = tf.reshape(x,[-1,28,28,1])#恢复数据类型

#定义第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2_2(h_conv1)

#定义第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2_2(h_conv2)


#定义第一个全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#根据之前讲过的，在训练数据比较小的情况下，为了防止过拟合，随机的将一些节点置0，增加网络的泛化能力

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#最后一个输出层也要对权重和偏差进行初始化

W_fc2 = weight_variable([1024,3])###类别数
b_fc2 = bias_variable([3])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#定义损失函数和训练的步骤，使用Adam优化器最小化损失函数

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#计算预测的精确度
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#读入数据集
data_file = 'data.pkl'
train_set,valid_set,test_set = load_data(data_file)

test_data,test_label = test_set
train_data,train_label = train_set

#对全局的变量进行初始化
tf.global_variables_initializer().run()
#进行训练

for i in range(1000):

    for batch_xs, batch_ys in next_batch(train_data,train_label, 50):       
        #print(batch_ys)
        batch_xs = batch_xs
        batch_ys = cov_label(batch_ys)#要对函数的对应标签处进行修改
        #print(batch_ys)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch_xs, y_: np.reshape(batch_ys,(-1,3)), keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict = {x: batch_xs, y_:np.reshape(batch_ys,(-1,3)), keep_prob: 0.5})

#输出最后的准确率
print(i)
test_label= cov_label(test_label)
print("test accuracy %g"%accuracy.eval(feed_dict = {x: test_data, y_:np.reshape( test_label,(-1,3)), keep_prob: 1.0}))

