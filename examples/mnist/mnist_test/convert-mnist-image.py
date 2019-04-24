import numpy as np
import struct
from PIL import Image
import os

caffe_root = r"C:/Caffe/caffe/"
caffe_mnist_data = caffe_root + r"examples/mnist/mnist_test/"

test_MNIST_data = caffe_mnist_data + r"test/t10k-images.idx3-ubyte"
train_MNIST_data = caffe_mnist_data + r"train/train-images.idx3-ubyte"
test_MNIST_label = caffe_mnist_data + r"test/t10k-labels.idx1-ubyte"
train_MNIST_label = caffe_mnist_data + r"train/train-labels.idx1-ubyte"

### LABEL ###
index = 0
print("TEST LABEL DATA:" + test_MNIST_label)
binfile = open(test_MNIST_label, 'rb')
buf = binfile.read()
magic, numImages = struct.unpack_from('>II', buf, index)
print(magic, numImages)
index += struct.calcsize('>II')

label_test = []
for i in range(numImages):
    if i%1000 == 0:
        print("test label:" + str(i))
    (l,) = struct.unpack_from('>B', buf, index)
    index += struct.calcsize('>B')
    label_test.append(l)

index = 0
print("TRAIN LABEL DATA:" + train_MNIST_label)
binfile = open(train_MNIST_label, 'rb')
buf = binfile.read()
magic, numImages = struct.unpack_from('>II', buf, index)
print(magic, numImages)
index += struct.calcsize('>II')

label_train = []
for i in range(numImages):
    if i%1000 == 0:
        print("test label:" + str(i))
    (l,) = struct.unpack_from('>B', buf, index)
    index += struct.calcsize('>B')
    label_train.append(l)

### DATA ###
# TEST
index = 0
print("TEST IMAGE DATA:" + test_MNIST_data)
binfile = open(test_MNIST_data, 'rb')
buf = binfile.read()
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index) #4 * Int
print(magic, numImages, numRows, numColumns)
index += struct.calcsize('>IIII')

FileDir = caffe_mnist_data + r"test/images/"
print(FileDir)
for i in range(numImages):
    if i%1000 == 0:
        print("test data:" + str(i))

    im = struct.unpack_from('>784B', buf, index)    #28*28=784 Byte
    index += struct.calcsize('>784B')
    im = np.array(im)
    im = im.reshape(28, 28)

    picName = FileDir + r'test_%d_label_%d.bmp'%(i,label_test[i])
    image = Image.new("L", (28,28))
    for a in range(28):
        for b in range(28):
            image.putpixel((a,b), int(im[b][a]))
    image.save(picName, 'bmp')


# TRAIN
index = 0
print("TRAIN IMAGE DATA:" + train_MNIST_data)
binfile = open(train_MNIST_data, 'rb')
buf = binfile.read()
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
print(magic, numImages, numRows, numColumns)
index += struct.calcsize('>IIII')

FileDir = caffe_mnist_data + r"train/images/"
print(FileDir)
ls = os.listdir(FileDir)
if len(ls) >= numImages:
    numImages = 0
for i in range(numImages):
    if i%1000 == 0:
        print("train:" + str(i))

    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    im = np.array(im)
    im = im.reshape(28, 28)

    picName = FileDir + r'train_%d_label_%d.bmp'%(i, label_train[i])
    image = Image.new("L", (28,28))
    for a in range(28):
        for b in range(28):
            image.putpixel((a,b), int(im[b][a]))
    image.save(picName, 'bmp')