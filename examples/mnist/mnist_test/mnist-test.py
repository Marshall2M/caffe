import caffe

# 1. 用deploy.prototxt和训练好的caffemodel初始化Classifier，并返回net
MODEL_FILE = r"C:/Caffe/lenet.prototxt"
PRETRAINED = r"C:/Caffe/lenet_iter_10000.caffemodel"
net = caffe.Classifier(model_file = MODEL_FILE, pretrained_file = PRETRAINED)

# 2. 读入待预测图片，并加入列表
input_image = []
input_image.append(caffe.io.load_image(r"C:/Caffe/caffe/examples/mnist/mnist_test/train/images/train_0_label_5.bmp", color=False))
input_image.append(caffe.io.load_image(r"C:/Caffe/caffe/examples/mnist/mnist_test/train/images/train_1_label_0.bmp", color=False))

# 3. 预测，predict函数会自动根据转Classifier的入参换图片格式
prediction = net.predict(input_image, oversample=False)

# 4. 输出结果，argmax是输出最大值
# print("#1st#", str(prediction[0].argmax()), "; #2nd#", str(prediction[1].argmax()))
for i in range(len(input_image)):
    print(i, "#", str(prediction[i].argmax()))
    result = prediction[i].flatten().argsort() #将预测从小到大排序
    print(result[::-1])