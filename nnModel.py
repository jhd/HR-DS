from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas
import sklearn.metrics
import sys

allData = pandas.read_csv('readyData.csv')
del allData['Unnamed: 0']
scrambledData = allData.sample(frac=1).reset_index(drop=True)
trainingData = scrambledData[:12000]
testData = scrambledData[12000:]
trainingLabels = trainingData.label
testLabels = testData.label
del trainingData['label']
del testData['label']
trainingData = PCA(n_components=6).fit_transform(trainingData)
testData = PCA(n_components=6).fit_transform(testData)
reg = 0.00000
nnModel = Sequential()
nnModel.add(Dense(output_dim=100, input_shape=(6,), W_regularizer=l2(reg)))
nnModel.add(Activation("relu"))
nnModel.add(Dense(output_dim=100, input_dim=100, W_regularizer=l2(reg)))
nnModel.add(Activation("relu"))
nnModel.add(Dense(output_dim=2, input_dim=100, W_regularizer=l2(reg)))
nnModel.add(Activation("softmax"))
nnModel.summary()
sgd = SGD(lr=0.1, momentum=0.0, decay=0.00001, nesterov=False)
import time

start = time.time()
nnModel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['fmeasure'])
nnModel.fit(np.array(trainingData), to_categorical(trainingLabels), batch_size=32,nb_epoch=10, shuffle=True)
pred = nnModel.predict_classes(np.array(testData))
end = time.time()
print(end - start)
print (pred)
f1 = sklearn.metrics.f1_score(testLabels, pred)
print (f1)
sys.exit(0)