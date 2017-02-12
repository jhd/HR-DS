from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas
import sklearn.metrics
import sys
import time

def createModel(learn_rate=0.1, decay = 0.00001):
	reg = 0.00000
	nnModel = Sequential()
	nnModel.add(Dense(output_dim=1000, input_shape=(9,), W_regularizer=l2(reg)))
	nnModel.add(Activation("relu"))
	nnModel.add(Dense(output_dim=1000, input_dim=1000, W_regularizer=l2(reg)))
	nnModel.add(Activation("relu"))
	nnModel.add(Dense(output_dim=2, input_dim=1000, W_regularizer=l2(reg)))
	nnModel.add(Activation("softmax"))
	nnModel.summary()
	sgd = SGD(lr=learn_rate, momentum=0.0, decay=decay, nesterov=False)
	nnModel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'fmeasure'])
	return nnModel

allData = pandas.read_csv('readyData.csv')
del allData['Unnamed: 0']
scrambledData = allData.sample(frac=1).reset_index(drop=True)
trainingData = scrambledData[:12000]
testData = scrambledData[12000:]
trainingLabels = trainingData.label
testLabels = testData.label
del trainingData['label']
del testData['label']

model = KerasClassifier(build_fn=createModel, verbose=1, nb_epoch = 1000, batch_size = 32)

learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3, 1]
decay = [0.0, 0.00001, 0.0001, 0.001, 0.01]
param_grid = dict(learn_rate=learn_rate, decay=decay)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
start = time.time()

grid_result = grid.fit(np.array(trainingData), np.array(trainingLabels))
pred = grid.predict(np.array(testData))

end = time.time()
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print(end - start)
print (pred)
f1 = sklearn.metrics.f1_score(testLabels, pred)
print (f1)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
sys.exit(0)