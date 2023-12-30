#!/usr/bin/env python
# coding: utf-8

# # Turning Neuromorphic Dynex Chips into Tensorflow Layers

import math
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from QSVM_Layer import QSVM_Layer

# create a tensorflow dataset
class BankDataset():
	def __init__(self, data_file):
		"""
        this function define the BankDataset class from a text or CSV file 

        Parameters
        ----------
        data_file : the data file to be loaded

        """
		training_data = np.loadtxt('./datasets/{}'.format(data_file), delimiter=',')
		for i in range(len(training_data)):
			if(training_data[i][-1] == 0):
				training_data[i][-1] = -1
		data = training_data[:, :2]
		target = training_data[:, -1]
		x_min, x_max = 1000, 0
		y_min, y_max = 1000, 0
    	# rescalling data
		for i in range(len(training_data)):
			x_min = min(data[i][0], x_min)
			x_max = max(data[i][0], x_max)
			y_min = min(data[i][1], y_min)
			y_max = max(data[i][1], y_max)
		for i in range(len(training_data)):
			data[i][0] = (data[i][0] - x_min)/(x_max - x_min)
			data[i][1] = (data[i][1] - y_min)/(y_max - y_min)
        
		self.data = data
		self.target = target	

	def to_tf_dataset(self, batch_size=1, shuffle=True):
		dataset = tf.data.Dataset.from_tensor_slices((self.data, self.target))
		if shuffle:
			dataset = dataset.shuffle(buffer_size=len(self.data))
		dataset = dataset.batch(batch_size)
		return dataset


###################################################

def plot_figure(SVM,dataset,train_percent,sampler_type, img):
	"""
    This function plot a contour image for dataset.
    Parameters:
    - SVM: the trained SVM model.
    - dataset: dataset for train and test.
    - train_percent: the percentage of dataset size for training.
    - sampler_type: sampler type.
    - img: Path to save the image.
    """
	plt.figure()
	cm = plt.cm.RdBu
	data = dataset.data
	t = dataset.target
	N = int(len(dataset.data)*train_percent)
	xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 80), np.linspace(0.0, 1.0, 80))
	Z = []
	for row in range(len(xx)):
		Z_row = []
		for col in range(len(xx[row])):
			target = np.array([xx[row][col], yy[row][col]])
			Z_row.append(SVM(target))
		Z.append(Z_row)
	
	cnt = plt.contourf(xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=0.8, extend="both")
	plt.contour(xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,))
	plt.colorbar(cnt, ticks=[-1, 0, 1])

	red_sv = []
	blue_sv = []
	red_pts = []
	blue_pts = []

	for i in range(N):
		if(SVM.alpha[i]):
			if(t[i] == 1):
				blue_sv.append(data[i, :2])
			else:
				red_sv.append(data[i, :2])
		else:
			if(t[i] == 1):
				blue_pts.append(data[i, :2])
			else:
				red_pts.append(data[i, :2])

	plt.scatter([el[0] for el in blue_sv],
                [el[1] for el in blue_sv], color='b', marker='^', edgecolors='k', label="Type 1 SV")

	plt.scatter([el[0] for el in red_sv],
                [el[1] for el in red_sv], color='r', marker='^', edgecolors='k', label="Type -1 SV")

	plt.scatter([el[0] for el in blue_pts],
                [el[1] for el in blue_pts], color='b', marker='o', edgecolors='k', label="Type 1 Train")

	plt.scatter([el[0] for el in red_pts],
                [el[1] for el in red_pts], color='r', marker='o', edgecolors='k', label="Type -1 Train")    
	plt.legend(loc='lower right', fontsize='x-small')
	plt.savefig(f'{img}.jpg')


def predict(model,model_file,dataset):
	"""
        This function predict a trained SVM result for a dataset.

        Parameters:
            - model: the trained SVM model.
            - model_file: the file saved a trained model.
            - dataset: the dataset for predict.
    """
	model.load_model(model_file)
	tp, fp, tn, fn = 0, 0, 0, 0
	for (x, y) in dataset:
		# send the input to the device
		x = tf.reshape(x, [x.shape[1], x.shape[2]])
		# perform a forward pass and calculate the training loss
		pred = model(x);
		if(y == 1):
			if(pred > 0):
				tp += 1
			else:
				fp += 1
		else:
			if(pred < 0):
				tn += 1
			else:
				fn += 1

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f_score = tp/(tp + 1/2*(fp+fn))
	accuracy = (tp + tn)/(tp+tn+fp+fn)
	print("precision result:")
	print(f"{precision=} {recall=} {f_score=} {accuracy=}") 

##############################################
# Initialize the BankDataset
bank_dataset = BankDataset(data_file='banknote.txt')

# Split the dataset into train and test sets
train_percent = 0.8
train_size = int(len(bank_dataset.data) * train_percent)
test_size = len(bank_dataset.data) - train_size

# Using TensorFlow for splitting the dataset
#full_dataset = tf.data.Dataset.from_tensor_slices((bank_dataset.data, bank_dataset.labels))
full_dataset = bank_dataset.to_tf_dataset(batch_size=1, shuffle=True)

train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)

# Convert to TensorFlow datasets with batching
train_dataset = train_dataset.batch(1).shuffle(train_size)
test_dataset = test_dataset.batch(1)

B = 2;
K = 2;
C = 3;
gamma = 16;
xi = 0.001;
spl = "DNX";     
device = "cpu" # no GPU used for Dynex only
mainnet = False
num_reads=20000 
annealing_time = 200

class QSVM_Model(tf.keras.Model):
  def __init__(self, B,K,C,gamma,xi,bank_dataset,train_percent,spl,mainnet,num_reads,annealing_time):
    
    super(QSVM_Model, self).__init__()
    self.qsvm_layer = QSVM_Layer(B,K,C,gamma,xi,bank_dataset,train_percent,spl,mainnet,num_reads,annealing_time)

  def call(self, x):
    x = self.qsvm_layer(x) 
    return x


### train a new model on the train dataset
model = QSVM_Model(B,K,C,gamma,xi,bank_dataset,train_percent,spl,mainnet,num_reads,annealing_time)
EPOCHS = 1
for e in range(0, EPOCHS):
	print("training a new model...")
	print('EPOCH',e+1,'of',EPOCHS);
	tp, fp, tn, fn = 0, 0, 0, 0
	# training a model on dataset
	model.qsvm_layer.train(save_model=True, save_path='./models', model_file='QSVM.model')
	print("training end")
	# predict on test dadaset
	for (x, y) in test_dataset:
		## remove batch dim
		x = tf.reshape(x, [x.shape[1], x.shape[2]])
		# perform a forward pass
		pred = model(x);
		if(y == 1):
			if(pred > 0):
				tp += 1
			else:
				fp += 1
		else:
			if(pred < 0):
				tn += 1
			else:
				fn += 1
	print("test dataset result:")				
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f_score = tp/(tp + 1/2*(fp+fn))
	accuracy = (tp + tn)/(tp+tn+fp+fn)
	print(f"{precision=} {recall=} {f_score=} {accuracy=}")

plot_figure(model.qsvm_layer,bank_dataset,train_percent,spl,"img")


## load a saved model to predict on test dataset
print("make a predict on a saved model...")
predict(model.qsvm_layer, './models/QSVM.model', test_dataset)