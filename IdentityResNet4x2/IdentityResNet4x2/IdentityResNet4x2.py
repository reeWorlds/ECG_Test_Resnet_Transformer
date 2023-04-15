import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import ModelCheckpoint

import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_addons as tfa
import sklearn
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from scipy.optimize import minimize


def testGPU():
	print(tf.config.list_physical_devices())

	if tf.config.list_physical_devices():
		print("GPU is available:")
		devices = device_lib.list_local_devices()
		for device in devices:
			if device.device_type == 'GPU':
				device_name, memory_limit = device.name, device.memory_limit / (1024**3)
				print (f"Device name: {device_name}, Memory limit: {memory_limit} GB")
	else:
		print("GPU is not available")


def readTrainData():
	trainData = tf.convert_to_tensor(pd.read_csv('../../data/trainData.csv', header=None), dtype=tf.float32)
	trainRef = keras.utils.to_categorical(tf.convert_to_tensor(pd.read_csv('../../data/trainRef.csv', header=None).transpose(), dtype=tf.float32), 3)
	trainMap = pd.read_csv('../../data/trainMap.csv', header=None).transpose()

	return trainData, trainRef, trainMap

def readValidData():
	validData =  tf.convert_to_tensor(pd.read_csv('../../data/validData.csv', header=None), dtype=tf.float32)
	validRef =  keras.utils.to_categorical(tf.convert_to_tensor(pd.read_csv('../../data/validRef.csv', header=None).transpose(), dtype=tf.float32), 3)

	return validData, validRef

def readTestData():
	testData =  tf.convert_to_tensor(pd.read_csv('../../data/testData.csv', header=None), dtype=tf.float32)
	testMap =  pd.read_csv('../../data/testMap.csv', header=None).transpose()
	testRef = pd.read_csv('../../data/testRef.csv', header=None)
	
	return testData, testMap, testRef


def getScedule(initial_learning_rate, datasetLen, minibatch, epochN, num_waves = 5, warmup_epochs = 4):
	min_learning_rate = 2.5e-6

	total_steps = (datasetLen + minibatch - 1) // minibatch * epochN
	warmup_steps = (datasetLen + minibatch - 1) // minibatch * warmup_epochs
	remaining_steps = total_steps - warmup_steps

	steps_per_wave = remaining_steps // num_waves + 1

	warmup_start_lr = initial_learning_rate * 0.1
	warmup_increment = (initial_learning_rate - warmup_start_lr) / warmup_steps

	def warmup_schedule(step):
		return warmup_start_lr + warmup_increment * step

	cosine_schedule = schedules.CosineDecayRestarts(initial_learning_rate, steps_per_wave, t_mul=1.0, m_mul=1.0, alpha=0.0)

	def cosine_schedule_with_min(step):
		lr = cosine_schedule(step)
		return tf.maximum(lr, min_learning_rate)

	def lr_schedule_with_warmup(step):
		return tf.cond(step < warmup_steps, lambda: warmup_schedule(step), lambda: cosine_schedule_with_min(step - warmup_steps))

	return lr_schedule_with_warmup

def plot_lr_schedule(path):
	total_steps = (68062 + 256 - 1) // 256 * 164
	lr_schedule = getScedule(1e-3, 68062, 256, 164)

	learning_rates = []

	steps = np.arange(total_steps)
	for step in range(total_steps):
		step_tensor = tf.constant(step, dtype=tf.float32)
		lr = lr_schedule(step_tensor).numpy()
		learning_rates.append(lr)

	plt.figure(figsize=(10, 5))
	plt.plot(steps, learning_rates)
	plt.xlabel('Steps')
	plt.ylabel('Learning Rate')
	plt.yscale('log')
	plt.title('Learning Rate Schedule')
	
	plt.savefig(path, dpi=300, bbox_inches='tight')
	plt.close()


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, initial_learning_rate, datasetLen, minibatch, epochN, num_waves=5, warmup_epochs=4):
		super(CustomLearningRateSchedule, self).__init__()
		self.initial_learning_rate = initial_learning_rate
		self.datasetLen = datasetLen
		self.minibatch = minibatch
		self.epochN = epochN
		self.num_waves = num_waves
		self.warmup_epochs = warmup_epochs
		self.lr_schedule = getScedule(initial_learning_rate, datasetLen, minibatch, epochN, num_waves, warmup_epochs)

	def __call__(self, step):
		return self.lr_schedule(step)

	def get_config(self):
		return {
			"initial_learning_rate": self.initial_learning_rate,
			"datasetLen": self.datasetLen,
			"minibatch": self.minibatch,
			"epochN": self.epochN,
			"num_waves": self.num_waves,
			"warmup_epochs": self.warmup_epochs,
		}

class CustomAdamW(tfa.optimizers.AdamW):
	def __init__(self, learning_rate, weight_decay_factor, **kwargs):
		kwargs.pop("weight_decay", None)
		super(CustomAdamW, self).__init__(learning_rate=learning_rate, weight_decay=0, **kwargs)
		self.weight_decay_factor = weight_decay_factor

	def _current_learning_rate(self, var_dtype):
		step = tf.cast(self.iterations, var_dtype)
		return self._get_hyper("learning_rate", var_dtype)(step)

	def _resource_apply_dense(self, grad, var, apply_state=None):
		weight_decay = self.weight_decay_factor * self._current_learning_rate(var.dtype)
		if apply_state is None:
			apply_state = {}
		apply_state["weight_decay"] = weight_decay
		return super(CustomAdamW, self)._resource_apply_dense(grad, var, apply_state=apply_state)

	def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
		weight_decay = self.weight_decay_factor * self._current_learning_rate(var.dtype)
		if apply_state is None:
			apply_state = {}
		apply_state["weight_decay"] = weight_decay
		return super(CustomAdamW, self)._resource_apply_sparse(grad, var, indices, apply_state=apply_state)

	def get_config(self):
		config = super().get_config()
		config["weight_decay_factor"] = self.weight_decay_factor
		return config


def getModel(custom_lr_schedule, decay):
	opt = CustomAdamW(learning_rate=custom_lr_schedule, weight_decay_factor=decay)

	inputs = tf.keras.Input(shape=(1500, 1))

	x = tf.keras.layers.Conv1D(64, 15, padding='same', activation='relu')(inputs)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.MaxPooling1D(5)(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	
	# Residual block 1
	shortcut = x
	
	x = tf.keras.layers.Conv1D(64, 7, padding='same', activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Conv1D(64, 7, padding='same', activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Add()([x, shortcut])
	
	x = tf.keras.layers.MaxPooling1D(3)(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	
	# Residual block 2
	shortcut = x
	
	x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Add()([x, shortcut])
	
	x = tf.keras.layers.MaxPooling1D(3)(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	
	# Residual block 3
	shortcut = x
	
	x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Add()([x, shortcut])
	
	x = tf.keras.layers.MaxPooling1D(3)(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	
	# Residual block 4
	shortcut = x
	
	x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Add()([x, shortcut])
	
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(48, activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)

	outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

	model = tf.keras.Model(inputs=inputs, outputs=outputs)

	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model


def trainNN(attempt, initial_learning_rate, weight_decay):
	epochs = 164
	batch_size = 256

	trainData, trainRef, trainMap = readTrainData()
	validData, validRef = readValidData()

	custom_lr_schedule = CustomLearningRateSchedule(initial_learning_rate, len(trainData), batch_size, epochs)
	model = getModel(custom_lr_schedule, weight_decay)

	checkpoint_callback = ModelCheckpoint(filepath=f'../../models/IdentityResNet4x2/best_attempt{attempt}.h5', 
									   monitor='val_accuracy', mode='max', save_best_only=True,
									   save_weights_only=False, verbose=1)

	history = model.fit(trainData, trainRef, epochs=epochs, batch_size=batch_size,
					 validation_data=(validData, validRef), verbose=1, callbacks=[checkpoint_callback])

	train_accuracy = history.history['accuracy']
	val_accuracy = history.history['val_accuracy']
	with open(f'../../models/IdentityResNet4x2/history_attempt{attempt}.csv', 'w') as f:
		f.write('epoch,train_accuracy,val_accuracy\n')
		for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracy, val_accuracy), start=1):
			f.write(f'{epoch},{train_acc},{val_acc}\n')


def plotTrainValidCurve(pathCSV, pathPNG):
	data = pd.read_csv(pathCSV)

	epochs = data['epoch']
	train_accuracy = data['train_accuracy']
	val_accuracy = data['val_accuracy']

	plt.figure()
	plt.plot(epochs, train_accuracy, label='Training Accuracy')
	plt.plot(epochs, val_accuracy, label='Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')

	plt.savefig(pathPNG)


def trainLogReg(attempt):
	trainData, trainRef, trainMap = readTrainData()
	
	custom_objects = { "CustomAdamW": CustomAdamW, "CustomLearningRateSchedule": CustomLearningRateSchedule }
	NNmodel = keras.models.load_model(f'../../models/IdentityResNet4x2/best_attempt{attempt}.h5', custom_objects=custom_objects)
	
	pred = NNmodel.predict(trainData)

	refMapping = {}
	joined = {}

	for index, row in trainMap.iterrows():
		refMapping[int(row[0])] = int(trainRef[index].argmax())
		joined[int(row[0])] = []

	for i,preds in enumerate(pred):
		joined[int(trainMap[0][i])].append([math.log(x) for x in list(preds)])

	X = []
	y = []

	for key in joined.keys():
		X.append(np.exp(np.mean(joined[key], axis=0)))
		y.append(refMapping[key])

	X = np.array(X)
	y = np.array(y)

	logreg = linear_model.LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1e9, fit_intercept=False)
	logreg.fit(X, y)

	def loss_function(weights, X, y, logreg):
		logreg.coef_ = weights.reshape(logreg.coef_.shape)
		y_pred = logreg.predict(X)
		f1 = metrics.f1_score(y, y_pred, average="weighted")
		return 1 - f1
	
	initial_weights = logreg.coef_.flatten()
	result = minimize(loss_function, initial_weights, args=(X, y, logreg), method="L-BFGS-B")
	
	logreg.coef_ = result.x.reshape(logreg.coef_.shape)

	with open(f'../../models/IdentityResNet4x2/logreg_attempt{attempt}.pkl', 'wb') as file:
		pickle.dump(logreg, file)

def getF1Score(mat):
	f1_scores = []
	for i in range(3):
		true_positive = mat[i, i]
		false_positive = np.sum(mat[:, i]) - true_positive
		false_negative = np.sum(mat[i, :]) - true_positive
	
		precision = true_positive / (true_positive + false_positive)
		recall = true_positive / (true_positive + false_negative)
	
		f1_score = 2 * (precision * recall) / (precision + recall)
		f1_scores.append(f1_score)

	average_f1_score = np.mean(f1_scores)
	return average_f1_score

def testF1(attempt):
	custom_objects = { "CustomAdamW": CustomAdamW, "CustomLearningRateSchedule": CustomLearningRateSchedule }
	model = keras.models.load_model(f'../../models/IdentityResNet4x2/best_attempt{attempt}.h5', custom_objects=custom_objects)

	testData, testMap, testRef = readTestData()

	refMapping = {}
	joined = {}

	for index, row in testRef.iterrows():
		refMapping[int(row[0])] = int(row[1])
		joined[int(row[0])] = []

	pred = model.predict(testData)
	for i,preds in enumerate(pred):
		joined[int(testMap[0][i])].append([math.log(x) for x in list(preds)])
	
	mat = np.zeros((3,3))

	with open(f'../../models/IdentityResNet4x2/logreg_attempt{attempt}.pkl', 'rb') as file:
		logreg = pickle.load(file)

	for key in joined.keys():
		predicted = logreg.predict(np.exp(np.mean(joined[key], axis=0)).reshape(1, -1))[0]
		gt = refMapping[key]
		mat[gt,predicted] += 1

	average_f1_score = getF1Score(mat)
	
	print(mat)
	print(f"attempt {attempt} has f1 score {average_f1_score}")


def trainLogRegFull(listAttempts):
	trainData, trainRef, trainMap = readTrainData()
	trainMap = pd.read_csv('../../data/trainMap.csv', header=None).transpose()
	
	custom_objects = { "CustomAdamW": CustomAdamW, "CustomLearningRateSchedule": CustomLearningRateSchedule }
	NNmodels = [keras.models.load_model(f'../../models/IdentityResNet4x2/best_attempt{x}.h5', custom_objects=custom_objects) for x in listAttempts]

	predicts = [NNmodels[x - 1].predict(trainData) for x in listAttempts]

	refMapping = {}
	joined = {}

	for index, row in trainMap.iterrows():
		refMapping[int(row[0])] = int(trainRef[index].argmax())
		joined[int(row[0])] = []

	for i in range(trainData.shape[0]):
		pred = []
		for x in listAttempts:
			pred.extend(list(predicts[x - 1][i]))
		joined[int(trainMap[0][i])].append([math.log(x) for x in list(pred)])

	X = []
	y = []

	for key in joined.keys():
		X.append(np.exp(np.mean(joined[key], axis=0)))
		y.append(refMapping[key])

	X = np.array(X)
	y = np.array(y)

	logreg = linear_model.LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1e9, fit_intercept=False)
	logreg.fit(X, y)

	def loss_function(weights, X, y, logreg):
		logreg.coef_ = weights.reshape(logreg.coef_.shape)
		y_pred = logreg.predict(X)
		f1 = metrics.f1_score(y, y_pred, average="weighted")
		return 1 - f1
	
	initial_weights = logreg.coef_.flatten()
	result = minimize(loss_function, initial_weights, args=(X, y, logreg), method="L-BFGS-B")
	
	logreg.coef_ = result.x.reshape(logreg.coef_.shape)

	with open(f'../../models/IdentityResNet4x2/logreg_full.pkl', 'wb') as file:
		pickle.dump(logreg, file)

def testF1Full(listAttempts):
	testData, testMap, testRef = readTestData()

	custom_objects = { "CustomAdamW": CustomAdamW, "CustomLearningRateSchedule": CustomLearningRateSchedule }
	NNmodels = [keras.models.load_model(f'../../models/IdentityResNet4x2/best_attempt{x}.h5', custom_objects=custom_objects) for x in listAttempts]

	predicts = [NNmodels[x - 1].predict(testData) for x in listAttempts]

	refMapping = {}
	joined = {}

	for index, row in testRef.iterrows():
		refMapping[int(row[0])] = int(row[1])
		joined[int(row[0])] = []

	for i in range(testData.shape[0]):
		pred = []
		for x in listAttempts:
			pred.extend(list(predicts[x - 1][i]))
		joined[int(testMap[0][i])].append([math.log(x) for x in list(pred)])
	
	mat = np.zeros((3,3))

	with open(f'../../models/IdentityResNet4x2/logreg_full.pkl', 'rb') as file:
		logreg = pickle.load(file)

	for key in joined.keys():
		predicted = logreg.predict(np.exp(np.mean(joined[key], axis=0)).reshape(1, -1))[0]
		gt = refMapping[key]
		mat[gt,predicted] += 1

	average_f1_score = getF1Score(mat)
	
	print(mat)
	print(f"Full f1 score {average_f1_score}")


if __name__ == "__main__":
	testGPU()

	plot_lr_schedule("../../models/IdentityResNet4x2/schedule.png")
	
	for attempt in range(1, 4):
		trainNN(attempt, 1e-3, 1e-1)
		plotTrainValidCurve(f"../../models/IdentityResNet4x2/history_attempt{attempt}.csv",
					 f"../../models/IdentityResNet4x2/history_attempt{attempt}.png")
	
	for attempt in range(1, 4):
		trainLogReg(attempt)
	
	for attempt in range(1, 4):
		testF1(attempt)

	#trainLogRegFull(range(1, 4))
	#testF1Full(range(1, 4))
