import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_addons as tfa


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


def readData():
	trainData = tf.convert_to_tensor(pd.read_csv('../../data/trainData.csv', header=None), dtype=tf.float32)
	trainRef = keras.utils.to_categorical(tf.convert_to_tensor(pd.read_csv('../../data/trainRef.csv', header=None).transpose(), dtype=tf.float32), 3)
	
	validData =  tf.convert_to_tensor(pd.read_csv('../../data/validData.csv', header=None), dtype=tf.float32)
	validRef =  keras.utils.to_categorical(tf.convert_to_tensor(pd.read_csv('../../data/validRef.csv', header=None).transpose(), dtype=tf.float32), 3)
	
	return trainData, trainRef, validData, validRef


def readTestData():

	testData =  tf.convert_to_tensor(pd.read_csv('../../data/testData.csv', header=None), dtype=tf.float32)
	testMap =  pd.read_csv('../../data/testMap.csv', header=None).transpose()
	testRef = pd.read_csv('../../data/testRef.csv', header=None)
	
	return testData, testMap, testRef


def getScedule(initial_learning_rate, datasetLen, minibatch, epochN, num_waves=8, warmup_epochs=5):
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
	total_steps = (68062 + 256 - 1) // 256 * 256
	lr_schedule = getScedule(1e-3, 68062, 256, 256)

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


def getModel(lr_schedule, decay):
	def weight_decay_schedule(step):
		return lr_schedule(step) * decay

	opt = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay_schedule)

	model = keras.Sequential()
	model.add(keras.layers.Conv1D(64, 192, kernel_initializer='normal', activation='relu', input_shape=(1500, 1)))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling1D(5))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Conv1D(48, 64, kernel_initializer='normal', activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling1D(5))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Conv1D(32, 18, kernel_initializer='normal', activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(64, kernel_initializer='normal', activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dense(3, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def train(attempt, initial_learning_rate, weight_decay):
	epochs = 256
	batch_size = 256

	trainData, trainRef, validData, validRef = readData()

	lr_schedule = getScedule(initial_learning_rate, len(trainData), batch_size, epochs)
	model = getModel(lr_schedule, weight_decay)

	checkpoint_callback = ModelCheckpoint(filepath=f'../../models/BasicCNN/best_attempt{attempt}.h5', 
									   monitor='val_accuracy', mode='max', save_best_only=True,
									   save_weights_only=True, verbose=1)

	history = model.fit(trainData, trainRef, epochs=epochs, batch_size=batch_size,
					 validation_data=(validData, validRef), verbose=1, callbacks=[checkpoint_callback])
	
	model.save(f'../../models/BasicCNN/final_attempt{attempt}.h5')

	train_accuracy = history.history['accuracy']
	val_accuracy = history.history['val_accuracy']
	with open(f'../../models/BasicCNN/history_attempt{attempt}.csv', 'w') as f:
		f.write('epoch,train_accuracy,val_accuracy\n')
		for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracy, val_accuracy), start=1):
			f.write(f'{epoch},{train_acc},{val_acc}\n')


if __name__ == "__main__":
	testGPU()
	
	# plot_lr_schedule("../../models/BasicCNN/schedule.png")
	
	train(1, 1e-3, 2.5e-4)
	train(2, 1e-3, 2.5e-4)
	train(3, 1e-3, 2.5e-4)
	
	# plotTrainValidCurve("../../models/basicCNN/dummy.csv", "../../models/basicCNN/dummyAccuracy.png")
	# todo get final accuracy on test set