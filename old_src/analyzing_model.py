from tensorflow.keras.layers import BatchNormalization
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

# NAME = f'wri-cnn-64x4-{int(time.time())}'
# tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.999)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def pickle_load(x_name, y_name):    
    pickle_in = open(f'{x_name}.pickle', 'rb')
    X = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(f'{y_name}.pickle', 'rb')
    y = pickle.load(pickle_in)
    pickle_in.close()
    return X, y

X_train, y_train = pickle_load('X_train', 'y_train')
X_val, y_val = pickle_load('X_val', 'y_val')

dense_layers = [0]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

models = []

for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:
			NAME = f'wri-{conv_layer}-conv-{layer_size}-layer-{dense_layer}-dense-{int(time.time())}'
			tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

			model = Sequential()
			model.add(Conv2D(layer_size, (3, 3), input_shape=X_train.shape[1:], activation='relu'))
			model.add(BatchNormalization())
			model.add(MaxPooling2D(pool_size=(2, 2)))

			for i in range(conv_layer - 1):
				model.add(Conv2D(layer_size, (3, 3), activation='relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))

			model.add(Flatten())
			for l in range(dense_layer):
				model.add(Dense(512, activation='relu'))
				# model.add(Dropout(0.5))

			model.add(Dense(1, activation='sigmoid'))
			model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
			model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=132, epochs=15, callbacks=[tensorboard])
			val_loss, val_acc = model.evaluate(X_val, y_val)
			models.append(f'{NAME}, LOSS: {val_loss}, ACC: {val_acc}')