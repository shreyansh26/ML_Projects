from __future__ import division, absolute_import

import tflearn
from tflearn.data_utils import VocabularyProcessor, to_categorical, pad_sequences

from sklearn.cross_validation import train_test_split
import pandas as pd 
import numpy as np

load_model = 0
save_model = 1

# Load csv (Only columns required)
data = pd.read_csv('ign.csv').ix[:, 1:3]
data.fillna(value='', inplace=True)

#print(data.score_phrase.value_counts())

value_x = data.title
value_y = data.score_phrase

# Convert the strings in the input into integers corresponding to the dictionary positions
# Data is automatically padded so we need to pad_sequences manually
vocab_proc = VocabularyProcessor(15)
value_x = np.array(list(vocab_proc.fit_transform(value_x)))

# 11 classes for predictions
vocab_proc2 = VocabularyProcessor(1)
value_y = np.array(list(vocab_proc2.fit_transform(value_y))) - 1 # Since 0-10
value_y = to_categorical(value_y, nb_classes=11)

# Split training data
trainX, testX, trainY, testY = train_test_split(value_x, value_y, test_size=0.1)

# Build network
# Each input has length 15
net = tflearn.input_data([None, 15])
net = tflearn.embedding(net, input_dim=10000, output_dim=256)
# Since using recurrent neural network
net = tflearn.gru(net, 256, dropout=0.9, return_seq=True)
net = tflearn.gru(net, 256, dropout=0.9)

net = tflearn.fully_connected(net, 11, activation='softmax')

net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

# Train the network
model = tflearn.DNN(net, tensorboard_verbose=0)

if load_model == 1:
	model.load('gamemodel.tfl')

model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32, n_epoch=20)

if save_model == 1:
	model.save('gamemodel.tfl')
	print("Saved model!")