import matplotlib.pyplot as plt
import os
import sys
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import yfinance as yf
import time
import numpy as np
import nltk
import string
import json
from nltk.stem import PorterStemmer
import scraper
import articledownloader

'''
VARIABLES
'''
#define the ticker symbol
tickerSymbol = 'MSFT'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this tickern
tickerDf = tickerData.history(period='1d', start='2017-1-01', end='2017-12-15').reset_index()
#model was trained on 2017-01-01 to 2017-12-14, on MSFT
#tickerDf = pd.read_csv('stockdata.csv')
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

'''
LOAD DATA
'''
def loaddata(directory,links):
    dates = sorted(os.listdir(directory))
    loadedarticles = []
    for date in dates:
        try:
            articles = sorted(os.listdir(directory+"/"+date))
        except NotADirectoryError:
             print("not a directory!")
             pass
        try:
            for article in articles:
                with open(directory+"/"+date+"/"+article) as f:
                    loadedarticles.append(f.read())
        except NotADirectoryError:
             print("not a directory!")
             pass
    f = open(links,'r')
    lines = f.readlines()
    newslinks = []
    for line in lines:
        newslinks.append(line[:-1])
    dates=[]
    for link in newslinks:
        date, link = link.split(",",1)
        dates.append(date)
    f.close()
    return loadedarticles,dates

loadedarticles,dates = loaddata("Articles","betterdatedlinks.txt")

#opens tokenizer from premade tokenizer
seq_length = 100
max_features = 100000
embed_dim = 100
tokenizer =  tf.keras.preprocessing.text.Tokenizer(num_words=max_features, split=' ')
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

def getsentimentmodel():
    inputs1 = tf.keras.layers.Input(shape=(seq_length,))
    embedding1 = tf.keras.layers.Embedding(max_features, embed_dim,input_length=seq_length)(inputs1)
    conv1 = tf.keras.layers.Conv1D(filters=100, kernel_size=2,padding="same")(embedding1)
    maxpool1 = tf.keras.layers.GlobalMaxPool1D()(conv1)
    flat1 = tf.keras.layers.Flatten()(maxpool1)
    conv2 = tf.keras.layers.Conv1D(filters=100, kernel_size=4,padding="same")(embedding1)
    maxpool2 = tf.keras.layers.GlobalMaxPool1D()(conv2)
    flat2 = tf.keras.layers.Flatten()(maxpool2)
    conv3 = tf.keras.layers.Conv1D(filters=50, kernel_size=6,padding="same")(embedding1)
    maxpool3 = tf.keras.layers.GlobalMaxPool1D()(conv3)
    flat3 = tf.keras.layers.Flatten()(maxpool3)
    concat = tf.keras.layers.Concatenate()([flat1,flat2,flat3])
    dropout1 = tf.keras.layers.Dropout(0.2)(concat)
    x = tf.keras.layers.Dense(128)(dropout1)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs1, outputs=outputs)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.summary()
    model.load_weights("80percentacc.h5")
    return model
model = getsentimentmodel()

def cleanstring(text):
    #very simple text cleaning; simply lowercases and removes punctuation
    cleaned = "".join(c for c in text if c not in string.punctuation)
    return cleaned.lower().replace("\n", " ")

def evalsentiment(text):
    text = cleanstring(text)
    text = [text]
    text = tokenizer.texts_to_sequences(text)
    test = np.array(text)
    fulllength = test.shape[1]
    truncatedlength = fulllength - fulllength%100
    test = test[0,0:truncatedlength]
    test = test.reshape(-1,100)
    negprob = 0.5
    posprob = 0.5
    try:
        sentiment = model.predict(test)
        negprob = np.mean(sentiment[:,0])
        posprob = np.mean(sentiment[:,1])
    except ValueError:
        print("Something went wrong")
        pass
    if negprob > posprob:
        return (-1 * negprob)+0.5
    return posprob-0.5

sentiments = []
def populatesentiments(texts):
    array = []
    for index,article in enumerate(texts):
        array.append(evalsentiment(article))
    return array
sentiments = populatesentiments(loadedarticles)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return norm, v / norm

'''
PREPROCESSING
'''

def createdata(sentiments,dates,tickerDf):
    sdf = pd.DataFrame(np.array(sentiments))
    ddf = pd.DataFrame(np.array(dates))
    fdf = pd.concat((ddf,sdf),axis=1)
    fdf.index = dates
    fdf.columns = ["Date","Sentiment"]
    #average sentiments from the same day
    grouped = fdf.groupby(["Date"]).mean()
    grouped = grouped.reset_index()
    dateframe = grouped.pop('Date')
    Y = tickerDf.pop('Close')
    totalinfo = pd.concat((tickerDf,grouped),axis = 1)
    #totalinfo.pop("Volume","Dividends","Stock Splits") all of the variables (X)
    totalinfo = totalinfo.drop(totalinfo.columns[[4,5,6]],1)
    popdates = totalinfo.pop("Date")
    #full data, X and Y
    complete = pd.concat((totalinfo,Y),axis = 1)
    #final data for LSTM:
    timesteps = 4
    final = series_to_supervised(complete,timesteps,1)
    final.drop(final.columns[[-2,-3,-4,-5]], axis=1, inplace=True)
    norm, final = normalize(final.values)
    X = final[:,:-1]
    Y = final[:,-1]
    #get X in the form of a sample, 4 timesteps per sample, and the data for each step
    reshapedX = X.reshape(X.shape[0],timesteps,int(X.shape[1]/timesteps))
    return reshapedX,Y,norm


'''
MODEL
'''

Input1 = tf.keras.layers.Input(shape = (reshapedX.shape[1], reshapedX.shape[2]))
LSTM1 = tf.keras.layers.LSTM(1000, input_shape=(reshapedX.shape[1], reshapedX.shape[2]))(Input1)
Output = tf.keras.layers.Dense(1)(LSTM1)
LSTMmodel = tf.keras.Model(inputs=Input1, outputs=Output)
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
LSTMmodel.compile(loss="mean_absolute_percentage_error", optimizer=opt,metrics=['linear_regression_equality'])
LSTMmodel.summary()

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True, period = 10)

history = LSTMmodel.fit(x=reshapedX,y = Y, epochs=100, callbacks=[checkpoint_callback])

import keras.backend as K
accepted_diff = 0.05
def linear_regression_equality(y_true, y_pred):
    y_true = y_true*norm
    y_pred = y_pred*norm
    diff = K.abs(y_true-y_pred)
    return K.mean(K.cast(diff < accepted_diff, tf.float32))


model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


testarticles,testdates = loaddata("ArticlesTest","ArticlesTest/datedlinkstestTESLA.txt")
testsentiments = populatesentiments(testarticles)
testtickerSymbol = 'TSLA'
testtickerData = yf.Ticker(tickerSymbol)
testtickerDf = testtickerData.history(period='1d', start='2020-08-31', end='2020-09-05').reset_index()
X,Y,norm = createdata(testsentiments,testdates,testtickerDf)
