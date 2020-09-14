import matplotlib as plt
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
from nltk.stem import PorterStemmer


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
VARIABLES
'''
#define the ticker symbol
tickerSymbol = 'MSFT'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this tickern
tickerDf = tickerData.history(period='1d', start='2017-1-01', end='2017-12-14').reset_index()

#loads downloaded articles into a list
dates = sorted(os.listdir("Articles"))
loadedarticles = []
for date in dates:
    articles = sorted(os.listdir("Articles/"+date))
    for article in articles:
        with open("Articles/"+date+"/"+article) as f:
            loadedarticles.append(f.read())

#gets the list of dates for each article
f = open('datedlinks.txt','r')
lines = f.readlines()
newslinks = []
for line in lines:
    newslinks.append(line[:-1])
dates=[]
for link in newslinks:
    date, link = link.split(",",1)
    dates.append(date)
f.close()

#function clean articles (punctuation)
def nopunc(text):
    cleaned = "".join(c for c in text if c not in string.punctuation)
    return cleaned.lower().replace("\n", " ")

#cleans the loaded articles
cleaned = []
def clean():
    for index, article in enumerate(loadedarticles):
        #loadedarticles[index] = nopunc(article)
        cleaned.append(nopunc(article))
clean()
#stem the articles
ps = PorterStemmer()
final = [[ps.stem(token) for token in sentence.split(" ")] for sentence in cleaned]
for i, text in enumerate(final):
    final[i] = " ".join(text)

df = pd.DataFrame(final)
df.index = dates
#combine articles from the same day
grouped = df.groupby(['Date'])['Article'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
dateframe = grouped.pop('Date')
dateframeS = tickerDf.pop('Date')
Y = tickerDf.pop('Close')

#merge with numerical data
#merged = pd.merge(grouped,tickerDf,on="Date")

#merge with just Y, this is to drop extra news articles at the end
mergedText = pd.merge(grouped,Y, left_index=True,right_index=True)
mergedText.pop('Close')
mergedStats = pd.merge(tickerDf,Y, left_index=True,right_index=True)



#tokenize articles
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(mergedText['Article'])
X = tokenizer.texts_to_sequences(mergedText['Article'])
X = pad_sequences(X, padding = "post",truncating="post")

#the timeseries-modified data that gets passed to the model
tstext = series_to_supervised(X,n_in=4,n_out=1)
tsstats = series_to_supervised(mergedStats,n_in=4,n_out=1)
tsstats = tsstats.drop(tsstats.columns[[-2,-3,-4,-5,-6,-7]],1)

#remove last column, because I'm not predicting today's news (or keep it, becaue you need today's news for today's stock price)
#tstext.drop(tstext.columns[[-1]],1)

seq_length = tstext.shape[1]
embed_dim = 128

#process news
Input1 = tf.keras.layers.Input(shape = (seq_length,))
Embedding1 = tf.keras.layers.Embedding(max_features, embed_dim,input_length = tstext.shape[1])(Input1)
LSTM1 = tf.keras.layers.LSTM(128)(Embedding1)

#process numerical data
Input2 = tf.keras.layers.Input(shape = (28,))

#combine data types
Concatenation1 = concatenate([nlp_out, meta_input]) 
