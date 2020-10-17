# MultiInputRegression
A program to input news and stock data to predict future stock prices

scraper.py scrapes links to news articles for a specified company on each day that stock data for that company exists. These parameters can be changed within the script. It writes these links out to a file

articledownlader.py downloads all of those articles as text files, sorted by date

network.py will use both the stock data and news on that company to make predictions of future stock prices. 

The articles are fed through an LSTM, as if they were being classified (but they are not) and the internal representation of the text data is combined with the numerical data on the stocks.
Regressionis applied to this combination of data to hopefully match the actual stock data.

newnetwork.py uses a pretrained sentiment model (not in this github) to reduce all of the articles to sentiments, and this is combined with stock data. This would probably work better than using the internal representation. 
