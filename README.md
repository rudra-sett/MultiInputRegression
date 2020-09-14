# MultiInputRegression
A complicated and poorly made program to input news and stock data to predict future stock prices

scraper.py scrapes links to news articles for a specified company on each day that stock data for that company exists. These parameters can be changed within the script. It writes these links out to a file

articledownlader.py downloads all of those articles as text files, sorted by date

network.py will use both the stock data and news on that company to make predictions of future stock prices. Currently, it doesn't do normalization of any kind on the data. (it's an easy fix that will come in the future)
