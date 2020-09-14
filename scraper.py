#scraping things
import requests
from bs4 import BeautifulSoup as BS
import newspaper as news
import datetime
import os
import sys
import numpy as np
import yfinance as yf
import time

'''
VARIABLES
'''
#define the ticker symbol
tickerSymbol = 'MSFT'
#define the company to search for
company = "microsoft"
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this tickern
tickerDf = tickerData.history(period='1d', start='2017-1-01', end='2017-12-25')
#get the dates of each trading day
dates = tickerDf.index.values
#header for Google search
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

#datetime to month, day, year
def getdatestring(day):
    return str(day).split("-")

#get multiple pages of news from google from a single 48 hour period
newslinks = []
def getnews(endday):
    print("getting news from "+str(endday)[:10]+" and earlier")
    #get a range of days from start to a week
    #endday = datetime.date(2018,4,3)
    date = str(endday)[:10]
    delta = datetime.timedelta(days=0)
    startday = endday-delta
    endyear, endmonth, endday = getdatestring(endday)
    endday = endday.split(" ")[0]
    startyear, startmonth, startday = getdatestring(startday)
    startday = startday.split(" ")[0]
    page = 0
    exists = True
    print("day is "+endday)
    while exists == True and page < 5:
        print("making google search for page... "+str(page))
        site = requests.get("https://www.google.com/search?q="+company+"&safe=active&biw=1517&bih=643&source=lnt&tbs=cdr%3A1%2Ccd_min%3A"+startmonth+"%2F"+startday+"%2F"+startyear+"%2Ccd_max%3A"+endmonth+"%2F"+endday+"%2F"+endyear+"&tbm=nws",headers=headers)
        #extract links from the search
        soup = BS(site.content, 'html.parser')
        infocards = soup.find_all('g-card')
        if (len(infocards) == 0):
            exists = False
        for card in infocards:
            link = card.find('a').get('href')
            if (date+","+link) not in newslinks:
                newslinks.append(date+","+link)
        page += 1
        time.sleep(0.5)

#get news for each trading day
for date in dates:
    date = datetime.datetime.utcfromtimestamp(date.astype(datetime.datetime)/1e9)
    getnews(date)
#write the links to a file
with open('datedlinks.txt', 'w') as f:
    for link in newslinks:
            f.write('%s\n' % link)



#site = requests.get("https://www.google.com/search?q=microsoft&safe=active&tbs=cdr: 1,cd_min:"+startmonth+"/"+startday+"/"+startyear+",cd_max:"+endmonth+"/"+endday+"/"+endyear+"&tbm=nws&ei=LEBZX-HJGLaEytMPuvS1-Ao&start="+str(page*10)+"&sa=N&ved=0ahUKEwjhj9bb-dzrAhU2gnIEHTp6Da84ChDy0wMIgAE&biw=1517&bih=643&dpr=0.9", headers=headers)
