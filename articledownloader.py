#This script downloads articles from the links produced by scraper.py
import os
import sys
import newspaper as news

#open files, strip \n from end
f = open('datedlinks.txt','r')
lines = f.readlines()
newslinks = []
for line in lines:
    newslinks.append(line[:-1])
#f.close()

def getdatestring(day):
    year, month, day = str(day)[:10].split("-")
    #day = day.split(" ")[0]
    return year, month, day

#links to articles
summaries = []
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = news.Config()
config.browser_user_agent = user_agent
x = 1
for link in newslinks:
    date, link = link.split(",",1)
    print("article number "+str(x))
    a = news.Article(link,config=config)
    data = ""
    try:
        a.download()
        a.parse()
        a.nlp()
        #a.summary()
        data = a.text
    except news.article.ArticleException:
        print("something bad happened on article "+str(x)+", for "+date)
    if not os.path.exists(date):
        os.mkdir(date)
    pathtowrite = "Articles/"+date+"/"+str(x)+".txt"
    with open(pathtowrite,"w") as out:
        out.write(data)
        #print(str(a.publish_date)[:10])
    x+=1






