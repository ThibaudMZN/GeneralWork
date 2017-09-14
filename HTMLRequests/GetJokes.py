#!/usr/bin/env python
# -*- coding: utf8 -*-

import string
from lxml import html
from lxml.etree import tostring
import requests
import urllib.request
import json


jokes = {}
i = 0
for pNum in range(100):
    print(pNum/50*100,"%")
    url="https://www.blague-drole.net/blagues/humour.noir-" + str(pNum) + ".html?tri=top"
    page=requests.get(url)
    tree=html.fromstring(page.content)
    divs=tree.xpath('//*[@class="panel panel-info blague"]/div[2]/div/div')
    for d in divs:
        p=tostring(d,encoding="utf-8").decode("utf-8").replace("<p>","").replace("</p>","")
        p=p.replace('<div class="text-justify texte">',"").replace('</div>',"")
        p=p.replace('<blockquote>','').replace('</blockquote>','')
        jokes[str(i)] = p
        i += 1


with open("jokes.json","w") as fp:
    json.dump(jokes,fp)
