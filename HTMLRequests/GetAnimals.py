#!/usr/bin/env python
# -*- coding: utf8 -*-

import string
from lxml import html
import requests
import urllib.request
from joblib import Parallel, delayed
import multiprocessing

alphabet=list(string.ascii_uppercase)

def task(l):
    print(l)
    url="https://a-z-animals.com/animals/pictures/" + l + "/"
    page=requests.get(url)
    tree=html.fromstring(page.content)
    hrefs=tree.xpath('//div[@class="picture-content"]//a/@href')
    for href in hrefs:
        url="https://a-z-animals.com" + href
        page=requests.get(url)
        tree=html.fromstring(page.content)
        name=tree.xpath('//*[@id="container"]/main/article/figure/div/a[1]/img/@src')
        urlSrc = "https://a-z-animals.com" + name[0]
        nm=urlSrc[urlSrc.rfind("/")+1:]
        urllib.request.urlretrieve(urlSrc,"Pictures/"+nm)
        
if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(task)(l) for l in alphabet)



