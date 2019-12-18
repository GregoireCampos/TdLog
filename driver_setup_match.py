#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:34:50 2019

@author: badrbelkeziz
"""
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import json
import csv
import pickle

global driver

def launch_driver() : 
    #PROXY = "103.241.204.225:8080"
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument("start-maximized")
    chrome_options.add_experimental_option("detach", True)
    #chrome_options.add_argument('--proxy-server=http://%s' % PROXY)
    chrome_options.add_argument('--data-path=/tmp/data-path')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--homedir=/tmp')
    chrome_options.add_argument('--disk-cache-dir=/tmp/cache-dir')
    chrome_options.add_argument(
    'user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')
    driver = webdriver.Chrome('/Users/Codage/Desktop/ProjTDLOG/Bet/chromedriver', options=chrome_options)
    driver.get("https://www.matchendirect.fr");
    return(driver)
    