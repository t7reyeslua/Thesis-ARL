#!/usr/bin/python

import time
from time import sleep

timeout = 5
filepath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/Server_plugwise/plugwise/dummy.txt'
x = 0

while True:
    x += 1
    datenow = time.strftime('%Y-%m-%dT%H:%M:%S+01:00')
    print datenow
    
    with open("test.txt", "a") as myfile:
        myfile.write(str(datenow) + ' ' + str(x) +'\n')
    
    sleep(timeout)
    