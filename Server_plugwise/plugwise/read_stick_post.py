# -*- coding: utf-8 -*-
"""
Requires this library: https://bitbucket.org/hadara/python-plugwise

This script reads the values from the Plugwise Stick and posts them to the
web server responsible to store them in the db.
"""
from plugwise import Circle, Stick
import time
import requests
import pprint as pp
from time import sleep


mac_circles = ['000D6F0004543055',
               '000D6F0004556FB7',
               '000D6F000454B84D',
               '000D6F000454386D',
               '000D6F000454BD9E'
               ]
               
port_stick  = '/dev/tty.usbserial-A103J264'
#/dev/ttyUSB1'

timeout = 5 #seconds


#Post to local database
urlphp = 'http://akshaysn.ddns.net/plug.php'

#Post to visualize with Emoncms
api_local  = 'bb30a1087d4bcbb1df5c80341a29e606'
emoncms_local_url  = 'http://akshaysn.ddns.net/emoncms/input/post.json?'

def postToLocalDB(readings, timestamp):    
    print 'Posting to local db...'
    for reading in readings.items():
        node = reading[0]
        power = str(reading[1])
        payload = {'name': node, 'timestamp': timestamp, 'power': power}
        r1 = requests.post(urlphp, data=payload)
        print node, r1 
    
def postToEmoncmsBulk(readings, epoch):    
    print 'Posting to local emoncms bulk...'  
    bulk_data = '[' + ''.join(['[0,' + reading[0] + ',' + str(reading[1]) + '],' for reading in readings.items()])[:-1] + ']'
    payload2 = 'time=' + str(epoch) + "&data=" + bulk_data + "&apikey=" + api_local
    r2 = requests.post(emoncms_local_url + payload2)
    print r2.url

def main():
    print ('Starting logging of Plugwise Stick...')  
    s = Stick(port = port_stick)

    circles = []
    [circles.append(Circle(mac,s)) for mac in mac_circles]
    
    x = 0
    while True:
        x += 1
        readings = {}
        for circle in circles:
            try:
                power = circle.get_power_usage()                
            except Exception:
                power = -1.0
            readings[circle.mac] = power
        
        epoch = time.time()
        datenow = time.strftime('%Y-%m-%dT%H:%M:%S+01:00')
        print x, epoch, datenow, '========================================================='
        pp.pprint(readings)
        
        #postToEmoncmsBulk(readings, epoch)  
        #postToLocalDB(readings, timestamp)
        
        sleep(timeout)

if __name__ == '__main__':
    main()



