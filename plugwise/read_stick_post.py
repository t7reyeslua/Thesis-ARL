#!/usr/bin/python
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
               '000D6F000454BD9E',
               '000D6F0004542EE3'
               ]
               
port_stick  = '/dev/ttyUSB0'

timeout = 5 #seconds


#Post to local database
urlphp = 'http://energydata-es.ewi.tudelft.nl/insert2db.php'

#Post to visualize with Emoncms
api_local  = '8c8251d33ec13b5282a5047e65275294'
emoncms_local_url  = 'http://energydata-es.ewi.tudelft.nl/emoncms/input/post.json?'


def postToLocalDB(readings, timestamp):    
    print 'Posting to local db...'
    for reading in readings.items():
        node = reading[0]
        power = str(reading[1])
        payload = {'name': node, 'timestamp': timestamp, 'power': power}
        r1 = requests.post(urlphp, data=payload)
        print node, r1 

def postToEmoncms(readings, epoch):    
    print 'Posting to local emoncms...'  
    for mac in readings.keys():                  
        payload2 = 'time=' + str(epoch) + "&node=" + mac + "&csv=" + str(readings[mac]) + "&apikey=" + api_local
        r2 = requests.post(emoncms_local_url + payload2)
        print r2.url
    
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
        
        postToEmoncms(readings, epoch)  
        postToLocalDB(readings, datenow)
        
        sleep(timeout)

if __name__ == '__main__':
    main()



