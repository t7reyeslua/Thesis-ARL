import datetime
import time
import requests
import xml.etree.ElementTree as ET
from time import sleep

#Credentials for accessing stretch
username = 'stretch'
password = 'ppwkcdzr'
url = 'http://reyeslua.ddns.net/minirest/appliances'

#Post to local database
#urlphp = 'http://reyeslua.ddns.net:8080/plug.php'
urlphp = 'http://localhost:8080/plug.php'

#Post to visualize with Emoncms
api_remote = '7723b5b2f433653c5e74ffd55c8b71ac'
api_local  = 'bb30a1087d4bcbb1df5c80341a29e606'
emoncms_remote_url = 'http://emoncms.org/input/post.json?'
#emoncms_local_url  = 'http://reyeslua.ddns.net:8080/emoncms/input/post.json?'
emoncms_local_url  = 'http://localhost:8080/emoncms/input/post.json?'


appliances_list_last = []    
#for x in range(0,15):  
x = 0
while True:
    try:
        i = datetime.datetime.now()
        r = requests.get(url, auth=(username, password), stream=True)
        i2 = datetime.datetime.now()
        
        root = ET.fromstring(r.content)
        appliances = list(root)
        appliances_list = [] 
            
        for appliance in appliances:
            appliance_info = {}
            appliance_info['name'] = appliance[0].text
            appliance_info['last_seen_date'] = appliance[4].text
            appliance_info['last_known_measurement'] = appliance[8].text
            appliance_info['power_state'] = appliance[5].text
            appliance_info['current_power_usage'] = appliance[6].text
            appliances_list.append(appliance_info)
        
        print x, i.isoformat(), str((i2-i))
        new_values = False
        if appliances_list_last:
            if appliances_list_last[0]['last_known_measurement'] != appliances_list[0]['last_known_measurement'] :
                new_values = True
        else:
            new_values = True
            
        if new_values:
            x = x + 1
            print appliances_list
            for appliance_info in appliances_list:    
                epoch = int(time.mktime(time.strptime(appliance_info['last_known_measurement'], '%Y-%m-%dT%H:%M:%S+01:00')))
                print 'Posting to local db...'
                payload = {'name': appliance_info['name'], 'timestamp': appliance_info['last_known_measurement'], 'power': appliance_info['current_power_usage']}
                r1 = requests.post(urlphp, data=payload)
                print r1            
                print 'Posting to local emoncms...'
                payload2 = 'time=' + str(epoch) + "&node=" + appliance_info['name'] + "&csv=" + appliance_info['current_power_usage'] + "&apikey=" + api_local
                r2 = requests.post(emoncms_local_url + payload2)
                print r2.url
    #            print 'Posting to remote emoncms...'
    #            payload3 = 'time=' + str(epoch) + "&node=" + appliance_info['name'] + "&csv=" + appliance_info['current_power_usage'] + "&apikey=" + api_remote
    #            r3 = requests.post(emoncms_remote_url + payload3)
    #            print r3.url
           
        appliances_list_last = appliances_list
        sleep(5)
    except Exception as e:
        print "Parse error"
