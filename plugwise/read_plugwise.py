import datetime
import requests
import xml.etree.ElementTree as ET
from time import sleep

username = 'stretch'
password = 'ppwkcdzr'
url = 'http://reyeslua.ddns.net/minirest/appliances'
urlphp = 'http://reyeslua.ddns.net:8080/plug.php'

appliances_list_last = []    
#for x in range(0,30):  
x = 0
while True:
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
    
    x = x+1
    print x, i.isoformat(), str((i2-i))
    new_values = False
    if appliances_list_last:
        if appliances_list_last[0]['last_known_measurement'] != appliances_list[0]['last_known_measurement'] :
            new_values = True
    else:
        new_values = True
        
    if new_values:
        print appliances_list
       
    appliances_list_last = appliances_list
    sleep(5)
            
