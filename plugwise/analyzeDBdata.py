# -*- coding: utf-8 -*-
import pandas as pd
from pandas import Series, Timedelta

log_path = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/TUD/house_akshay/'
log_file = 'log_2015-04-06_2015-04-12.csv'
log = log_path + log_file

tstart  = '2015-04-06T00:00:00'
tfinish = '2015-04-12T23:59:59'


name = { 'microwave'        :'4542F6A', #Microwave
         'washing_machine'  :'36BC6A4', #Washing machine
         'various'          :'4554FB3', #Various
         'fridge'           :'45569E8', #Fridge
         'unknown'          :'4542EE3'} #Unknown

plugs = ['4542F6A', #Microwave
         '36BC6A4', #Washing machine
         '4554FB3', #Various
         '45569E8', #Fridge
         '4542EE3'] #Unknown

log_values = {}
for plug in plugs:
    log_values[plug] = {}

#Read CSV file
with open(log) as fp:
    for line in fp:
        line = line.replace('"','')
        line = line.replace('\n','')        
        fields = line.split(',')
        
        plug_id = fields[1]
        tstamp  = fields[2]
        power   = fields[3]
        
        #Save into corresponding dict
        if (tstart <= tstamp <=tfinish) and (plug_id in plugs):
            log_values[plug_id][pd.to_datetime(tstamp)] = float(power)

log_series = {}       
log_series_time_deltas = {} 
for plug in log_values.keys():
    #Turn dict into series
    log_series[plug] = Series(log_values[plug])
    log_series_time_deltas[plug] = {}
    timeIndex = log_series[plug].index
    #Count how many times each time delta ocurrs
    for i,ts in enumerate(timeIndex):
        if i > 0:
            delta = timeIndex[i] - timeIndex[i-1]
            n = log_series_time_deltas[plug].get(delta,0)
            [plug][delta] = n + 1


#Calculate how many deltas are larger than a reference (1min)  
d1m = Timedelta('0 days 00:01:00')
large_deltas = {}
for plug in log_series_time_deltas.keys():
    n = 0
    tot = 0
    for delta in log_series_time_deltas[plug].keys():
        tot += log_series_time_deltas[plug][delta]
        if delta > d1m:
            n += log_series_time_deltas[plug][delta]
    large_deltas[plug] = (n,tot)