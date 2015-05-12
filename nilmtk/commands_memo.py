# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:30:14 2015

@author: t7
"""

#Useful commands examples

#Use my location_inference
import numpy as np
import pandas as pd
from pandas import *
from nilmtk import HDFDataStore
from datetime import datetime
import sys
mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/nilmtk'
sys.path.append(mypath)
from location_inference import LocationInference
loc = LocationInference('REDD')

loc.dataset.set_window(start=None, end='2011-04-27 00:00:00')
loc.infer_locations()


#Disag step by step
print("Training...")
loc.dataset.set_window(start=None, end='2011-05-01 00:00:00')
from combinatorial_optimisation_location import CombinatorialOptimisation
co = CombinatorialOptimisation()
co.train(loc.elec)

print("Disagreggating...")
loc.dataset.set_window(start='2011-05-01 00:00:00', end=None)
loc.infer_locations()
mains = loc.elec.mains()


disagout = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd-disag-loc.h5'
output = HDFDataStore(disagout, 'w')
co.disaggregate(mains, output, location_data=loc)
output.close()


disagout = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd-disag-noloc.h5'
print("Metrics...")
disag = DataSet(disagout)
disag_elec = disag.buildings[1].elec
f1_noloc = f1_score(disag_elec, loc.elec)

disagout = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd-disag.h5'
print("Metrics...")
disago = DataSet(disagout)
disago_elec = disago.buildings[1].elec
f1_original = f1_score(disago_elec, loc.elec)




from sklearn.utils.extmath import cartesian
centroids = [model['states'] for model in co.model]
state_combinations = cartesian(centroids)

mains = loc.elec.mains()

vampire_power = mains.vampire_power()
print("vampire_power = {} watts".format(vampire_power))
n_rows = state_combinations.shape[0]
vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
state_combinations = np.hstack((state_combinations, vampire_power_array))
summed_power_of_each_combination = np.sum(state_combinations, axis=1)

load_kwargs={}
load_kwargs.setdefault('resample', True)
load_kwargs.setdefault('sample_period', 60)
load_kwargs['sections'] = mains.good_sections()

chunks = list(mains.power_series(**load_kwargs))

#Write API emoncsm = 8c8251d33ec13b5282a5047e65275294
#Read API emoncsm  = 5be909f21b261c793b7d4a0ad3fd07fb


#131.180.174.90
#energydata-es.ewi.tudelft.nlenergydata-es.ewi.tudelft.nl
#hdfhg
#energydata-es.ewi.tudelft.nl
#@Kshay2014

#t7@t7Envy-Ubuntu:~$ ssh asrirangamnara@linux-bastion.tudelft.nl
#asrirangamnara@linux-bastion.tudelft.nl's password: 
#Last login: Thu Mar 19 15:33:16 2015 from x-145-94-62-68.wired.tudelft.nl
#-bash-4.1$ ssh asrirangamnara@energydata-es.ewi.tudelft.nl
#asrirangamnara@energydata-es.ewi.tudelft.nl's password: 
#Permission denied, please try again.
#asrirangamnara@energydata-es.ewi.tudelft.nl's password: 
#Welcome to Ubuntu 14.04.2 LTS (GNU/Linux 3.13.0-46-generic x86_64)
#
# * Documentation:  https://help.ubuntu.com/
#
#24 packages can be updated.
#15 updates are security updates.
#
#Last login: Thu Mar 19 15:33:42 2015 from srv227.tudelft.net
#asrirangamnara@energydata-es:~$ 

#loc.fridge_nnv['2011-04-18 11:44:15':'2011-04-18 11:46:10'].plot()
#list(loc.elec[1].power_series())[0]['2011-04-18 11:44:15':'2011-04-18 11:46:10'].plot()

co.model[0]['states'] = np.array([0, 49, 198])
co.model[1]['states'] = np.array([0, 1075])
co.model[2]['states'] = np.array([14,21])
co.model[3]['states'] = np.array([22,49,78])
co.model[4]['states'] = np.array([0,41,82])
co.model[5]['states'] = np.array([0,1520])
co.model[6]['states'] = np.array([0,1620])
co.model[7]['states'] = np.array([0,15])
co.model[8]['states'] = np.array([0,1450])
co.model[9]['states'] = np.array([0,1050])
co.model[10]['states'] = np.array([0,1450])
co.model[11]['states'] = np.array([0,65])
co.model[12]['states'] = np.array([0,45,60])
co.model[14]['states'] = np.array([0,4200])
co.model[15]['states'] = np.array([0,3750])


#plot dataset powerseries of an appliance
list(loc.elec[6].power_series())[0].plot()
#plot my inferred events
loc.events[6].plot()

#plot activity histogram of an appliance: minutes in day
loc.elec[6].plot_activity_histogram(bin_duration='T', on_power_threshold=200)
#plot activity histogram of an appliance: days in week
loc.elec[6].plot_activity_histogram(bin_duration='D', period='W', on_power_threshold=200)

outputhdf5 = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd.h5'
redd = DataSet(outputhdf5)
elec = redd.buildings[1].elec
co2 = CombinatorialOptimisation()
co3 = CombinatorialOptimisation()
co4 = CombinatorialOptimisation()

co2.train(elec, max_num_clusters = 2, resample_seconds=60)
co3.train(elec, max_num_clusters = 3, resample_seconds=60)
co4.train(elec, max_num_clusters = 4, resample_seconds=60)
centroids2 = [model['states'] for model in co2.model]
centroids3 = [model['states'] for model in co3.model]
centroids4 = [model['states'] for model in co4.model]

ii = [5,6,7,8,9,11,12,13,14,15,16,17,18,19,(3,4),(10,20)]
cc = DataFrame({'2': centroids2, '3': centroids3, '4': centroids4}, index=ii)





#Print summed power of each combination and the devices that are responsible for that combination and their location
#for i in range(0,100):
#    b = []
#    [b.extend(loc.appliances_location[t]) for t in [appliances_order[v] for v,j in enumerate(state_combinations[index_sorted[i]]) if ((j != 0) and (j != vampire_power))]]
#    bb = list(set(b))
#    print (known_array_sorted[i] - vampire_power), [appliances_order[v] for v,j in enumerate(state_combinations[index_sorted[i]]) if ((j != 0) and (j != vampire_power))], '\t\t', bb