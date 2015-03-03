# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:43:17 2015

@author: t7
"""

from nilmtk.dataset_converters import download_dataport
from nilmtk.dataset_converters import convert_redd
from nilmtk import DataSet
import nilmtk
from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk.metrics import f1_score
from nilmtk import HDFDataStore


source = '/home/t7/Dropbox/Documents/TUDelft/Thesis/dev/Datasets/REDD/low_freq'
outputhdf5 = '/home/t7/Dropbox/Documents/TUDelft/Thesis/dev/Datasets/REDD/redd.h5'


print("Converting...")
#convert_redd(source, outputhdf5)
redd = DataSet(outputhdf5)

print("Training...")
redd.set_window(start=None, end='2011-05-01 00:00:00')

elec = redd.buildings[1].elec
co = CombinatorialOptimisation()
co.train(elec)

print("Disagreggating...")
redd.set_window(start='2011-05-01 00:00:00', end=None)
mains = elec.mains()
output = HDFDataStore('output.h5', 'w')
co.disaggregate(mains, output)
output.close()

print("Metrics...")
disag = DataSet('output.h5')
disag_elec = disag.buildings[1].elec
#f1 = f1_score(disag_elec, elec)


#download_dataport('Aeo7SFRDqkRv', 
#                  'BIBoegKX88H6', 
#                  'wikienergy.h5',
#                  periods_to_load = {434: ('2014-04-01', '2014-05-01')})