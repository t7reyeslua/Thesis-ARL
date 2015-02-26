# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 17:46:09 2015

@author: t7
#CREATE TABLE tbl_appliances_power (appliance_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, name TEXT, measurement_timestamp DATETIME, current_power DECIMAL(10,3));
"""

from nilmtk.dataset_converters import convert_redd
from nilmtk import DataSet
from nilmtk.utils import print_dict
from nilmtk.elecmeter import ElecMeterID
import nilmtk
from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk.preprocessing import Apply
from nilmtk.stats import DropoutRate
from nilmtk import TimeFrame
import pandas as pd
import nilmtk.timeframe
from nilmtk.metrics import f1_score
from nilmtk import HDFDataStore

source = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/data/REDD/low_freq'
output = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/data/REDD/redd.h5'

#Convert datato NILMTK format
# - Now redd.h5 holds all the REDD power data and all the relevant metadata.
print("Converting...")
#convert_redd(source, output)

#==============================================================================
#Load Data
# - All the metadata is loaded into memory but none of the power data.
# - It loads the power data in chunks until it is required.
redd = DataSet(output)

redd.set_window(start=None, end='2011-05-01 00:00:00')



#==============================================================================
#Print information about DataSet
#print_dict(redd.metadata)
#print_dict(redd.buildings)
#print_dict(redd.buildings[1].metadata)





#==============================================================================
#Print information about MeterGroups
elec = redd.buildings[1].elec
#elec.draw_wiring_graph()
#elec.mains()
#elec.nested_metergroups()
#elec.submeters()
#
#
#
#
##==============================================================================
##Stats for MeterGroups
#
## -Proportion of energy submetered
#elec.proportion_of_energy_submetered()
#
##-Active, apparent and reactive power
#elec.mains().available_power_ac_types()
#elec.submeters().available_power_ac_types()
#
##-Total Energy
#elec.mains().total_energy()
#
##-Energy per submeter
#energy_per_meter = elec.submeters().energy_per_meter()
#
##-Select meters on the basis of their energy consumption
## energy_per_meter is a DataFrame where each row is a
## power type ('active', 'reactive' or 'apparent').
## All appliance meters in REDD are record 'active' so just select
## the 'active' row:
#energy_per_meter = energy_per_meter.loc['active']
#more_than_20 = energy_per_meter[energy_per_meter > 20]
#instances = more_than_20.index
#
#elec.from_list([ElecMeterID(instance=instance, building=1, dataset='REDD') for instance in instances])
#
##-Draw wiring diagram for the MeterGroup
#elec.draw_wiring_graph()
#
##-Get only the meters immediately downstream of mains
#elec.meters_directly_downstream_of_mains()
#
#
#
#
#
#
#
#
##==============================================================================
##Stats and info for individual meters
#
#fridge_meter = elec['fridge']
#
##-Get upstream meter
#fridge_meter.upstream_meter()
#
##- Metadata
#fridge_meter.device
#
##-Dominant appliance
#fridge_meter.dominant_appliance()
#
##-Total Energy
#fridge_meter.total_energy() # kWh
#
##-Get good sections
#fridge_meter.plot()
#good_sections = fridge_meter.good_sections(full_results=True)
## specifying full_results=False would give us a simple list of
## TimeFrames.  But we want the full GoodSectionsResults object so we can
## plot the good sections...
#good_sections.plot()
#good_sections.combined()
#
##-Dropout rate
#fridge_meter.dropout_rate()
#
##-Only load data from good sections
#fridge_meter.dropout_rate(sections=good_sections.combined())
#
#
#
#
#
##==============================================================================
##Selection
#
##Group
##-Select using a metadta field.
#nilmtk.global_meter_group.select_using_appliances(type='washer dryer')
#nilmtk.global_meter_group.select_using_appliances(category='heating')
#nilmtk.global_meter_group.select_using_appliances(building=1, category='single-phase induction motor')
#nilmtk.global_meter_group.select_using_appliances(building=2, category='laundry appliances')
#
##-Select a group of meters from properties of the meters (not the appliances)
#elec.select(device_model='REDD_whole_house')
#elec.select(sample_period=3)
#
#
##Individual
##-Search for a meter using appliances connected to each meter
#elec['fridge']
#
##-Specifying which instnance
#elec['light', 2]
#
##-To uniquely identify an appliance in nilmtk.global_meter_group then we must specify the dataset name, building instance number, appliance type and appliance instance in a dict
#nilmtk.global_meter_group[{'dataset': 'REDD', 'building': 1, 'type': 'fridge', 'instance': 1}]
#
##-Instance numbering
##ElecMeter and Appliance instance numbers uniquely identify the meter or appliance type within the building, not globally. To uniquely identify a meter globally, we need three keys
#nilmtk.global_meter_group[ElecMeterID(instance=8, building=1, dataset='REDD')]
#
##-Select nested MeterGroup
#elec[[ElecMeterID(instance=3, building=1, dataset='REDD'), ElecMeterID(instance=4, building=1, dataset='REDD')]]
#elec[ElecMeterID(instance=(3,4), building=1, dataset='REDD')]
#
##-Specify Mains by asking for meter instance 0
#elec[ElecMeterID(instance=0, building=1, dataset='REDD')]
#elec.mains() == elec[ElecMeterID(instance=0, building=1, dataset='REDD')]
#
#
#
#
#
#co = CombinatorialOptimisation()
#co.train(elec)
#
#redd.set_window(start='2011-05-01 00:00:00', end=None)
#mains = elec.mains()
#output = HDFDataStore('output.h5', 'w')
#co.disaggregate(mains, output)
##output.store.get('/building1/elec/meter9')[:10]
#output.close()
#disag = DataSet('output.h5')
#disag_elec = disag.buildings[1].elec
#f1_score(disag_elec, elec)