# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:23:56 2015

@author: t7
"""
from nilmtk.metergroup import MeterGroup
from pandas import DataFrame, Series, DateOffset
from nilmtk import DataSet
from nilmtk.metrics import f1_score, error_in_assigned_energy, fraction_energy_assigned_correctly, mean_normalized_error_power, rms_error_power
import numpy as np
from nilmtk import HDFDataStore
import sys
mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/nilmtk'
sys.path.append(mypath)

from combinatorial_optimisation_location import CombinatorialOptimisation
from location_inference import LocationInference

from pylab import rcParams
import matplotlib.pyplot as plt

    
def find_nearest1(test_value, known_array):                                                                                    
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]
            
    idx1 = np.searchsorted(known_array_sorted, test_value)
    idx2 = np.clip(idx1 - 1, 0, len(known_array_sorted)-1)
    idx3 = np.clip(idx1,     0, len(known_array_sorted)-1)

    diff1 = known_array_sorted[idx3] - test_value
    diff2 = test_value - known_array_sorted[idx2]

    index = index_sorted[np.where(diff1 <= diff2, idx3, idx2)]
    residual = test_value - known_array[index]
    
    return index, residual

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
    
def get_gt_state_combinations(gt_apps, loc, vampire_power, timestamp):
    
    values = {}
    for app in gt_apps:
        values[app] = ps[app][timestamp] 
    
    try:
        v34 = values[3] + values[4]
        del values[3]
        del values[4]
        values[(3,4)] = v34
    except Exception:
        tpt = 0  
    
    try:
        v1020 = values[10] + values[20]
        del values[10]
        del values[20]
        values[(10,20)] = v1020
    except Exception:
        tpt = 0
    #gt_apps = list(gt_apps_orig)
    #Take care of REDDs tuples names (3,4) and (10,20)   
    
    
    if loc.name == 'REDD':
        if 10 in gt_apps:
            gt_apps.remove(10)
            gt_apps.remove(20)
            gt_apps.append((10,20))
        if 3 in gt_apps:
            gt_apps.remove(3)
            gt_apps.remove(4)
            gt_apps.append((3,4))
       
    centroids_gt = []
    ordering = []

    for model in co.model:
        try:
            if  model['training_metadata'].instance() in gt_apps:
                centroids_gt.append(model['states'])
                ordering.append(model['training_metadata'].instance())
        except Exception:
            for app in gt_apps:
                try:
                    if model['training_metadata'].instance() == app:
                        centroids_gt.append(model['states'])
                        ordering.append(model['training_metadata'].instance())
                except Exception:
                    continue                    
    
    #We know all these appliances are ON, take away the states when they are off
    centroids_on = {}
    for i,centroid_array in enumerate(centroids_gt):
        cd = [centroid for centroid in centroid_array if centroid != 0]
        centroids_on[gt_apps[i]] = np.array(cd)
        
    state_combinations =  [(v, find_nearest(centroids_on[v], values[v])) for v in values]   
    values_of_combination = [find_nearest(centroids_on[v], values[v]) for v in values] 
    summed_power_of_combination = sum(values_of_combination) + vampire_power
    
#    from sklearn.utils.extmath import cartesian
#    state_combinations = cartesian(centroids_on)
#    n_rows = state_combinations.shape[0]
#    vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
#    state_combinations = np.hstack((state_combinations, vampire_power_array))
#    summed_power_of_each_combination = np.sum(state_combinations, axis=1)

    return state_combinations, summed_power_of_combination, ordering

        
#%matplotlib inline
rcParams['figure.figsize'] = (14, 6)
plt.style.use('ggplot')


loc = LocationInference('REDD')
loc.dataset.set_window(start='2011-04-19 00:00:00', end='2011-04-27 00:00:00')
loc.infer_locations()


print("Training...")
co = CombinatorialOptimisation()
co.train(loc.elec, max_num_clusters = 2, resample_seconds=60)

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


print("Disagreggating...")
loc.dataset.set_window(start=None, end='2011-04-19 00:00:00')
loc.infer_locations()
mains = loc.elec.mains()

disagout = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd-disag-loc.h5'
output = HDFDataStore(disagout, 'w')
co.disaggregate(mains, output, location_data=loc, resample_seconds=60)
output.close()


#METRICS=======================================================================

print("Metrics...")
disag = DataSet(disagout)
disag_elec = disag.buildings[1].elec
f1_loc = f1_score(disag_elec, loc.elec)

disagouto = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd-disag-original-modified-centroids.h5'
print("Metrics...")
disago = DataSet(disagouto)
disago_elec = disago.buildings[1].elec
f1_original = f1_score(disago_elec, loc.elec)

diff1  = [f1_loc.values[i] - f1_original.values[i] for i,v in enumerate(f1_loc)]
s1 = Series(diff1, index=f1_loc.index)


#fraction_energy_assigned_correctly 
predictions = disago_elec
ground_truth = loc.elec

predictions_submeters = MeterGroup(meters=predictions.submeters().meters)
ground_truth_submeters = MeterGroup(meters=ground_truth.submeters().meters)

fraction_per_meter_predictions_original = predictions_submeters.fraction_per_meter()
fraction_per_meter_ground_truth = ground_truth_submeters.fraction_per_meter()

fraction_original = 0
for meter_instance,v in enumerate(predictions_submeters.instance()):
    fraction_original += min(fraction_per_meter_ground_truth.values[meter_instance],
                    fraction_per_meter_predictions_original.values[meter_instance])
                
predictions = disag_elec
predictions_submeters = MeterGroup(meters=predictions.submeters().meters)
fraction_per_meter_predictions_loc = predictions_submeters.fraction_per_meter()

fraction_loc = 0
for meter_instance,v in enumerate(predictions_submeters.instance()):
    fraction_loc += min(fraction_per_meter_ground_truth.values[meter_instance],
                    fraction_per_meter_predictions_loc.values[meter_instance])




#HELPERS=======================================================================
from sklearn.utils.extmath import cartesian
centroids = [model['states'] for model in co.model]
state_combinations = cartesian(centroids)

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

load_kwargs={}
load_kwargs.setdefault('resample', True)
load_kwargs.setdefault('sample_period', 60)
ps = {}
for i in  loc.min_power_threshold:
    ps[i] = list(loc.elec[i].power_series(**load_kwargs))[0]
#COMPARISON====================================================================
#time, aggregated_energy(mains), locations [appliances that triggered location], co_original_combo [summed power of combo], co_loc_combo [summed power of combo]


template = "{0:26}|{1:6}{2:10}|{3:20}|{4:20}"
locations_lists  = []
appliances_lists = []
timestamps_list = []
mains_values = []
gt = []
gt_sums = []
gt_residuals = []
gt_states = []
for chunk in chunks:
    for ts, value in enumerate(chunk):
        timestamp = chunk.index[ts]
        concurrent_events = loc.events_locations['Locations'][(timestamp - DateOffset(seconds = 60)):(timestamp)]
        concurrent_appliances = loc.events_locations['Events'][(timestamp - DateOffset(seconds = 60)):(timestamp)]
        
        gt_appliances = None
        gt_apps = []
        for gt_event_ts in loc.appliances_status.index:
            if gt_event_ts <= timestamp:
                gt_appliances = loc.appliances_status[str(gt_event_ts)]
                gt_ts = gt_event_ts
        if gt_appliances is not None:
            gt_apps = [v for i,v in enumerate(gt_appliances) if gt_appliances.values[0][i] == True]  
        
        
        if (len(gt_apps) == 0):
            gt.append([])
            gt_sums.append(0)
            gt_residuals.append(0)
            gt_states.append([])
        else:
            gt_state_combinations, summ, order_of_appliances = get_gt_state_combinations(
                                                                        gt_apps,
                                                                        loc, 
                                                                        vampire_power,
                                                                        timestamp)
            #index_gt, residual_gt = find_nearest(value, summed_power_of_each_gt_combination)
            
            gt_apps1 = [v[0] for v in gt_state_combinations if v[1] not in (0,vampire_power)]

            gt.append(gt_apps1)
            gt_sums.append("{0:.2f}".format(summ))
            gt_residuals.append("{0:.2f}".format((summ-value)))
            
            #gt_sc = [int(v) for v in gt_state_combinations if v not in (0,vampire_power)] 
            gt_sc = [int(v[1]) for v in gt_state_combinations if v[1] not in (0,vampire_power)]
            gt_states.append(gt_sc)
        
        
        locs = []
        [locs.extend(j) for j in concurrent_events.values]
        locations_within_timespan = list(set(locs))
        apps = []
        [apps.extend(j) for j in concurrent_appliances.values]
        appliances_within_timespan = list(set(apps))
        
        timestamps_list.append(timestamp)
        mains_values.append("{0:.2f}".format(value))
        locations_lists.append(locs)
        appliances_lists.append(apps)
                
        #print template.format(str(timestamp), 'mains:', '{0:.4f}'.format(value), str(appliances_within_timespan), str(locations_within_timespan))

co_original_combo_sums = [ "{0:.2f}".format(summed_power_of_each_combination[index]) for index in co.co_indices_original]
co_location_combo_sums_t = np.sum(co.co_combos_location, axis=1)
co_location_combo_sums = [ "{0:.2f}".format(summm) for summm in co_location_combo_sums_t ]


appliances_order = [model['training_metadata'].instance() for i, model in enumerate(co.model)]
co_original_combos = [co.get_appliances_in_state_combination(appliances_order, state_combinations[index], loc) for index             in co.co_indices_original]
co_location_combos = [co.get_appliances_in_state_combination(appliances_order, state_combination        , loc) for state_combination in co.co_combos_location]

for combo in co_original_combos:    
    if 10 in combo:
        combo.remove(10)
        combo.remove(20)
        combo.append((10,20))
    if 3 in combo:
        combo.remove(3)
        combo.remove(4)
        combo.append((3,4))

for combo in co_location_combos:    
    if 10 in combo:
        combo.remove(10)
        combo.remove(20)
        combo.append((10,20))
    if 3 in combo:
        combo.remove(3)
        combo.remove(4)
        combo.append((3,4))

combo_values_considered_original = [[int(v) for v in state_combinations[index] if v not in (0,vampire_power)] for index in co.co_indices_original]
combo_values_considered_loc = [[int(v) for v in combo if v not in (0,vampire_power)] for combo in co.co_combos_location]

co_residuals_original = ["{0:.2f}".format(residual) for residual in co.co_residuals_original]
co_residuals_location = ["{0:.2f}".format(residual) for residual in co.co_residuals_location]

gt_sum_res  = [(gt_sums[i],gt_residuals[i])for i,v in enumerate(gt_sums)]
co_sum_res  = [(co_original_combo_sums[i],co_residuals_original[i])for i,v in enumerate(co_original_combo_sums)]
loc_sum_res = [(co_location_combo_sums[i],co_residuals_location[i])for i,v in enumerate(co_location_combo_sums)]

gt[0] = gt[1]
gt_states[0] = gt_states[1]
gt_combo_states = [[(gt[i_combo][i_app], gt_states[i_combo][i_app]) for i_app, app in enumerate(combo)]for i_combo, combo in enumerate(gt)]

co_combo_states  = [[(co_original_combos[i_combo][i_app], combo_values_considered_original[i_combo][i_app]) for i_app, app in enumerate(combo)]for i_combo, combo in enumerate(co_original_combos)]
loc_combo_states = [[(co_location_combos[i_combo][i_app],      combo_values_considered_loc[i_combo][i_app]) for i_app, app in enumerate(combo)]for i_combo, combo in enumerate(co_location_combos)]


jaccard_co  = []
jaccard_loc = []
jaccard_co_states  = []
jaccard_loc_states = []
for i in range (0, len(gt_combo_states)):
    gt_apps =  [app_state[0] for app_state in gt_combo_states[i]]
    co_apps =  [app_state[0] for app_state in co_combo_states[i]]
    loc_apps = [app_state[0] for app_state in loc_combo_states[i]]
    
    gt_u_co  = list(set(gt_apps) | set(co_apps))
    gt_u_loc = list(set(gt_apps) | set(loc_apps))
    gt_n_co  = list(set(gt_apps) & set(co_apps))
    gt_n_loc = list(set(gt_apps) & set(loc_apps))
    
    jaccard_co.append((len(gt_n_co), len(gt_u_co)))
    jaccard_loc.append((len(gt_n_loc), len(gt_u_loc)))

    
    gt_n_co_gt_states = []
    for app in gt_n_co:
        for app_state in gt_combo_states[i]:
            try:
                if app_state[0] == app:
                    gt_n_co_gt_states.append(app_state[1])
            except Exception:
                continue
          
    gt_n_co_co_states = []
    for app in gt_n_co:
        for app_state in co_combo_states[i]:
            try:
                if app_state[0] == app:
                    gt_n_co_co_states.append(app_state[1])
            except Exception:
                continue
            
    #gt_n_co_gt_states = [item for sublist in gt_n_co_gt_states for item in sublist]
    #gt_n_co_co_states = [item for sublist in gt_n_co_co_states for item in sublist]

    gt_n_loc_gt_states = []
    for app in gt_n_loc:
        for app_state in gt_combo_states[i]:
            try:
                if app_state[0] == app:
                    gt_n_loc_gt_states.append(app_state[1])
            except Exception:
                continue
    gt_n_loc_loc_states = []
    for app in gt_n_loc:
        for app_state in loc_combo_states[i]:
            try:
                if app_state[0] == app:
                    gt_n_loc_loc_states.append(app_state[1])
            except Exception:
                continue
    #gt_n_loc_gt_states  = [item for sublist in gt_n_loc_gt_states  for item in sublist]
    #gt_n_loc_loc_states = [item for sublist in gt_n_loc_loc_states for item in sublist]

    gt_n_co_states  = list(set(gt_n_co_gt_states) & set(gt_n_co_co_states))
    gt_n_loc_states = list(set(gt_n_loc_gt_states) & set(gt_n_loc_loc_states))
    
    jaccard_co_states.append((len(gt_n_co_states), len(gt_n_co)))
    jaccard_loc_states.append((len(gt_n_loc_states), len(gt_n_loc)))

jaccard_co_apps_states = []
jaccard_loc_apps_states = []
n_jacc_app_co = 0
n_jacc_app_loc = 0
n_jacc_states_co = 0
n_jacc_states_loc = 0
n_jacc_app_gt_co = 0
n_jacc_app_gt_loc = 0
n_jacc_states_gt_co = 0
n_jacc_states_gt_loc = 0
for i in range(0, len(jaccard_co)):
    jacc_app_co = jaccard_co[i][0]
    jacc_app_gt_co = jaccard_co[i][1]
    
    jacc_app_loc = jaccard_loc[i][0]
    jacc_app_gt_loc = jaccard_loc[i][1]
    
    jacc_states_co = jaccard_co_states[i][0]
    jacc_states_gt_co = jaccard_co_states[i][1]
    
    jacc_states_loc = jaccard_loc_states[i][0] 
    jacc_states_gt_loc = jaccard_loc_states[i][1] 
    
    n_jacc_app_co        += jacc_app_co
    n_jacc_app_loc       += jacc_app_loc
    n_jacc_states_co     += jacc_states_co
    n_jacc_states_loc    += jacc_states_loc
    n_jacc_app_gt_co     += jacc_app_gt_co
    n_jacc_app_gt_loc    += jacc_app_gt_loc
    n_jacc_states_gt_co  += jacc_states_gt_co
    n_jacc_states_gt_loc += jacc_states_gt_loc

    jaccard_co_apps_states.append((jaccard_co[i],jaccard_co_states[i]))
    jaccard_loc_apps_states.append((jaccard_loc[i],jaccard_loc_states[i]))



ptg_co_apps = "{0:.2f}%".format(100* n_jacc_app_co/n_jacc_app_gt_co)
ptg_loc_apps = "{0:.2f}%".format(100* n_jacc_app_loc/n_jacc_app_gt_loc)
ptg_co_states = "{0:.2f}%".format(100* n_jacc_states_co/n_jacc_states_gt_co)
ptg_loc_states = "{0:.2f}%".format(100* n_jacc_states_loc/n_jacc_states_gt_loc)
jaccard_results = {
                    'CO apps':((n_jacc_app_co,n_jacc_app_gt_co),ptg_co_apps),
                    'Loc apps':((n_jacc_app_loc,n_jacc_app_gt_loc),ptg_loc_apps),
                    'CO states':((n_jacc_states_co,n_jacc_states_gt_co),ptg_co_states),
                    'Loc states':((n_jacc_states_loc,n_jacc_states_gt_loc),ptg_loc_states)
                   }

      
comparison = {}
comparison['01. Locs w/event'] = locations_lists
comparison['02. Apps w/event'] = appliances_lists
#comparison['03. GT'] = gt
comparison['03. GT combo/states'] = gt_combo_states
#comparison['04. CO combo'] = co_original_combos
#comparison['05. Loc combo'] = co_location_combos
comparison['04. CO combo/states'] = co_combo_states
comparison['05. Loc combo/states'] = loc_combo_states
#comparison['06. GT states'] = gt_states
#comparison['07. CO states'] = combo_values_considered_original
#comparison['08. Loc states'] = combo_values_considered_loc
comparison['06. Mains'] = mains_values
comparison['07. GT sum/res'] = gt_sum_res
comparison['08. CO sum/res'] = co_sum_res
comparison['09. Loc sum/res'] = loc_sum_res
comparison['10. CO jacc app/states'] = jaccard_co_apps_states
comparison['11. Loc jacc app/states'] = jaccard_loc_apps_states
#comparison['12. CO states jacc'] = jaccard_co_states
#comparison['13. Loc states jacc'] = jaccard_loc_states

#comparison['10. GT sum'] = gt_sums
#comparison['11. CO sum'] = co_original_combo_sums
#comparison['12. Loc sum'] = co_location_combo_sums
#comparison['13. GT res'] = gt_residuals
#comparison['14. CO res'] = co_residuals_original
#comparison['15. Loc res'] = co_residuals_location

d = DataFrame(comparison, index=timestamps_list)
ii = [5,6,7,8,9,11,12,13,14,15,16,17,18,19,(3,4),(10,20)]
cc = Series(centroids, index=ii)
