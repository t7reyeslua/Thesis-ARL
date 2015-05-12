# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:00:01 2015

Combinatorial Optimization using location constraints. Based on the code from
NILMTK combinatorial_optimisation.py.

@author: t7
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np
from pandas import *
import json
from datetime import datetime
from nilmtk.appliance import ApplianceID
from nilmtk.utils import container_to_string, find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.timeframe import merge_timeframes, list_of_timeframe_dicts, TimeFrame
from nilmtk.preprocessing import Clip
import pprint
# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


class CombinatorialOptimisation(object):

    """1 dimensional combinatorial optimisation NILM algorithm.

    Attributes
    ----------
    model : list of dicts
       Each dict has these keys:
           states : list of ints (the power (Watts) used in different states)
           training_metadata : ElecMeter or MeterGroup object used for training 
               this set of states.  We need this information because we 
               need the appliance type (and perhaps some other metadata)
               for each model.
    """

    def __init__(self):
        self.model = []

    def train(self, metergroup, max_num_clusters = 3):
        """Train using 1D CO. Places the learnt model in the `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object

        Notes
        -----
        * only uses first chunk for each meter (TODO: handle all chunks).
        """

        if self.model:
            raise RuntimeError("This implementation of Combinatorial Optimisation"
                               " does not support multiple calls to `train`.")

        num_meters = len(metergroup.meters)
#        if num_meters > 12:
#            max_num_clusters = 2
#        else:
#            max_num_clusters = 3

        for i, meter in enumerate(metergroup.submeters().meters):
            print("Training model for submeter '{}'".format(meter))
            for chunk in meter.power_series(preprocessing=[Clip()]):
                states = cluster(chunk, max_num_clusters)
                self.model.append({
                    'states': states,
                    'training_metadata': meter})
                break  # TODO handle multiple chunks per appliance 
        print("Done training!")
        
    def train_resampled(self, metergroup, max_num_clusters = 2):
        load_kwargs={}
        load_kwargs.setdefault('resample', True)
        load_kwargs.setdefault('sample_period', 60)
        
        
        for i, meter in enumerate(metergroup.submeters().meters):
            print("Training model for submeter '{}'".format(meter))   
            for chunk in meter.power_series(**load_kwargs):
                states = cluster(chunk, max_num_clusters)
                self.model.append({
                    'states': states,
                    'training_metadata': meter})
                break  # TODO handle multiple chunks per appliance 
        print("Done training!")


    def get_ordering_of_appliances_in_state_combinations_table(self):
        appliances_order = [model['training_metadata'].instance() for i, model in enumerate(self.model)]        
        return appliances_order
           

    def get_locations_of_events_within_timespan(self, location_data, timestamp, max_time_difference = 60):
        # Get all events that fall within +- the maximum_time_difference
    
        self.maximum_time_difference = max_time_difference #seconds        
        concurrent_events = location_data.locations[
        (timestamp - DateOffset(seconds = self.maximum_time_difference)):
        (timestamp)]# + DateOffset(seconds = self.maximum_time_difference))]
        
        locs = []
        [locs.extend(j) for j in concurrent_events.values]
        locations_within_timespan = list(set(locs))
        
        return locations_within_timespan
       
    def get_appliances_and_locations_in_state_combination(self, appliances_order, state_combination, location_data):
        #Remove vampire power column
        state_combination = state_combination[:-1]
        appliances_in_state_combination_temp = [appliances_order[v] for v,j in enumerate(state_combination) if ((j != 0) and (j != vampire_power))]        
        
        #Disintegrate any tuple into its individual elements if a tuple exists
        appliances_in_state_combination = []
        for app in appliances_in_state_combination_temp:
            if type(app) is tuple:
                appliances_in_state_combination.extend([y for y in app])
            else:
                appliances_in_state_combination.append(app)                
        
        locations = []
        [locations.extend(location_data.appliances_location[app]) for app in appliances_in_state_combination]        
        return appliances_in_state_combination, list(set(locations)) 
        
    def get_appliances_in_state_combination(self, appliances_order, state_combination, location_data):
        #Remove vampire power column
        state_combination = state_combination[:-1]
        appliances_in_state_combination_temp = [appliances_order[v] for v,j in enumerate(state_combination) if ((j != 0))]        
        
        #Disintegrate any tuple into its individual elements if a tuple exists
        appliances_in_state_combination = []
        for app in appliances_in_state_combination_temp:
            if type(app) is tuple:
                appliances_in_state_combination.extend([y for y in app])
            else:
                appliances_in_state_combination.append(app)                
        
        return appliances_in_state_combination
    
    def appliances_that_changed(self, state_combination, last_state_combination, appliances_order):
        changed_appliances_bool_array = np.isclose(state_combination, last_state_combination)
        index_of_changed_appliances = [i for i, closeEnough in enumerate(changed_appliances_bool_array) if closeEnough == False]
        
        changed_appliances_ON  = [appliances_order[i] for i in index_of_changed_appliances if state_combination[i] > last_state_combination[i]]
        changed_appliances_OFF = [appliances_order[i] for i in index_of_changed_appliances if state_combination[i] < last_state_combination[i]]
        
        return changed_appliances_ON, changed_appliances_OFF
        
    def get_locations_of_appliances(self, appliances, location_data, user_dependent_only = False):
        #Disintegrate any tuple into its individual elements if a tuple exists
        appliances_list = []
        for app in appliances:
            if type(app) is tuple:
                appliances_list.extend([y for y in app])
            else:
                appliances_list.append(app)   
        
        locations = []
        #print('5 Location_data', type(location_data))
#        print (type(location_data))
#        pprint.pprint(location_data.appliances_location)
        if (user_dependent_only):
            [locations.extend(location_data.appliances_location[app]) for app in appliances_list if (app in location_data.user_dependent_appliances)]
        else:
            [locations.extend(location_data.appliances_location[app]) for app in appliances_list]
        return list(set(locations))
        
    def check_if_valid_state_combination(self, state_combination, last_state_combination, valid_locations, appliances_order, location_data):
        changed_to_ON, changed_to_OFF = self.appliances_that_changed(state_combination, last_state_combination, appliances_order)
        
#        pprint.pprint(state_combination)
#        pprint.pprint(last_state_combination)
#        pprint.pprint(valid_locations)
#        pprint.pprint(changed_to_ON)
#        pprint.pprint(changed_to_OFF)
        #print('4 Location_data', type(location_data))
        locations_ON  = self.get_locations_of_appliances(changed_to_ON,  location_data)
        locations_OFF_user_dependent = self.get_locations_of_appliances(changed_to_OFF, location_data, user_dependent_only=True)
        locations_of_changes = locations_ON + locations_OFF_user_dependent

        valid_state_combination = all(location in valid_locations for location in locations_of_changes)
        return valid_state_combination, locations_of_changes    
        
        
    def find_nearest_original_co(self, known_array_sorted, test_value, index_sorted, known_array):
        idx1 = np.searchsorted(known_array_sorted, test_value)
        idx2 = np.clip(idx1 - 1, 0, len(known_array_sorted)-1)
        idx3 = np.clip(idx1,     0, len(known_array_sorted)-1)
    
        diff1 = known_array_sorted[idx3] - test_value
        diff2 = test_value - known_array_sorted[idx2]
    
        index = index_sorted[np.where(diff1 <= diff2, idx3, idx2)]
        residual = test_value - known_array[index]
        
        return index, residual
     
    def get_constrained_state_combinations(self, valid_locations, last_combination_appliances, loc, vampire_power):
        appliances_in_valid_locations = [app for app in loc.appliances_location if all(locs in loc.appliances_location[app] for locs in valid_locations)]
        appliances_in_valid_locations.extend(last_combination_appliances)
        
        #Take care of REDDs tuples names (3,4) and (10,20)
        if 10 in appliances_in_valid_locations:
            appliances_in_valid_locations.remove(10)
            appliances_in_valid_locations.remove(20)
            appliances_in_valid_locations.append((10,20))
        if 3 in appliances_in_valid_locations:
            appliances_in_valid_locations.remove(3)
            appliances_in_valid_locations.remove(4)
            appliances_in_valid_locations.append((3,4))
           
        centroids = [model['states'] for model in co.model if  model['training_metadata'].instance() in appliances_in_valid_locations]

        from sklearn.utils.extmath import cartesian
        state_combinations = cartesian(centroids)
        n_rows = state_combinations.shape[0]
        vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
        state_combinations = np.hstack((state_combinations, vampire_power_array))
        summed_power_of_each_combination = np.sum(state_combinations, axis=1)

        return state_combinations, summed_power_of_each_combination
        
        
    def constrain_existing_state_combinations(self, valid_locations, state_combinations, last_combination, appliances_order, loc):
        appliances_in_valid_locations = [app for app in loc.appliances_location if all(locs in loc.appliances_location[app] for locs in valid_locations)]

        
        #Take care of REDDs tuples names (3,4) and (10,20)
        if 10 in appliances_in_valid_locations:
            appliances_in_valid_locations.remove(10)
            appliances_in_valid_locations.remove(20)
            appliances_in_valid_locations.append((10,20))
        if 3 in appliances_in_valid_locations:
            appliances_in_valid_locations.remove(3)
            appliances_in_valid_locations.remove(4)
            appliances_in_valid_locations.append((3,4))
           
        appliances_not_in_valid_locations = [app for app in appliances_order if app not in appliances_in_valid_locations]
        for c in last_combination:
            try:
                appliances_not_in_valid_locations.remove(c)
            except ValueError:
                pass        
        appliances_not_in_valid_locations_index = [appliances_order.index(app) for app in appliances_not_in_valid_locations]
        sc = DataFrame(state_combinations)
        sc.drop(appliances_not_in_valid_locations_index, axis=1, inplace=True)
        sc = sc.drop_duplicates()
        sc['sum'] = sc.sum(axis=1)
        scs = sc.sort(['sum'], ascending=[True])
        return scs

    def find_nearest_location_constraint(self, 
                                         known_array_original, 
                                         test_value, 
                                         location_data, 
                                         state_combinations_original, 
                                         last_state_combination, 
                                         appliances_order,
                                         valid_locations,
                                         vampire_power):
        
        last_combination_appliances = self.get_appliances_in_state_combination(appliances_order, 
                                                                               last_state_combination, 
                                                                               location_data)
        state_combinations, summed_power_of_each_combination = self.get_constrained_state_combinations(valid_locations, 
                                                                                                       last_combination_appliances, 
                                                                                                       location_data, 
                                                                                                       vampire_power)
        valid_combination = False
        first_guess = -1
        print('Finding nearest combination considering location constraint...')        
        while (valid_combination != True) :
                
            
        return combo, residual
        
    def find_nearest_location_constraint_previous_version(self, 
                                         known_array_original, 
                                         test_value, 
                                         location_data, 
                                         state_combinations_original, 
                                         last_state_combination, 
                                         appliances_order,
                                         valid_locations):
        # We cannot turn ON any appliance if its location (location_data.appliances_location)
        # is not found in the valid_locations. For determining if we are turning ON an appliance
        # we need to compare the found combination with the last_state_combination and check
        # if there is an appliance ON that was OFF the last time.
                                         
                                         
        # We cannot turn OFF any appliance if its location (location_data.appliances_location)
        # is not found in the valid_locations. For determining if we are turning OFF an appliance
        # we need to compare the found combination with the last_state_combination and check
        # if there is an appliance OFF that was ON the last time. This only applies to 
        # appliances in the list of user dependent appliances (location_data.user_dependent_appliances)
        
        valid_combination = False
        known_array = np.copy(known_array_original)
        state_combinations = np.copy(state_combinations_original)
        self.location_used += 1
        self.location_loop = 0
        first_guess = -1
        print('Finding nearest combination considering location constraint...')
        while (valid_combination != True) :
            index_sorted = np.argsort(known_array)
            known_array_sorted = known_array[index_sorted]        
            index, residual = self.find_nearest_original_co(known_array_sorted, test_value, index_sorted, known_array)
            #print('3 Location_data', type(location_data))
            #check if it is a valid state combination choice the one that was found
            valid_combination, locations_of_changes = self.check_if_valid_state_combination(state_combinations[index], 
                                                                 last_state_combination, 
                                                                 valid_locations,
                                                                 appliances_order,
                                                                 location_data)
            if (first_guess == -1):
                first_guess = index
                
            #if (valid_combination != True):
            # delete that combination from the pool of possible ones
            temp = known_array[index]
            known_array[index] = -1.0 # we make it invalid
            self.location_loop += 1
            combos_left = len(known_array) - len(known_array[known_array==-1])
            tempstr = str(self.location_loop) + '...value=' + str(test_value) + '...' + str(temp) + '...Diff='+ str(residual) + '...index=' + str(index) + '...'+ str(combos_left) + ' ['+','.join(valid_locations) + ']'+ ', ['+','.join(locations_of_changes) + ']' 
            self.out.append(tempstr)
            print('Looking again for ', 
                  str(test_value),
                  '...', 
                  str(temp), str(residual), str(self.location_loop), str(index), str(combos_left), '[',','.join(valid_locations) , ']', '[',','.join(locations_of_changes) , ']')
            if (abs(residual) > 5):
                index = first_guess
                valid_combination = True
                print('Residual too large, going back to initial guess:', str(known_array_original[index]))
                self.out.append('Residual too large, going back to initial guess:' + str(known_array_original[index]))
                
            
        return index, residual
    
       
    def find_nearest(self, test_array, location_data, vampire_power):
        indices = np.array(test_array)
        indices = indices.astype(np.int64)
        residuals = np.array(test_array)
        appliances_order = self.get_ordering_of_appliances_in_state_combinations_table()
        
        last_state_combination = -1
        for i, test_value in enumerate(test_array.values):
            valid_locations = self.get_locations_of_events_within_timespan(location_data, test_array.index[i]) 
            print(i,'===========================================================')
                
            if (len(valid_locations) == 0):
                index, residual = self.find_nearest_original_co(known_array_sorted, 
                                                                test_value, 
                                                                index_sorted,
                                                                known_array)
            else:
                index, residual = self.find_nearest_location_constraint(known_array, 
                                                                        test_value, 
                                                                        location_data, 
                                                                        state_combinations,
                                                                        last_state_combination,
                                                                        appliances_order,
                                                                        valid_locations,
                                                                        vampire_power)
            
            
            indices[i] = index
            residuals[i] = residual
            last_state_combination = state_combinations[index]        
        return indices, residuals
        
    def find_nearest_using_location_previous_version(self, known_array, test_array, location_data, state_combinations):

        """Find closest value in `known_array` for each element in `test_array`.
    
        Parameters
        ----------
        known_array : numpy array
            consisting of scalar values only; shape: (m, 1)
        test_array : numpy array
            consisting of scalar values only; shape: (n, 1)
        state_combinations: numpy array
            j*k matrix; shape: (j, k). j = number of combinations/ k = number of appliances
        
    
        Returns
        -------
        indices : numpy array; shape: (n, 1)
            For each value in `test_array` finds the index of the closest value
            in `known_array`.
        residuals : numpy array; shape: (n, 1)
            For each value in `test_array` finds the difference from the closest
            value in `known_array`.
        """
        #indices_of_state_combinations, residual_power = self.find_nearest(
        #        summed_power_of_each_combination, chunk, location_data, state_combinations)
        # from http://stackoverflow.com/a/20785149/732596
    
        #known_array = summed_power_of_each_combination
        #test array  = values that we want to disaggregate
        if (location_data is None):            
            print('No location data provided. Calculating as in original method...')
        
        index_sorted = np.argsort(known_array)
        known_array_sorted = known_array[index_sorted]
    
        indices = np.array(test_array)
        indices = indices.astype(np.int64)
        residuals = np.array(test_array)
        appliances_order = self.get_ordering_of_appliances_in_state_combinations_table()
        
        last_state_combination = -1
        
        for i, test_value in enumerate(test_array.values):
            valid_locations = []
            if (location_data != None):
                valid_locations = self.get_locations_of_events_within_timespan(location_data, test_array.index[i])    
                
                print(i,'===========================================================')
                self.out.append(str(i) + '===========================================================' )
                
            if (len(valid_locations) == 0):
                index, residual = self.find_nearest_original_co(known_array_sorted, 
                                                           test_value, 
                                                           index_sorted,
                                                           known_array)
            else:  
                index, residual = self.find_nearest_location_constraint(known_array, 
                                                                   test_value, 
                                                                   location_data, 
                                                                   state_combinations,
                                                                   last_state_combination,
                                                                   appliances_order,
                                                                   valid_locations)
            
            
            indices[i] = index
            residuals[i] = residual
            last_state_combination = state_combinations[index]
            
            #print (i, test_value, known_array[index], residual, index)
            #pprint.pprint(last_state_combination)
            
        #pprint.pprint(indices)        
        return indices, residuals

    def disaggregate(self, mains, output_datastore, location_data=None, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        output_name : string, optional
            The `name` to use in the metadata for the `output_datastore`.
            e.g. some sort of name for this experiment.  Defaults to 
            "NILMTK_CO_<date>"
        resample_seconds : number, optional
            The desired sample period in seconds.
        location_data: LocationInference object
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        MIN_CHUNK_LENGTH = 100
        
        if not self.model:
            raise RuntimeError("The model needs to be instantiated before"
                               " calling `disaggregate`.  For example, the"
                               " model can be instantiated by running `train`.")

        # If we import sklearn at the top of the file then auto doc fails.
        from sklearn.utils.extmath import cartesian

        # sklearn produces lots of DepreciationWarnings with PyTables
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Extract optional parameters from load_kwargs
        date_now = datetime.now().isoformat().split('.')[0]
        output_name = load_kwargs.pop('output_name', 'NILMTK_CO_' + date_now)
        resample_seconds = load_kwargs.pop('resample_seconds', 60)

        # Get centroids
        centroids = [model['states'] for model in self.model]
        state_combinations = cartesian(centroids)
        # state_combinations is a 2D array
        # each column is a chan
        # each row is a possible combination of power demand values e.g.
        # [[0, 0, 0, 0], [0, 0, 0, 100], [0, 0, 50, 0], [0, 0, 50, 100], ...]

        # Add vampire power to the model
        vampire_power = mains.vampire_power()
        print("vampire_power = {} watts".format(vampire_power))
        n_rows = state_combinations.shape[0]
        vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
        state_combinations = np.hstack((state_combinations, vampire_power_array))

        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
        # summed_power_of_each_combination is now an array where each
        # value is the total power demand for each combination of states.

        load_kwargs['sections'] = load_kwargs.pop('sections',
                                                  mains.good_sections())
        load_kwargs.setdefault('resample', True)
        load_kwargs.setdefault('sample_period', resample_seconds)
        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = '{}/elec/meter1'.format(building_path)

        self.out = []
        self.location_used = 0
        self.location_loop = 0
        self.comparison = {}
        self.co_indices_original = []
        self.co_indices_location = []
        self.co_residuals_original = []
        self.co_residuals_location = []
        for chunk in mains.power_series(**load_kwargs):

            # Check that chunk is sensible size before resampling
            if len(chunk) < MIN_CHUNK_LENGTH:
                continue

            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            # Start disaggregation
            print('Calculating original indices of state combinations...')
            indices_of_state_combinations_original, residuals_power_original = find_nearest(
                summed_power_of_each_combination, chunk.values)            
            
            self.co_indices_original.extend(indices_of_state_combinations_original)
            self.co_residuals_original.extend(residuals_power_original)
            
            print('Calculating indices of state combinations...')
            indices_of_state_combinations, residuals_power_location = self.find_nearest(chunk, location_data, vampire_power)
            
            self.co_indices_location.extend(indices_of_state_combinations)
            self.co_residuals_location.extend(residuals_power_location)
                  
            for i, model in enumerate(self.model):
                print("Estimating power demand for '{}'".format(model['training_metadata']))
                predicted_power = state_combinations[indices_of_state_combinations, i].flatten()
                cols = pd.MultiIndex.from_tuples([chunk.name])
                meter_instance = model['training_metadata'].instance()
                output_datastore.append('{}/elec/meter{}'
                                        .format(building_path, meter_instance),
                                        pd.DataFrame(predicted_power,
                                                     index=chunk.index,
                                                     columns=cols))

            # Copy mains data to disag output
            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))

        ##################################
        # Add metadata to output_datastore

        # TODO: `preprocessing_applied` for all meters
        # TODO: split this metadata code into a separate function
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.

        # DataSet and MeterDevice metadata:
        meter_devices = {
            'CO': {
                'model': 'CO',
                'sample_period': resample_seconds,
                'max_sample_period': resample_seconds,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            },
            'mains': {
                'model': 'mains',
                'sample_period': resample_seconds,
                'max_sample_period': resample_seconds,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            }
        }

        merged_timeframes = merge_timeframes(timeframes, gap=resample_seconds)
        total_timeframe = TimeFrame(merged_timeframes[0].start,
                                    merged_timeframes[-1].end)

        dataset_metadata = {'name': output_name, 'date': date_now,
                            'meter_devices': meter_devices,
                            'timeframe': total_timeframe.to_dict()}
        output_datastore.save_metadata('/', dataset_metadata)

        # Building metadata

        # Mains meter:
        elec_meters = {
            1: {
                'device_model': 'mains',
                'site_meter': True,
                'data_location': mains_data_location,
                'preprocessing_applied': {},  # TODO
                'statistics': {
                    'timeframe': total_timeframe.to_dict(),
                    'good_sections': list_of_timeframe_dicts(merged_timeframes)
                }
            }
        }

        # Appliances and submeters:
        appliances = []
        for model in self.model:
            meter = model['training_metadata']

            meter_instance = meter.instance()

            for app in meter.appliances:
                meters = app.metadata['meters']
                appliance = {
                    'meters': [meter_instance], 
                    'type': app.identifier.type,
                    'instance': app.identifier.instance
                    # TODO this `instance` will only be correct when the
                    # model is trained on the same house as it is tested on.
                    # https://github.com/nilmtk/nilmtk/issues/194
                }
                appliances.append(appliance)

            elec_meters.update({
                meter_instance: {
                    'device_model': 'CO',
                    'submeter_of': 1,
                    'data_location': ('{}/elec/meter{}'
                                      .format(building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict(),
                        'good_sections': list_of_timeframe_dicts(merged_timeframes)
                    }
                }
            })

            #Setting the name if it exists
            if meter.name:
                if len(meter.name)>0:
                    elec_meters[meter_instance]['name'] = meter.name

        building_metadata = {
            'instance': mains.building(),
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)

