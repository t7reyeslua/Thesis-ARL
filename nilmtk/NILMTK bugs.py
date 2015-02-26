elec.from_list([ElecMeterID(instance=instance, building=1, dataset='REDD') for instance in instances])
Traceback (most recent call last):

  File "<ipython-input-23-03fe343586b4>", line 1, in <module>
    elec.from_list([ElecMeterID(instance=instance, building=1, dataset='REDD') for instance in instances])

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/metergroup.py", line 405, in from_list
    meters.append(self[meter_id])

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/metergroup.py", line 220, in __getitem__
    raise KeyError(key)

KeyError: ElecMeterID(instance=MeterGroupID(meters=(ElecMeterID(instance=10, building=1, dataset='REDD'), ElecMeterID(instance=20, building=1, dataset='REDD'))), building=1, dataset='REDD')


BUG when create a new MeterGroup from a list of ElecMeterIDs..different than tutorial!!!!!!!!!!!












============================================================
fridge_meter.dropout_rate(sections=good_sections.combined())
Traceback (most recent call last):

  File "<ipython-input-52-b2055d9dc06b>", line 1, in <module>
    fridge_meter.dropout_rate(sections=good_sections.combined())

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 617, in dropout_rate
    nodes, DropoutRate.results_class(), loader_kwargs)

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 688, in _get_stat_from_cache_or_compute
    results_obj.import_from_cache(cached_stat, sections)

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/results.py", line 158, in import_from_cache
    rows_matching_start = cached_stat.loc[section.start]

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/indexing.py", line 1202, in __getitem__
    return self._getitem_axis(key, axis=0)

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/indexing.py", line 1345, in _getitem_axis
    self._has_valid_type(key, axis)

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/indexing.py", line 1307, in _has_valid_type
    error()

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/indexing.py", line 1292, in error
    "cannot use label indexing with a null key")

ValueError: cannot use label indexing with a null key


BUG dropout rate functions in v0.2 only gives the data from good sections..therefore, if you give the good sections as parameter there is an error.







================================================================

from nilmtk.timeframe import timeframes_from_periodindex

BUG import....just import nilmtk.timeframe instead of timeframes_from_periodindex











==================================================================
energy_per_day = fridge_meter.total_energy(full_results=True, sections=period_index)
Traceback (most recent call last):

  File "<ipython-input-51-2c99323b6905>", line 1, in <module>
    energy_per_day = fridge_meter.total_energy(full_results=True, sections=period_index)

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 596, in total_energy
    nodes, TotalEnergy.results_class(), loader_kwargs)

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 682, in _get_stat_from_cache_or_compute
    sections = [s for s in sections if not s.empty]

AttributeError: 'Period' object has no attribute 'empty'


BUG ..a workaround would be to load a restricted window of data if you want to know the consumption within a time frame




=========================================================================
dropout_rate2.run()
Traceback (most recent call last):

  File "<ipython-input-102-4ae17e8d94e7>", line 1, in <module>
    dropout_rate2.run()

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/node.py", line 42, in run
    for _ in self.process():

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/stats/dropoutrate.py", line 16, in process
    metadata = self.upstream.get_metadata()

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/node.py", line 80, in get_metadata
    metadata = self.upstream.get_metadata()

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/node.py", line 75, in get_metadata
    results_dict = self.results.to_dict()

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/stats/dropoutrateresults.py", line 39, in to_dict
    return {'statistics': {'dropout_rate': self.combined()}}

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/stats/dropoutrateresults.py", line 24, in combined
    tot_samples = self._data['n_samples'].sum()

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/frame.py", line 1780, in __getitem__
    return self._getitem_column(key)

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/frame.py", line 1787, in _getitem_column
    return self._get_item_cache(key)

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/generic.py", line 1068, in _get_item_cache
    values = self._data.get(item)

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/internals.py", line 2849, in get
    loc = self.items.get_loc(item)

  File "/home/t7/anaconda/lib/python2.7/site-packages/pandas/core/index.py", line 1402, in get_loc
    return self._engine.get_loc(_values_from_object(key))

  File "pandas/index.pyx", line 134, in pandas.index.IndexEngine.get_loc (pandas/index.c:3807)

  File "pandas/index.pyx", line 154, in pandas.index.IndexEngine.get_loc (pandas/index.c:3687)

  File "pandas/hashtable.pyx", line 696, in pandas.hashtable.PyObjectHashTable.get_item (pandas/hashtable.c:12310)

  File "pandas/hashtable.pyx", line 704, in pandas.hashtable.PyObjectHashTable.get_item (pandas/hashtable.c:12261)

KeyError: 'n_samples'

NO WORKAROUND YET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!










=====================================================================================
redd.store.window = TimeFrame(start=None, end='2011-05-01 00:00:00-04:00')
/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/timeframe.py:246: UserWarning: Using a pytz._FixedOffset timezone may cause issues (e.g. might cause Pandas to raise 'TypeError: too many timezones in this block, create separate data columns').  It is better to set the timezone to a geographical location e.g. 'Europe/London'.
  warn("Using a pytz._FixedOffset timezone may cause issues"



BUG https://github.com/nilmtk/nilmtk/issues/342




=====================================================================================
/home/t7/anaconda/envs/nilmtk/lib/python2.7/site-packages/sklearn/metrics/metrics.py:1771: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
  'precision', 'predicted', average, warn_for)
/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py:786: PendingDeprecationWarning: object.__format__ with a non-empty format string is deprecated
  .format(self.building(), self.instance(), stat_name))
Traceback (most recent call last):
  File "test.py", line 190, in <module>
    f1_score(disag_elec, elec)
  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/metrics.py", line 222, in f1_score
    'when_on'):
  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/electric.py", line 785, in align_two_meters
    sections = master.good_sections()
  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 635, in good_sections
    nodes, results_obj, loader_kwargs)        
  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 685, in _get_stat_from_cache_or_compute
    key_for_cached_stat = self.key_for_cached_stat(results_obj.name)
  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 786, in key_for_cached_stat
    .format(self.building(), self.instance(), stat_name))
ValueError: Unknown format code 'd' for object of type 'str'


/home/t7/anaconda/lib/python2.7/site-packages/sklearn/metrics/metrics.py:1771: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
  'precision', 'predicted', average, warn_for)
/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py:786: PendingDeprecationWarning: object.__format__ with a non-empty format string is deprecated
  .format(self.building(), self.instance(), stat_name))
Traceback (most recent call last):

  File "<ipython-input-5-9b5a04a7cbc9>", line 1, in <module>
    f1_score(disag_elec, elec)

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/metrics.py", line 222, in f1_score
    'when_on'):

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/electric.py", line 785, in align_two_meters
    sections = master.good_sections()

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 635, in good_sections
    nodes, results_obj, loader_kwargs)

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 685, in _get_stat_from_cache_or_compute
    key_for_cached_stat = self.key_for_cached_stat(results_obj.name)

  File "/home/t7/Dropbox/Documents/TUDelft/Thesis/Software/nilmtk/nilmtk/elecmeter.py", line 786, in key_for_cached_stat
    .format(self.building(), self.instance(), stat_name))

ValueError: Unknown format code 'd' for object of type 'str'
