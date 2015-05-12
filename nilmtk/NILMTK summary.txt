
"""
NILM METADATA
"""





"""
NILMTK
"""

#=============Convert data to NILMTK format and load into NILMTK=============
- NILMTK uses an open file format based on the HDF5 binary file format to store both the power data and the metadata. 
- The very first step when using NILMTK is to convert your dataset to the NILMTK HDF5 file format. 
- At first only the metadata is loaded to the dataset. The power data is loaded into memmory in chunks as it is required.


#=============MeterGroup, ElecMeter, selection and basic statistics=============
- All NILM datasets consists of various groupings of electricity meters. 
- We can group the meters by house. Or by the type of appliance they are directly connected to. Or by sample rate. Or by whether the meter is a whole-house “site meter” or an appliance-level submeter, or a circuit-level submeter.

 - In NILMTK v0.2, one of the key classes is MeterGroup which stores a list of meters and allows us to select a subset of meters, aggregate power from all meters and many other functions.

#Stats for MeterGroups
- Proportion of energy submetered
- Active, apparent and reactive power
- Raw data
- Total Energy
- Energy per each submeter
- Selection of submeters the basis of energy consumption (and create new MeterGroups from that)
- Draw wiring diagram

#Stats and infor for individual meters (ElecMeters)
The ElecMeter class represents a single electricity meter. Each ElecMeter has a list of associated Appliance objects. ElecMeter has many of the same stats methods as MeterGroup
- Get upstream meter
- Metadata about the class of meter
- Dominant appliance : If the metadata specifies that a meter has multiple meters connected to it then one of those can be specified as the ‘dominant’ appliance, and this appliance can be retrieved with this method
- Total energy
- Get good sections : We can automatically identify the ‘good sections’ (i.e. the sections where every pair of consecutive samples is less than max_sample_period specified in the dataset metadata).
- Dropout rate : As well as large gaps appearing because the entire system is down, we also get frequent small gaps from wireless sensors dropping data. This is sometimes called ‘dropout’. The dropout rate is a number between 0 and 1 which specifies the proportion of missing samples. 
- Only load data from good sections to all stats functions.

#Selection
- Select subgroups of meters using a metadata field.
- Select a group of meters from properties of the meters 


#=============Processing pipeline, preprocessing and more stats=============
- At the core of NILMTK v0.2 is the concept of an ‘out-of-core’ processing pipeline. 
	- What does that mean? ‘out-of-core’ refers to the ability to handle datasets which are too large to fit into system memory. 
	- NILMTK achieves this by setting up a processing pipeline which handle a chunk of data at a time. 
	- We load small chunks from disk and pull these chunks through a processing pipeline. 
	- Each pipeine is made up of Nodes. These can either be stats nodes or preprocessing nodes. 
	- You only have to dive in an play directly with Nodes and the pipeline if you want to assemble your own stats and preprocessing functions.

- Why store results in their own objects? Because these objects need to know how to combine results from multiple chunks.
	- We can get total energy per day, for example.
	- Load a restricted window of data.

- Apply preprocessing node :
	- Applies an arbitrary Pandas function to every chunk as it moves through the pipeline.
		- Fill gaps in appliance data.

#=============Disaggregation and Metrics=============
1- Open data
2- Select only the first half of the dataset for training
3- Select house
4- Train

- To allow disaggregation to be done on any arbitrarily large dataset, disaggregation output is dumped to disk chunk-by-chunk

# Metrics
1- First we load the disag output exactly as if it were a normal dataset
2- f1_score : all metrics take the same arguments