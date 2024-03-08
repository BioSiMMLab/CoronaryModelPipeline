# Computational pipeline to automatically generate left coronary artery models.
To execute this pipeline, type the following in your command prompt/terminal

`location/to/simvascular/build/on/your/computer --python -- /location/where/this/pipeline/is/stored/on/your/computer/StartModelingPipeline.py`

## scripts and their functionality
| Script name           | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| StartModelingPipeline.py | This script is to be executed to initiate the pipeline. It calls the other scripts (for centerline modification, model generation, meshing, and HPC file generation). You may need to modify this. |
| ModifyCenterline.py  | Contains functions to modify a set of 3D points describing the centerline path and radius description for each geometry to build.                    |
| CreateVessel.py      | SVpython functions for generating paths, segmentations, models, and meshes.                     |
| AllometricScaling.py | Script with allometric scaling functions for cardiac output, mean pressure, and capacitance.   |
| InputSVSolver.py     | Generates necessary presimulation files (sim.svpre, solver.inp) for running SVsolver simulations. |
| SBatchGen.py         | Generates HPC .job files that can be used to run these simulations, as well as postprocessing.  |
| utilsVTK.py          | General functions that are called by numerous functions.                                       |


## Inputs
`simulation_inputs.csv` is the file that contains all the geometric descriptors for the models you wish to generate. Modify individual parameter (such as total number of models, edge size, folder names, etc). Some important details
1. Only one of phase_1, phase_2, or phase_3 can be true. 
	* If you wish to incorporate variations in geometric features for two of these simultaneously, you can change the baseline values for each separate feature (such as the diameters, etc). The code will generate the geometric features in accordance to the list of values you provide.
2. Ensure that the total number of geometries is the squared value of the number of variations in features you include for each geometric feature.
	* If you want to create 16 geometries, than you should have 4 values for angles, positions, and lengths for each of the rows listed. If you want these parameters to be identical for each geometry you build, then simply repeat the geometric value 4 times.
3. Models will be built in a parametric sweep; i.e. a grid search of the geometric features you prescribed. This pipeline is intended for two-dimensional (2D) variations in features. 
	* Depending on the feature you wish to study, you should only change that specific list of features. For example, if you wish to study changes in LAD/LCx angle, and LMCA length, only change those two parameters. The rest should be constant.
	* Note that you have the ability to vary numerous features at once (more than two at a time) but do so with caution, as
		* The pipeline is designed to change two features at a time (namely angles and positions, diameter of the LAD and LCx, tortuosity of the LAD and LCx, or plaque in the LAD and LCx).
		* This order of parametric sweep is described in **StartModelingPipeline.py**, in the functions `ObtainAnglePositionsDiameters() ObtainTortuosityFeatures(), and ObtainPlaqueTopology()`. Change these to suit your needs.
4. Plaque modeling is known to fail most frequently during the modeling phase. 
	* It is possible that if you are repeatedly seeing models not successfully build, you may need to adjust the geometric description for the plaque itself (slightly change the stenosis or length).
		* Unfortunately, there are no known solutions for this at this time. This is thought to be a modeling issue related to overconstraint of the segmentations needed to build some plaque topologies, that may be topologically intractable with the modeling method implemented in SV.
5. The data folder contains the inlet flow profile (inlet.flow), and the intramyocardial pressure profile (plv.dat)


## Outputs
All geometric models will be stored within the model folder named by the variable `model_folder_name` in the `simulation_inputs.csv` file (line 14).
* This main folder will contain subfolders called Geometry_#, which correspond to each of the geometries in the parametric sweep features
* These Geometry_# subfolders will contain folders with paths, segmentations, models, and meshes. There will also be the necessary presimulation files, and a measurements folder with some information about the geometric description for each model.
* Meshing can take between 10 minutes to nearly 1 hour depending on the edge size. This is a consequence of the radius-based meshing algorithm used. This can be changed if necessary, at the detriment of the surface mesh quality.


Last revised by Arnav Garcha, March 8 2024.