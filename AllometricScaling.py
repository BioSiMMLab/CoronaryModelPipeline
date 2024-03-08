import os
import vtk
import shutil
import numpy as np


########################################################### Constants
path_root = str(os.path.dirname(os.path.abspath(__file__)))
path_to_data = path_root + '/Data/'
path_to_BCs  = path_root + '/CoronaryBC/'

cardiac_output_f = 76.667       # [ml/s]
mean_artery_p_f  = 83.87        # [mmHg]
heart_rate_f     = 79.1         # [bpm]
average_volume_f = 1.12006      # [ml]

cardiac_output_m = 98.333       # [ml/s]
mean_artery_p_m  = 88.77        # [mmHg]
heart_rate_m     = 74.3         # [bpm]
average_volume_m = 1.93527      # [ml]

cardiac_exponent = 0.25         # changed from 0.75,
heart_rate_exponent = -0.25

phase1_heart_period    = 0.86   # [s]
phase1_coronary_output = 80.0   # [ml/s]
phase1_mean_pressure   = 93.3   # [mmHg]


coronary_aorta_flow_split = 0.04
left_coronary_flow_split  = 0.7

########################################################### Functions


def AllometricScaledValues(directory_path, population_sex, scaling_flag = True):

    '''
    1. determine the volume of the new mesh
    2. scale the heart rate, cardiac output, and intramyocardial pressure accordingly
    3. pass the required values to the writecoronaryLPN function for calculating resistances
    
    '''
    
    if not scaling_flag:
        print("______ NOT ALLOMETRICALLY SCALING  ______")
        print("using default values from the original, phase 1 file")
        # copy the results to the simulation folder
        shutil.copyfile(path_to_BCs + '/plv.dat', directory_path + '/plv.dat')
        shutil.copyfile(path_to_BCs + '/inlet_sv.flow', directory_path + '/inlet_sv.flow')
        
        return phase1_coronary_output, phase1_mean_pressure, 1.0 / phase1_heart_period * 60.0, 1.0
    

    else:
        # allometrically scale
        print("______  ALLOMETRICALLY SCALING  ______")

        # Load the .vtp file
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(directory_path + '/model/LCA_final.vtp')
        reader.Update()

        # Convert the unstructured grid to polydata
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputConnection(reader.GetOutputPort())
        geometry_filter.Update()
        polydata = geometry_filter.GetOutput()

        # Measure the volume
        mass = vtk.vtkMassProperties()
        mass.SetInputData(polydata)
        mass.Update()
        total_volume = mass.GetVolume()

        print("The volume of the new mesh is: {0} from the {1} population data".format(total_volume, population_sex))

        if population_sex == 'female':
            print('scaling based on female population data')
            scaled_cardiac_output = cardiac_output_f * (total_volume**cardiac_exponent) / (average_volume_f**cardiac_exponent)
            scaled_mean_pressure  = mean_artery_p_f * (total_volume**cardiac_exponent) / (average_volume_f**cardiac_exponent)
            scaled_heart_rate  = heart_rate_f * (total_volume**heart_rate_exponent) / (average_volume_f**heart_rate_exponent)
            capacitance_scale  = (total_volume**cardiac_exponent) / (average_volume_f**cardiac_exponent)
        elif population_sex == 'male':
            print('scaling based on male population data')
            scaled_cardiac_output = cardiac_output_m * (total_volume**cardiac_exponent) / (average_volume_m**cardiac_exponent)
            scaled_mean_pressure  = mean_artery_p_m * (total_volume**cardiac_exponent) / (average_volume_m**cardiac_exponent)
            scaled_heart_rate  = heart_rate_m * (total_volume**heart_rate_exponent) / (average_volume_m**heart_rate_exponent)
            capacitance_scale  = (total_volume**cardiac_exponent) / (average_volume_m**cardiac_exponent) 
        else:
            raise ValueError("pass 'male' or 'female' to the population_sex parameter")
        

        flow_mag_scaling      = np.abs(scaled_cardiac_output / phase1_coronary_output)
        heart_period_scale    = np.abs( (1.0 / scaled_heart_rate * 60.0) / phase1_heart_period)
        intramyocardial_scale = np.abs(scaled_mean_pressure / phase1_mean_pressure)
        scaled_heart_beat_period = round(1.0/scaled_heart_rate*60.0, 2)

        print('scaled cardiac output', scaled_cardiac_output)
        print('scaled mean pressure', scaled_mean_pressure)
        print('scaled heart rate', scaled_heart_rate)
        print('scaled heart beat period', scaled_heart_beat_period)


        ######## 2. Scale the .txt files with flow and intramyocardial pressure, save as new files
        plv_flow_og   = np.loadtxt(path_to_BCs + '/plv.dat')
        inlet_flow_og = np.loadtxt(path_to_BCs + '/inlet_sv.flow')

        inlet_flow_scaled = np.copy(inlet_flow_og)
        plv_flow_scaled = np.copy(plv_flow_og)

        inlet_flow_scaled[:,0] = inlet_flow_og[:,0] * heart_period_scale
        inlet_flow_scaled[-1,0] = round(1.0 / scaled_heart_rate * 60.0,2)
        inlet_flow_scaled[:,1] = inlet_flow_og[:,1] * flow_mag_scaling

        plv_flow_scaled[:,0] = plv_flow_og[:,0] * heart_period_scale
        plv_flow_scaled[-1,0] = round(1.0 / scaled_heart_rate * 60.0,2)
        plv_flow_scaled[:,1] = plv_flow_og[:,1] * intramyocardial_scale

        # save the results to the simulation folder
        np.savetxt(directory_path + '/inlet_sv.flow', inlet_flow_scaled, fmt='%.5f', delimiter=' ')
        np.savetxt(directory_path + '/plv.dat', plv_flow_scaled, fmt='%.5f', delimiter=' ')

        np.savetxt(directory_path + '/measurements/scaled_vals.txt', 
                [scaled_cardiac_output, scaled_mean_pressure, scaled_heart_rate, total_volume], fmt='%.5f', delimiter=' ')


        return scaled_cardiac_output, scaled_mean_pressure, scaled_heart_rate, capacitance_scale
