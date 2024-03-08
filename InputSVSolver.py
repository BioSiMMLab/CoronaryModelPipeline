import os
import re
import sys
import numpy as np
from pathlib import Path

################################################ Add Directory Paths
script_path = Path(os.path.realpath(__file__)).parent
sys.path.append(script_path)
import utilsVTK as utils


## The function below writes sim.svpre and numstart.dat

def GenerateSVPRE(directory_path, fmodes, num_points, hr):
    file_path = directory_path + 'sim.svpre'
    file_name_ls = []

    period = 1.0 / hr * 60.0


    with open(file_path, 'w') as f:
        f.write('mesh_and_adjncy_vtu mesh/mesh-complete.mesh.vtu\n')
        num_surface = 1
        wall_num = 0
        num_caps = []
        wall_name = ''
        for file_name in os.listdir(directory_path + '/mesh/mesh-surfaces'):
            file_name_ls.append(file_name)

            path_to_surface_mesh = 'mesh/mesh-surfaces/' + file_name
            f.writelines('set_surface_id_vtp ' + path_to_surface_mesh + ' ' + str(num_surface) + '\n')
            
            if 'wall' not in file_name:
                if 'cap' in file_name:
                    num_caps.append(num_surface)
            else:
                wall_name = file_name
                wall_num = num_surface
            num_surface += 1

        
        f.write('''fluid_density 1.06
fluid_viscosity 0.04
initial_pressure 0
initial_velocity 0.0001 0.0001 0.0001''')

        for file_name in file_name_ls:
            if 'inlet' in file_name:
                path_to_inlet_mesh = 'mesh/mesh-surfaces/' + file_name
                f.writelines('\nprescribed_velocities_vtp ' + path_to_inlet_mesh)

        f.write('''
bct_analytical_shape parabolic''')
        f.write('\nbct_period ' + str(round(period, 2)))
        f.write('\nbct_point_number ' + str(num_points) + '\n')
        f.write('bct_fourier_mode_number ' + str(fmodes) + '\n')
                
        for file_name in file_name_ls:
            if 'inlet' in file_name:
                path_to_inlet_mesh = 'mesh/mesh-surfaces/' + file_name
                f.writelines('bct_create ' + path_to_inlet_mesh + ' inlet_sv.flow')

        f.write('''
bct_write_dat bct.dat
bct_write_vtp bct.vtp''')

        for file_name in file_name_ls:
            if 'cap' in file_name:
                path_to_cap_mesh = 'mesh/mesh-surfaces/' + file_name
                f.writelines('\npressure_vtp ' + path_to_cap_mesh + ' 0')                
        
        path_to_noslip_wall = 'mesh/mesh-surfaces/' + wall_name
        f.write('\nnoslip_vtp ' + path_to_noslip_wall + '\n')
        f.write('write_geombc geombc.dat.1\n')
        f.write('write_restart restart.0.1')

    return num_caps, wall_num



def GenerateInputSVSolver(directory_path, num_caps, wall_num, hr, time_step, total_periods, increment):
    file_path = directory_path + 'solver.inp'
    
    period = round(1.0 / hr * 60.0, 2) 

    

    with open(file_path, 'w') as f:
        f.write('''Density: 1.06
Viscosity: 0.04''')
        f.write("\n\nNumber of Timesteps: " + str(round(period * total_periods / time_step)))
        f.write("\nTime Step Size: " + str(time_step))
        f.write("\n")

        f.write("\nNumber of Timesteps between Restarts: " + str(increment))
        f.write("\nNumber of Force Surfaces: 1")

                
        f.write("\nSurface ID's for Force Calculation: " + str(int(wall_num)) + '\n')
        f.write('''Force Calculation Method: Velocity Based
Print Average Solution: True
Print Error Indicators: False
        
Time Varying Boundary Conditions From File: True

Step Construction: 0 1 0 1 0 1

Number of COR Surfaces: 6
List of COR Surfaces:''')

        for cap_num in num_caps:
            f.writelines(' ' + str(int(cap_num)))


        f.write('''
COR Values From File: True

Pressure Coupling: Implicit
Number of Coupled Surfaces: 6

Backflow Stabilization Coefficient: 0.2
Residual Control: True
Residual Criteria: 0.001
Minimum Required Iterations: 3
svLS Type: NS
Number of Krylov Vectors per GMRES Sweep: 100
Number of Solves per Left-hand-side Formation: 1
Tolerance on Momentum Equations: 0.001
Tolerance on Continuity Equations: 0.001
Tolerance on svLS NS Solver: 0.001
Maximum Number of Iterations for svLS NS Solver: 10
Maximum Number of Iterations for svLS Momentum Loop: 10
Maximum Number of Iterations for svLS Continuity Loop: 400
Time Integration Rule: Second Order
Time Integration Rho Infinity: 0.5
Flow Advection Form: Convective
Quadrature Rule on Interior: 2
Quadrature Rule on Boundary: 3''')


    file_path_numstart = directory_path + 'numstart.dat'

    with open(file_path_numstart, 'w') as f:
        f.write('{0}'.format(0))

    return None






def DetermineCycleTimes(save_dir, hr, time_step = 0.001, total_periods = 4):
    period = round(1 / hr * 60, 2) 

    print(period, hr)

    final_time = int(round((period * total_periods / time_step)/10)*10)
    start_time = int(round((period * (total_periods-1) / time_step)/10)*10)

    print('final time', final_time)
    print('start time', start_time)

    # Load the data from the .txt file
    flow_data = np.loadtxt(save_dir + '/inlet_sv.flow')

    # Separate the time and flow columns
    time = flow_data[:, 0]
    flow = flow_data[:, 1]

    # Find the index of the minimum and maximum flow values
    max_index = np.argmin(flow)     # since the inlet_sv.flow is inversed, the min is actually the max flow
    min_index = np.argmax(flow)

    # Get the corresponding time values
    min_flow_time = int(round((time[min_index] / time_step + start_time)/10)*10)
    max_flow_time = int(round((time[max_index] / time_step + start_time)/10)*10)

    # Print the results
    print("Time with the lowest flow:", min_flow_time)
    print("Time with the highest flow:", max_flow_time)

    # Load the data from the .txt file
    pressure_data = np.loadtxt(save_dir + '/plv.dat')
    time = pressure_data[:, 0]
    pres = pressure_data[:, 1]

    max_pres_index = np.argmax(pres)
    max_pres_time  = int(round((time[max_pres_index]/time_step + start_time)/10)*10)
    dias_systole_turn = int(round(((final_time - start_time)/3 + start_time)/10)*10)

    print('Time with highest pressure', max_pres_time)
    print('diastole systole switch time', dias_systole_turn)

    times = [start_time, final_time, min_flow_time, max_flow_time, max_pres_time, dias_systole_turn]

    np.savetxt(save_dir + '/measurements/times.csv', times, delimiter=',')

    return
