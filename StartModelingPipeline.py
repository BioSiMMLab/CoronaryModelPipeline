import os
import sys
import csv
import time
import shutil
import subprocess
import numpy as np
from collections import OrderedDict

''' Sequence of operations
0. Perform a baseline change in segment lengths and angles
1. Change the angles
2. Shift the branch locations
3. Impose tortuosity
3. Remove unncessary branches
4. Evenly space centerline points
5. Re-assign radii values for the start and end of each branch
6. Start modeling pipeline, where segmentations, models, and meshes are created. Plaque is imposed in this stage
'''

############################################################ Set path to packages and data locally
path_root = str(os.path.dirname(os.path.abspath(__file__)))
path_to_data = path_root + '/Data/'
path_to_BCs  = path_root + '/CoronaryBC/'

for path_names in [path_root, path_to_data, path_to_BCs]:
    sys.path.append(path_names)

############################################################ Import scripts to generate models
import ModifyCenterline as mc
import CreateVessel as cv
import utilsVTK as utils
import InputSVSolver as infile
import WriteCoronaryLPN as LPN
import SBatchGen as SBJ
import AllometricScaling as ascale


############################################################ Constant values, preconditions for each simulation phase
sim_inputs = []  # to store booleans, integers, or strings (parameter values)

'''
Users should change this input file name if they changed the name of this file, or use different files to store different simulation conditions
'''

simulation_description_filename = 'simulation_inputs.csv'


# read in the file geometric_variables which contains all of the ncessary simulation details
with open(path_root + '/' + simulation_description_filename, 'r') as file:
    for i, line in enumerate(file):

        if i == 0:
            continue

        parts = line.strip().split(',')                     # Split the line by comma
        entry = parts[1].strip()                            # Identify the type of each entry in the second column

        if entry.lower() in ('true', 'false'):
            sim_inputs.append(bool(entry.lower() == 'true'))   # If the entry is a boolean
        elif entry.isdigit():
            sim_inputs.append(int(entry))                      # If the entry is an number
        else:
            sim_inputs.append(str(entry))                      # Otherwise, treat it as a string
        

# Print the data for verification
print("simulation inputs:", sim_inputs)


phase_1 = sim_inputs[0]
phase_2 = sim_inputs[1]
phase_3 = sim_inputs[2]
phase_5 = sim_inputs[3]

allo_scale = sim_inputs[4]         # allometrically scale the flows
non_circular_diams = sim_inputs[5] # phase 3, eccentricity, specify to make the lumen assymetric

num_geoms = sim_inputs[6]
num_cycle = sim_inputs[7]
num_nodes = sim_inputs[8]
num_cores = sim_inputs[9]
m_size = np.sqrt(num_geoms)         # square of the total number of geometries to generate.

skip_nums = sim_inputs[10]
if isinstance(skip_nums, str): 
    skip_nums = [int(num) for num in sim_inputs[10].split(':')]
    print("SKIP NUMS")
else:
    skip_nums = [skip_nums]


population_sex = sim_inputs[11]




simulation_folder = "/" + sim_inputs[12] + "/"                      # file save directory
save_dir_path = path_root + simulation_folder                       # file save directory
utils.CreateFolder(save_dir_path, sim_folder = simulation_folder)


shutil.copyfile(path_root + '/' + simulation_description_filename, path_root + simulation_folder + simulation_description_filename)



point_spacing = 0.065               # point spacing for even spacing
edge_size = sim_inputs[-2]          # stores the desired starting mesh edge size in a local variable. This can be "auto" which the script will automatically compute, or a user-defined number
if edge_size != 'auto':
    edge_size = float(edge_size)


msize = np.float(sim_inputs[-1])    # stores a model remeshing edge size in a local variable. This should be smaller than the smallest edge size to best resolve the geometry of the model
model_name = 'LCA'




############################################################ Functions
def ModelingPipeline(centerline_points, angle_dict, shift_dict, diam_dict, phase_2, phase_3, num, save_path, 
                     inflection=None, len_curve=None, amplitude=None, eccentric_radius=False, phase_5_params = None):
    
    np.set_printoptions(precision=3)
    print('Start model generation')
    start_time = time.time()

    centerline_prev0 = centerline_points



    
    # perform a baseline shift, with downstream branches first to avoid errors later on in the process
    baseline_shift = OrderedDict([['9', 3.55 * 2/3], ('8', 3.55), ('11',4.13 * 2/3), ('10', 4.13)])
    for i in baseline_shift:
        centerline_shifted0 = mc.ShiftBranch(int(i), baseline_shift, centerline_prev0, maintain_angle = True)
        centerline_prev0 = centerline_shifted0

    # Align the bifurcation points to be straight with each bifurcated branch
    centerline_prev1 = mc.AlignBifurcationPoints(centerline_prev0)


    ################################################# Centerline Modification, shift bifurcation and angles
    for i in angle_dict:
        centerline_angle = mc.AngleManipulate(int(i), angle_dict[i], centerline_prev1, 'side_branch')
        centerline_prev1 = centerline_angle

    centerline_prev2 = centerline_angle

    for i in shift_dict:
        centerline_shifted = mc.ShiftBranch(int(i), shift_dict, centerline_prev2, maintain_angle = True, 
                                        sphere_check=True, LAD_LCx_a = angle_dict['5'], diams = diam_dict)
        if np.isin('python', centerline_shifted):
            return 'python'
        centerline_prev2 = centerline_shifted


    if phase_3:
        tot_tortuosity_score = 0
        lad_tortuosity_score = 0
        lcx_tortuosity_score = 0
        angles_for_segments = []

        for i in inflection:
            branchID = int(i)
            n_inflection = inflection[i]
            curve_fraction = len_curve[i]
            amp_fraction = amplitude[i]
            centerline_curved, angle_sine = mc.Tortuosity(branchID, centerline_prev2, 
                                                          n_inflection, curve_fraction, amp_fraction)
            angles_for_segments.append(angle_sine)
            centerline_prev2 = centerline_curved


        ################################## calculate tortuosity score
        # M. F. Eleid et al., “Coronary Artery Tortuosity in Spontaneous Coronary Artery Dissection,” 
        # Circulation: Cardiovascular Interventions, vol. 7, no. 5, pp. 656–662, Oct. 2014, 
        # doi: 10.1161/CIRCINTERVENTIONS.114.001676.

        # check the total tortuosity score. divide the angles and inflection numbers based on the lad and lcx
        lad_inflections = inflection['2'] + inflection['3']
        lad_angles = [angles_for_segments[0], angles_for_segments[1]]
        print(lad_inflections, lad_angles)

        if lad_inflections == 0:
            lad_tortuosity_score = 0
        elif lad_inflections >= 3:
            if all(value >= 45 for value in lad_angles) and all(value < 90 for value in lad_angles):
                lad_tortuosity_score = 1
            if all(value >= 90 for value in lad_angles) and all(value < 180 for value in lad_angles):
                lad_tortuosity_score = 2
            else:
                lad_tortuosity_score = 1
        else:
            lad_tortuosity_score = 1

        lcx_inflections = inflection['5'] + inflection['6']
        lcx_angles = [angles_for_segments[2], angles_for_segments[3]]
        print(lcx_inflections, lcx_angles)

        if lcx_inflections == 0:
            lcx_tortuosity_score = 0
        elif lcx_inflections >= 3:
            if all(value >= 45 for value in lcx_angles) and all(value < 90 for value in lcx_angles):
                lcx_tortuosity_score = 1
            if all(value >= 90 for value in lcx_angles) and all(value < 180 for value in lcx_angles):
                lcx_tortuosity_score = 2
            else:
                lcx_tortuosity_score = 1
        else:
            lcx_tortuosity_score = 1

        tot_tortuosity_score = lcx_tortuosity_score + lad_tortuosity_score

        print('total tortuosity score:', tot_tortuosity_score)
        print('LAD tortuosity score:', lad_tortuosity_score)
        print('LCX tortuosity score:', lcx_tortuosity_score)
    

    centerline_RB = mc.RemoveBranches([0,12,13,14,15,16,17,18], centerline_prev2, 4)


    centerline_branchshort0 = mc.ChangeBranchLength(4, sim_inputs[24], centerline_RB)
    centerline_branchshort1 = mc.ChangeBranchLength(7, sim_inputs[25], centerline_branchshort0)
    centerline_branchshort2 = mc.ChangeBranchLength(8, sim_inputs[26], centerline_branchshort1)
    centerline_branchshort3 = mc.ChangeBranchLength(9, sim_inputs[27], centerline_branchshort2)
    centerline_branchshort4 = mc.ChangeBranchLength(10, sim_inputs[28], centerline_branchshort3)
    centerline_branchshort5 = mc.ChangeBranchLength(11, sim_inputs[29], centerline_branchshort4)
    
    
    Bshort_branch_lengths = mc.BranchLengthCalculator(centerline_branchshort5)
    print('branch length after shortening, before evenly spacing')
    print(Bshort_branch_lengths)


    vectors, origin_points_unused = mc.AngleCalculator(centerline_branchshort5)
    print('angles after modification')
    print(vectors[np.r_[1:4,5:7],-1])


    measurements_path = save_path + 'measurements/'
    utils.CreateFolder(measurements_path)

    file_out_name = 'centerlines-modified.vtp'
    output_file_path = measurements_path + file_out_name
    modified_VTP = utils.CreateNewVTP(centerline_branchshort5, output_file_path, True)


    
    ############################################################## Evenly space the centerline points

    centerline_spaced, radii, radii_mb = mc.EvenSpacing(centerline_branchshort5, 
                                                        point_spacing, diam_dict, [8,9,10,11], phase3 = phase_3)
    

    branch_11_end = centerline_spaced[np.in1d(centerline_spaced[:, 3], 11)][-1,:]
    branch_10_end = centerline_spaced[np.in1d(centerline_spaced[:, 3], 10)][-1,:]


    space_branch_lengths = mc.BranchLengthCalculator(centerline_spaced, True)
    print('evenly spaced branch lengths')
    print(space_branch_lengths)

    radii_fixed = mc.CorrectRadius(radii, radii_mb, space_branch_lengths)
    print('radii values, fixed')
    print(radii_fixed)


    np.savetxt(measurements_path + 'radii_mainbranch.csv', radii_mb, delimiter=',')
    np.savetxt(measurements_path + 'radii_final_vals.csv', radii_fixed, delimiter=',')
    np.savetxt(measurements_path + 'angles_postchange.csv', vectors[:,3:], delimiter=',')
    np.savetxt(measurements_path + 'lengths_spaced.csv', space_branch_lengths, delimiter=',')
    np.savetxt(measurements_path + 'lengths_b4_spaced.csv', Bshort_branch_lengths, delimiter=',')
    
    if phase_3:
        np.savetxt(measurements_path + 'total_tort.csv', np.array([tot_tortuosity_score, lad_tortuosity_score, 
                                                                   lcx_tortuosity_score]), delimiter=',')


    file_out_name = 'centerlines-spaced.vtp'
    output_file_path = measurements_path + file_out_name
    new_VTP = utils.CreateNewVTP(centerline_spaced, output_file_path, True)

    print('branch lengths after spacing')
    print(mc.BranchLengthCalculator(centerline_spaced, True))


    ############################################################## Generate 3D model and input files 
    print('Starting 3D modeling pipeline ...')


    if not phase_5:
        err_check = cv.GenerateFinalModel(centerline_spaced, radii_fixed, radii_mb, save_path, num, model_name, phase_2, 
                                          msize, edge = edge_size, bl = True, eccentricity = eccentric_radius,
                                          radius_meshing = True, phase_5 = phase_5_params)
    else:
        err_check, plaque_burden = cv.GenerateFinalModel(centerline_spaced, radii_fixed, radii_mb, save_path, num, model_name, 
                                                         phase_2, msize, edge = edge_size, bl = True, 
                                                         eccentricity = eccentric_radius, radius_meshing = True, phase_5 = phase_5_params)

    # checking for errors
    if err_check == 'SV':
       return 'SV'

    mesh_path  = save_path + 'mesh/mesh-surfaces/'


    ############################################################## Generate simulation input files
    print('Allometric scaling...')
    co, map_val, hr, cap_scale = ascale.AllometricScaledValues(save_path, population_sex, scaling_flag = allo_scale)
    print('Writing solver.svpre...')
    num_caps, wall_num = infile.GenerateSVPRE(save_path, fmodes=30, num_points=1000, hr=hr)
    print('Writing solver.inp ...')
    infile.GenerateInputSVSolver(save_path, num_caps, wall_num, hr, time_step=0.001, total_periods=num_cycle, increment=10)
    print('Writing cort.dat ...')
    LPN.WriteCortDat(mesh_path, save_path, 1000, num_caps, co, map_val, cap_scale, pop_sex=population_sex)
    print('Generating cycle markers ...')
    infile.DetermineCycleTimes(save_path, hr, time_step=0.001, total_periods=num_cycle)


    # Execute svpre
    os.chdir(save_path)
    subprocess.call("/usr/local/sv/svsolver/2022-07-22/bin/svpre sim.svpre", shell=True)
    os.chdir(save_dir_path)


    # generate HPC simulation files
    SBJ.RunMPI(save_path,num_nodes,num_cores,num,hr, time_step=0.001,total_periods=num_cycle,increment=10)



    print("Total modeling time: {0:.2f} seconds".format(time.time() - start_time))
    print('--------Done--------')
    



    if phase_3 and not phase_5:
        return [vectors[np.r_[1:4,5:7],-1], Bshort_branch_lengths[:,-1], space_branch_lengths[:,-1], radii_fixed, 
                tot_tortuosity_score, lad_tortuosity_score, lcx_tortuosity_score]
    elif phase_3 and phase_5:
        return [vectors[np.r_[1:4,5:7],-1], Bshort_branch_lengths[:,-1], space_branch_lengths[:,-1], radii_fixed, 
                tot_tortuosity_score, lad_tortuosity_score, lcx_tortuosity_score, plaque_burden]
    elif phase_5 and not phase_3:
        # adding 3 empty variables to simplify the np.savetxt operation in the primary loop to only have phase_3,5
        return [vectors[np.r_[1:4,5:7],-1], Bshort_branch_lengths[:,-1], space_branch_lengths[:,-1], radii_fixed,
                0, 0, 0, plaque_burden]
    else:
        print('not plaque or tort ')
        return [vectors[np.r_[1:4,5:7],-1], Bshort_branch_lengths[:,-1], space_branch_lengths[:,-1], radii_fixed]





def ConvertDictToList(geom_dict, geom_list, vals_list): 
    
    for keys in geom_dict.keys():
        vals_list.append([d[keys] for d in geom_list])

    return




def WriteFormerDict(vals, vals_dict, dict_save_path):
    j = 0
    for values, dict_keys in zip([vals], [vals_dict]):
        with open(dict_save_path, 'w') as f:
            w = csv.writer(f)
            for key in dict_keys[0].keys():
                w.writerow([str(key)])
            w.writerows(values)
        j += 1

    return




def CreateMapping():
    relation = {}
    for i in range(num_geoms):
        row = i // m_size
        col = i % m_size
        relation[str(i+1)] = [row, col]

    return relation

relation = CreateMapping()





def NumToCellMapping(num):
    
    row_col = relation[str(int(num))]
    row = row_col[0]
    col = row_col[1]
    
    return row, col





def ObtainAnglePositionsDiameters():
    '''

    This function is responible for taking the index of the current geometry, and output the corresponding
    angles, diameters, and branch segment lengths.

    To create combinatorial changes in segment values, you need to
        1. Change the values of the given parametric feature you wish to adjust in the simulation_inputs.csv file
            a. The values you list are the unique variations in that feature you wish to change. For example, if you want to 
            study four different bifurcation angles at the LAD/LCx, you would include four different numbers separated by colons
            b. If you wish to generate a nxn combination of geometries, ensure that the features you prescribe is equal to n 
        2. Keep in mind the order of the for loop structure
            a. Positions are shifted first, then angles. Change the order if you desire.
            b. LCx diameters are changed first, then LAD ones. Change the order if you desire.
    '''
    
    diams_dicts = []
    shift_dicts = []
    angle_dicts = []

    angle_ladlcx = [float(num) for num in sim_inputs[14].split(':')]
    angle_ladd1  = [float(num) for num in sim_inputs[15].split(':')]
    angle_ladd2  = [float(num) for num in sim_inputs[16].split(':')]
    angle_lcxom1 = [float(num) for num in sim_inputs[17].split(':')]
    angle_lcxom2 = [float(num) for num in sim_inputs[18].split(':')]
    
    length_lmca     = [float(num) for num in sim_inputs[19].split(':')]
    length_prox_lad = [float(num) for num in sim_inputs[20].split(':')]
    length_med_lad  = [float(num) for num in sim_inputs[21].split(':')]
    length_prox_lcx = [float(num) for num in sim_inputs[22].split(':')]
    length_med_lcx  = [float(num) for num in sim_inputs[23].split(':')]

    diameter_lad = [float(num) for num in sim_inputs[30].split(':')]
    diameter_lcx = [float(num) for num in sim_inputs[31].split(':')]


    for diam_lad in diameter_lad:                                               # external loop defines rows
        for diam_lcx in diameter_lcx:                                           # external loop defines columns
            diams_dicts.append(OrderedDict([('1', diam_lad), ('5', diam_lcx)]))



    for a_ladlcx, a_ladd1, a_ladd2, a_lcxom1, a_lcxom2 in zip(angle_ladlcx, angle_ladd1, angle_ladd2, angle_lcxom1, angle_lcxom2):
        for s_lmca, s_proxlad, s_medlad, s_proxlcx, s_medlcx in zip(length_lmca, length_prox_lad, length_med_lad, length_prox_lcx, length_med_lcx):
            shift_dicts.append(OrderedDict([('5', s_lmca), ('8', s_proxlad), ('9', s_medlad),  ('10', s_proxlcx), ('11', s_medlcx)]))
            angle_dicts.append(OrderedDict([('5', a_ladlcx), ('8', a_ladd1), ('9', a_ladd2), ('10', a_lcxom1), ('11', a_lcxom2)]))

    if len(angle_dicts) != num_geoms:
        print('length of angle list:', len(diams_dicts), " total number of requested geometries:", num_geoms)
        raise ValueError('length the angle list is not appropriate. please ensure that the total parameters in the \
                         simulation_inputs.csv file is a multiple of the number of geometries you wish to build, or \
                         is equal to the expected combinations of angles.')
    elif len(shift_dicts) != num_geoms:
        print('length of branch segment length list:', len(diams_dicts), " total number of requested geometries:", num_geoms)
        raise ValueError('length the position list is not appropriate. please ensure that the total values in the \
                         simulation_inputs.csv file is a multiple of the number of geometries you wish to build, or \
                         is equal to the expected combinations of lengths.')
    elif len(diams_dicts) != num_geoms:
        print('length of diameter list:', len(diams_dicts), " total number of requested geometries:", num_geoms)
        raise ValueError('length the diameter list is not appropriate. please ensure that the total values in the \
                         simulation_inputs.csv file is a multiple of the number of geometries you wish to build, or \
                         is equal to the expected combinations of diameters.')


    return shift_dicts, angle_dicts, diams_dicts





def ObtainTortuosityFeatures():
    '''
    This function is responible for taking the index of the current geometry, and output the corresponding
    angles, diameters, and branch segment lengths.

    To create combinatorial changes in segment values, you need to
        1. Change the values of the given parametric feature you wish to adjust in the simulation_inputs.csv file
            a. The values you list are the unique variations in that feature you wish to change. For example, if you want to 
            study four different tortuosity LAD branch, you would include four different numbers separated by colons
            b. If you wish to generate a nxn combination of geometries, ensure that the features you prescribe is equal to n 
        2. Keep the loop structure in mind.
            a. LCx tortuosity is varied first, then LAD. Change if desired
    '''

    inflections_list = []
    length_curves_list = []
    amplitudes_list = []

    s2_inflections = [int(num) for num in sim_inputs[33].split(':')]
    s3_inflections = [int(num) for num in sim_inputs[34].split(':')]
    segment_inflections_lad = list(zip(s2_inflections, s3_inflections))

    s5_inflections = [int(num) for num in sim_inputs[35].split(':')]
    s6_inflections = [int(num) for num in sim_inputs[36].split(':')]
    segment_inflections_lcx = list(zip(s5_inflections, s6_inflections))


    s2_len_curves = [float(num) for num in sim_inputs[37].split(':')]
    s3_len_curves = [float(num) for num in sim_inputs[38].split(':')]
    segment_len_curves_lad = list(zip(s2_len_curves, s3_len_curves))
    s5_len_curves = [float(num) for num in sim_inputs[39].split(':')]
    s6_len_curves = [float(num) for num in sim_inputs[40].split(':')]
    segment_len_curves_lcx = list(zip(s5_len_curves, s6_len_curves))


    s2_amplitudes = [float(num) for num in sim_inputs[41].split(':')]
    s3_amplitudes = [float(num) for num in sim_inputs[42].split(':')]
    segment_amplitudes_lad = list(zip(s2_amplitudes, s3_amplitudes))
    s5_amplitudes = [float(num) for num in sim_inputs[43].split(':')]
    s6_amplitudes = [float(num) for num in sim_inputs[44].split(':')]
    segment_amplitudes_lcx = list(zip(s5_amplitudes, s6_amplitudes))

    # store the tortuosity values in a list
    for infl_lad, len_lad, ampl_lad in zip(segment_inflections_lad, segment_len_curves_lad, segment_amplitudes_lad):
        for infl_lcx, len_lcx, ampl_lcx in zip(segment_inflections_lcx, segment_len_curves_lcx, segment_amplitudes_lcx):
            inflections_list.append(OrderedDict([('2', infl_lad[0]), ('3', infl_lad[1]),  ('5', infl_lcx[0]), ('6', infl_lcx[1])]))
            length_curves_list.append(OrderedDict([('2', len_lad[0]), ('3', len_lad[1]),  ('5', len_lcx[0]), ('6', len_lcx[1])]))
            amplitudes_list.append(OrderedDict([('2', ampl_lad[0]), ('3', ampl_lad[1]),  ('5', ampl_lcx[0]), ('6', ampl_lcx[1])]))


    return inflections_list, length_curves_list, amplitudes_list







def ObtainPlaqueTopology():
    '''
    1. branch where place is placed
    2. plaque length
    3. lumenal narrowing (stenosis)
    '''

    plaque_loc_list = []
    plaque_len_list = []
    plaque_wid_list = []
    narrowing_list  = []

    # 1. branch where plaque is placed
    lad_plaque = [int(sim_inputs[46]) for num in range(num_geoms)]
    lcx_plaque = [int(sim_inputs[49]) for num in range(num_geoms)]


    # 2. width (as a percentage of the total lumen) and length (in units of cm)
    plaque_len_lad = [float(num) for num in sim_inputs[48].split(':')]
    plaque_len_lcx = [float(num) for num in sim_inputs[51].split(':')]


    # 3. luminal narrowing, or stenosis degree/severity
    narrowing_lad  = [float(num) for num in sim_inputs[47].split(':')]
    narrowing_lcx  = [float(num) for num in sim_inputs[50].split(':')]



    # these are hard incoded
    for lad, lcx in zip(lad_plaque, lcx_plaque):
        plaque_loc_list.append(OrderedDict([('1', lad), ('5', lcx)]))




    # LAD OR LCX ONLY loop through the changing parameters
    for sten_lad, sten_lcx in zip(narrowing_lad, narrowing_lcx):                    # external loop defines rows
        for len_lad, len_lcx in zip(plaque_len_lad, plaque_len_lcx):                # internal loop defines columns
            narrowing_list.append(OrderedDict([('1', sten_lad), ('5', sten_lcx)]))
            plaque_len_list.append(OrderedDict([('1', len_lad), ('5', len_lcx)]))



    # loop through 1 value for as many simulations as there are
    for width_lad in narrowing_lad:
        for width_lcx in narrowing_lcx:
            if (width_lad + 0.25) > 0.99:
                width_acc_lad = 0.7
            else:
                width_acc_lad = width_lad + 0.25    # +25% width makes a smooth plaque

            if (width_lcx + 0.25) > 0.99:
                width_acc_lcx = 0.7
            else:
                width_acc_lcx = width_lcx + 0.25    # +25% width makes a smooth plaque


            plaque_wid_list.append(OrderedDict([('1', width_acc_lad), ('5', width_acc_lcx)]))  

    
    # concatenate all of the previous lists into one list, for ease of reading and integration into existing functions
    plaque_topology_list = [plaque_loc_list, plaque_len_list, plaque_wid_list, narrowing_list]
    
    return plaque_topology_list









############################################################ Perform model generation if this script is executed
if __name__ == '__main__':
    np.set_printoptions(precision=3)

    # Extract original data
    file_name = 'centerlines-OG.vtp'
    input_file = path_to_data + file_name

    centerline_points = mc.ReadCenterlineVTP(input_file)

    # Set constants

    angle_dict, shift_dict, diam_dict = [], [], []
    angle_acc, length_prespace, length_postspace, diam_acc = [], [], [], []

    i, j = 1, 0
    fail_counter = 0
    retry = False



    shift_main, angle_main, diams_main = ObtainAnglePositionsDiameters()         # obtain angles, shifted branch positions (corresponds to branch segment length), and diameters


    if phase_3:
        inflection, len_curve, amplitudes = ObtainTortuosityFeatures()

        tort_scores = []
        lad_tort_scores = []
        lcx_tort_scores = []

    if phase_1 + phase_2 + phase_3 != 1:
        raise ValueError('please set only 1 of phase_1, phase_2, or phase_3 as True.')
    elif phase_1 + phase_2 + phase_3 == 0:
        raise ValueError('please set phase_1, phase_2, or phase_3 as True or False')


    if phase_5:
        plaque_map_all_geoms = ObtainPlaqueTopology()
        print('all plaque data')
        print(plaque_map_all_geoms)

        plaque_location = []
        plaque_stenosis = []
        plaque_lengths  = []
        plaque_widths   = []
        plaque_burden   = []


    ################################################# Model generation loop

    while j < (num_geoms):
        if retry:
            print('------------------------ FAILED GEOMETRY: {0} ------------------------'.format(i))
            fail_counter += 1
        else:
            print('------------------------ STARTING GEOMETRY: {0} ------------------------'.format(i))
        if fail_counter == 1:
            raise Exception('failed generating this geometry. Please double check the inputs and where the pipeline failed to diagnose the issue.')


        angle = angle_main[j]
        shift = shift_main[j]
        diam  = diams_main[j]

        print('angle', angle)
        print('shift', shift)
        print('diam', diam)


        if phase_3:
            n_inf = inflection[j]
            len_c = len_curve[j]
            amp_c = amplitudes[j]
            print('number of inflections   ', n_inf)
            print('fractional length of curvature   ', len_c)
            print('amplitude of curved segment   ', amp_c)
        else:
            n_inf = None
            len_c = None
            amp_c = None

        if phase_5:
            plaque_current = [sublist[j] for sublist in plaque_map_all_geoms]       # use j since it starts at 0
            print('plaque curr', i)
            print(plaque_current)
        else:
            plaque_current = None




        if i not in skip_nums:
            save_sim_path = save_dir_path + 'Geometry_' + str(i) + '/'
            utils.CreateFolder(save_sim_path)

            error_check = ModelingPipeline(centerline_points, angle, shift, diam, phase_2, phase_3, i, save_sim_path, n_inf, 
                                           len_c, amp_c, eccentric_radius = non_circular_diams, phase_5_params = plaque_current)

            if np.any(error_check == 'SV'):
                print('failed due to python SV')
                retry = True
            elif np.any(error_check == 'python'):
                print('failed due to some python geometric modification errors. please double check inputs')
                retry = True
            elif np.any(error_check == 'python_sc'):
                print('failed due to python smoothing')
                retry = True                
            else:
                # Successfully generated the model
                angle_dict.append(angle)
                shift_dict.append(shift)
                diam_dict.append(diam)


                angle_acc.append(error_check[0])
                length_prespace.append(error_check[1])
                length_postspace.append(error_check[2])
                diam_acc.append(error_check[3])

                if phase_3:
                    tort_scores.append(error_check[4])
                    lad_tort_scores.append(error_check[5])
                    lcx_tort_scores.append(error_check[6])
                
                if phase_5:
                    plaque_location.append(plaque_current[0])
                    plaque_stenosis.append(plaque_current[3])
                    plaque_lengths.append(plaque_current[1])
                    plaque_widths.append(plaque_current[2])
                    plaque_burden.append(error_check[7])        # error_check[7] is used to not interfere with tortuosity

                retry = False

                i += 1
                j += 1
            
        else:
            # skip the index provided. still increment the number of models
            print('skipping: ', i, ' because it is in the list: ', skip_nums)

            i += 1
            j += 1




    SBJ.GenerateSBatchJob(save_dir_path)


    ## Save the geometric outputs
    np.savetxt(save_dir_path + 'angles_actual.csv', angle_acc, delimiter=',')
    np.savetxt(save_dir_path + 'lengths_prespace.csv', length_prespace, delimiter=',')
    np.savetxt(save_dir_path + 'lengths_postspace.csv', length_postspace, delimiter=',')
    
    if phase_3:
        np.savetxt(save_dir_path + 'tortuosity_scores.csv', tort_scores, delimiter=',')
        np.savetxt(save_dir_path + 'lad_tortuosity_scores.csv', lad_tort_scores, delimiter=',')
        np.savetxt(save_dir_path + 'lcx_tortuosity_scores.csv', lcx_tort_scores, delimiter=',')


    with open(save_dir_path + 'diams_actual.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for arr in diam_acc:
            writer.writerows(arr)
    
    
    angle_vals, shift_vals, diam_vals = [], [], []

    ConvertDictToList(angle, angle_dict, angle_vals)
    ConvertDictToList(shift, shift_dict, shift_vals)
    ConvertDictToList(diam, diam_dict, diam_vals)


    WriteFormerDict(angle_vals, angle_dict, save_dir_path + 'angle_dict.csv')
    WriteFormerDict(shift_vals, shift_dict, save_dir_path + 'shift_dict.csv')
    WriteFormerDict(diam_vals, diam_dict, save_dir_path + 'diam_dict.csv')



    if phase_5:
        print('generating plaque .csv files')
        plaque_location_vals, plaque_stenosis_vals, plaque_lengths_vals, plaque_widths_vals = [], [], [], []

        ConvertDictToList(plaque_current[0], plaque_location, plaque_location_vals)
        ConvertDictToList(plaque_current[3], plaque_stenosis, plaque_stenosis_vals)
        ConvertDictToList(plaque_current[1], plaque_lengths, plaque_lengths_vals)
        ConvertDictToList(plaque_current[2], plaque_widths, plaque_widths_vals)


        plaque_burden_flattened = [entry for subarray in plaque_burden for entry in subarray]

        headers_burden = ['Geom_num', 'branch', 'segmentation_index', 'plaque_burden']

        with open(save_dir_path + 'plaque_location.csv', "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers_burden) 
            writer.writerows(plaque_burden_flattened)
        
        WriteFormerDict(plaque_location_vals, plaque_location, save_dir_path + 'plaque_location.csv')
        WriteFormerDict(plaque_stenosis_vals, plaque_stenosis, save_dir_path + 'plaque_stenosis.csv')
        WriteFormerDict(plaque_lengths_vals, plaque_lengths, save_dir_path + 'plaque_lengths.csv')
        WriteFormerDict(plaque_widths_vals, plaque_widths, save_dir_path + 'plaque_width.csv')


