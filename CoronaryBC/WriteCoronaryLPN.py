import os
import vtk
import glob
import shutil
import numpy as np

########################################################### Constants
path_root = str(os.path.dirname(os.path.abspath(__file__)))



Ca_scale = 0.11
Cim_scale = 0.89

Ra_scale = 0.34
Ra_micro_scale = 0.48
Rv_scale = 0.18

# Model parameters
coronary_aorta_flow_split = 0.04
left_coronary_flow_split  = 0.7
murray_exp_cor = 2.6
LRsplit = 7/3

pconv = 1333


# Phase 1 values
Ccor_val_phase1 = 3.6e-5
cardiac_output_phase1 = 80.0		    # [ml/s]
mean_artery_p_phase1 = 93.3		        # [mmHg]
period_phase1 = 0.86    		        # [s/beat]
HR_phase1 = 1.0 / period_phase1 * 60.0 	# [bpm]


# Phase 2 values, sex based
Ccor_male = 3.2e-5
Ccor_female = 2.9e-5

cardiac_output_f = 76.667       # [ml/s]
mean_artery_p_f  = 83.87        # [mmHg]
heart_rate_f     = 79.1         # [bpm]

cardiac_output_m = 98.333       # [ml/s]
mean_artery_p_m  = 88.77        # [mmHg]
heart_rate_m     = 74.3         # [bpm]

########################################################### Functions

def GetArea(vtp_file):
    # read polydata and compute area
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()
    surface = reader.GetOutputPort()

    surface_mass = vtk.vtkMassProperties()
    surface_mass.SetInputConnection(surface)
    surface_mass.Update()

    area = surface_mass.GetSurfaceArea()
    return area


def ComputeBranchResistance(area, R, murray_exp):
    sum_r = 0
    NoO = len(area)
    r_branch = [0]*NoO

    for i in range(NoO):
        sum_r = sum_r + (np.sqrt(area[i]/np.pi))**murray_exp

    for i in range(NoO):
        r_branch[i] = R*sum_r/((np.sqrt(area[i]/np.pi))**murray_exp)
    
    return r_branch


def ComputeBranchCapacitance(area, C):
    NoO = len(area)
    c_branch = [0]*NoO

    sum_A = sum(area)

    for i in range(NoO):
        c_branch[i] = C*area[i]/sum_A
    
    return c_branch


def WriteCortDat(path_to_files, path_to_sim, num_points, num_caps, Qtotal, pressure, Ccor_scaling, pop_sex):
    cor_l_files = glob.glob(path_to_files + 'cap*')
    area_cor_l = []

    for i in range(len(cor_l_files)):
        area_cor_l.append(GetArea(cor_l_files[i]))
        # print(cor_l_files[i])
        # print('area: ' + str(area_cor_l[i]))
        # print('radius: ' + str(np.sqrt(area_cor_l[i]/np.pi)))
        # print('diameter: ' + str(2*np.sqrt(area_cor_l[i]/np.pi)))

    # if phase2:
    Rtotal_cor = pressure*pconv/Qtotal/coronary_aorta_flow_split
    # else:
    #     Rtotal_cor = mean_artery_p_phase1*pconv/cardiac_output_phase1/coronary_aorta_flow_split

    Rtotal_cor_l = Rtotal_cor * (LRsplit + 1)/LRsplit


    print('Coronary resistance total: {0}, left coronary: {1}'.format(Rtotal_cor, Rtotal_cor_l))

    # if phase2:
    if pop_sex == 'male':
        Ccor_val = Ccor_scaling * Ccor_male
    elif pop_sex == 'female':
        Ccor_val = Ccor_scaling * Ccor_female
    else:
        Ccor_val = Ccor_val_phase1 #3.2e-5
    # else:
    #     Ccor_val = Ccor_val_phase1


    print('scaled capacitance,', Ccor_val, Ccor_scaling, pop_sex)
    np.savetxt(path_to_sim + '/measurements/capacitance.txt', np.array([Ccor_val]), fmt='%.9f')


    Rcor_l = ComputeBranchResistance(area_cor_l, Rtotal_cor_l, murray_exp_cor)
    Ccor_l = ComputeBranchCapacitance(area_cor_l, Ccor_val)

    Ra_l  = Ra_scale*np.array(Rcor_l)
    Ram_l = Ra_micro_scale*np.array(Rcor_l)
    Rv_l  = Rv_scale*np.array(Rcor_l)

    Cim_l = Cim_scale*np.array(Ccor_l)
    Ca_l = Ca_scale*np.array(Ccor_l)


    # shutil.copyfile(path_root + '/plv.dat', path_to_sim + 'plv.dat')
    # shutil.copyfile(path_root + '/inlet_sv.flow', path_to_sim + 'inlet_sv.flow')


    with open(path_to_sim + 'cort.dat','w') as coronaryModel:
        coronaryModel.write('{0}'.format(num_points))

        for i in range(len(num_caps)):
            q0 = Ra_l[i] + Ram_l[i] + Rv_l[i]
            q1 = Ra_l[i]*Ca_l[i]*(Ram_l[i] + Rv_l[i]) + Cim_l[i]*(Ra_l[i] + Ram_l[i])*Rv_l[i]
            q2 = Ca_l[i]*Cim_l[i]*Ra_l[i]*Ram_l[i]*Rv_l[i]
            p0 = 1
            p1 = Ram_l[i]*Ca_l[i] + Rv_l[i]*(Ca_l[i] + Cim_l[i])
            p2 = Ca_l[i]*Cim_l[i]*Ram_l[i]*Rv_l[i]
            b0 = 0
            b1 = Cim_l[i]*Rv_l[i]
            b2 = 0

            coronaryModel.write('\n{0}\n'.format(num_points))

            for q in [q0, q1, q2]:
                coronaryModel.write('{0:.2f}\n'.format(q))

            coronaryModel.write('{0}\n'.format(p0))
            coronaryModel.write('{0:.7f}\n'.format(p1))
            coronaryModel.write('{0:.7f}\n'.format(p2))

            coronaryModel.write('{0}\n'.format(b0))
            coronaryModel.write('{0:.7f}\n'.format(b1))
            coronaryModel.write('{0}\n'.format(b2))

            coronaryModel.write('{0:.1f}\n'.format(0.0))
            coronaryModel.write('{0:.1f}'.format(100.0))

            # Use the new plv.dat file generated for each simulation, which is scaled allometrically
            with open(path_to_sim + '/plv.dat', 'r') as input_file:
                # Read the contents of the file into a list of strings
                lines = input_file.readlines()

            for line in lines:
                # Split each line into two numbers
                num1, num2 = line.split()
                # Write the two numbers separated by a space into the output file
                coronaryModel.write('\n' + num1 + ' ' + num2)
        
    return None




def WriteModelFiLe(path_to_files, path_to_genBC, path_to_sim, Qtotal, pressure, hr, phase2, pop_sex = None):
    cor_l_files = glob.glob(path_to_files + 'cap*')

    area_cor_l = []

    for i in range(len(cor_l_files)):
        area_cor_l.append(GetArea(cor_l_files[i]))

    if pop_sex == 'male':
        Ccor_scaling = Qtotal/cardiac_output_m
    elif pop_sex == 'female':
        Ccor_scaling = cardiac_output_f/Qtotal
    else:
        Ccor_scaling = 1

    print('scaled capacitance, ', Ccor * Ccor_scaling)
    np.savetxt(path_to_sim + '/measurements/edgesize.txt', np.array([Ccor * Ccor_scaling]), fmt='%.5f')


    Rtotal_cor = pressure*pconv/Qtotal/coronary_aorta_flow_split
    Rtotal_cor_l = Rtotal_cor * (LRsplit + 1)/LRsplit

    print('Coronary resistance total: {0}, left coronary: {1}'.format(Rtotal_cor, Rtotal_cor_l))

    Rcor_l = ComputeBranchResistance(area_cor_l, Rtotal_cor_l, murray_exp_cor)
    Ccor_l = ComputeBranchCapacitance(area_cor_l, Ccor * Ccor_scaling)

    Ra_l = Ra_scale*np.array(Rcor_l)/pconv
    Ram_l = Ra_micro_scale*np.array(Rcor_l)/pconv
    Rv_l = Rv_scale*np.array(Rcor_l)/pconv

    Cim_l = Cim_scale*np.array(Ccor_l)
    Ca_l = Ca_scale*np.array(Ccor_l)


    print('printing LPN parameters')
    print(Ra_l)
    print(Ram_l)
    print(Rv_l)
    print(Ca_l)
    print(Cim_l)


    # Copy necessary files
    shutil.copyfile(path_root + '/Files/coronary.csv', path_to_sim + 'coronary.csv')
    shutil.copyfile(path_root + '/GenBC/CoronaryBC.h', path_to_genBC + 'CoronaryBC.h')
    shutil.copyfile(path_root + '/GenBC/CoronaryBC.cpp', path_to_genBC + 'CoronaryBC.cpp')
    shutil.copyfile(path_root + '/GenBC/GenBC.cpp', path_to_genBC + 'GenBC.cpp')
    # shutil.copyfile(path_root + '/GenBC/GenBC.exe', path_to_genBC + 'GenBC.exe')
    shutil.copyfile(path_root + '/GenBC/Makefile', path_to_genBC + 'Makefile')

    # Write GenBC input files
    with open(path_to_sim + 'coronaryParams.txt','w') as coronaryParam:
        for i in range(38):
            if i == 4:
                coronaryParam.write('1.0\n')
                # coronaryParam.write('0.5599974545\n') # iml
            else:
                coronaryParam.write('1.0\n')


    with open(path_to_sim + 'coronaryModel.txt','w') as coronaryModel:
        coronaryModel.write('{0},{1:.4f}\n'.format(6,hr))

        
        coronaryModel.write('FaceToStateMapping')
        temp_string = ''
        for face in cor_l_files:
            temp_string = temp_string + ',{0}'.format(face[-5])
        temp_string = temp_string + '\n'
        coronaryModel.write(temp_string)

        temp_string = 'Ra_l'
        for r_a_l in Ra_l:
            temp_string = temp_string + ',{0:.6f}'.format(r_a_l)
        temp_string = temp_string + '\n'
        coronaryModel.write(temp_string)

        temp_string = 'Ram_l'
        for r_am_l in Ram_l:
            temp_string = temp_string + ',{0:.6f}'.format(r_am_l)
        temp_string = temp_string + '\n'
        coronaryModel.write(temp_string)

        temp_string = 'Rv_l'
        for r_v_l in Rv_l:
            temp_string = temp_string + ',{0:.6f}'.format(r_v_l)
        temp_string = temp_string + '\n'
        coronaryModel.write(temp_string)

        temp_string = 'Ca_l'
        for c_a_l in Ca_l:
            temp_string = temp_string + ',{0:.10f}'.format(c_a_l)
        temp_string = temp_string + '\n'
        coronaryModel.write(temp_string)

        temp_string = 'Cam_l'
        for c_im_l in Cim_l:
            temp_string = temp_string + ',{0:.10f}'.format(c_im_l)
        temp_string = temp_string + '\n'
        coronaryModel.write(temp_string)
