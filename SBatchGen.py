import os
import re
import sys

################################################ Add Directory Paths
path_root = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_root)
import utilsVTK as utils


'''
directory path is a list of the geometries that will be run
savedirectory/ == geom_1, geom_2, etc
'''

################################################ Functions


def GenerateSBatchJob(directory_path):

    file_path = directory_path + 'runsims.sh'


    with open(file_path, 'w') as f:

        for curr_dir, dirs, files in os.walk(directory_path):
            for subdirs in dirs:
                if 'Geometry' in subdirs:
                    # find the number in the subdirs name
                    temp = re.findall(r'\d+', subdirs)
                    num = list(map(int, temp))

                    f.write('\ncd ' + str(subdirs))
                    f.write('\nchmod u+x geom_' + str(num[0]) + '.job')
                    f.write('\nsbatch geom_' + str(num[0]) + '.job')
                    f.write('\ncd ..')
                    f.write('\n')      


    return None




def RunMPI(directory_path, num_nodes, num_procs, i, hr, time_step, total_periods, increment):
    file_path = directory_path + 'geom_' + str(int(i)) + '.job'

    tot_cores = num_procs*num_nodes

    with open(file_path, 'w') as f:

        f.write('#!/bin/bash')
        f.write('\n#SBATCH -N ' + str(int(num_nodes)))

        if num_procs != 128:
            f.write('\n#SBATCH -p RM-shared')
        else:
            f.write('\n#SBATCH -p RM')

        # '''
        # Note to user. Please change the total time you request for your .job file as required.
        # '''

        f.write('\n#SBATCH -t 24:00:00')
        f.write('\n#SBATCH --ntasks-per-node=' + str(int(num_procs)))
        f.write('\n\nset -x\n')


        # '''
        # Note to user. Please change the file path to where simvascular's svsolver is built in your HPC file network.
        # '''

        f.write('\nmodule load gcc')
        f.write('\nmodule load python/3.8.6')
        f.write('\nmodule load openmpi/4.1.1-gcc8.3.1')
        f.write('\nmodule load openblas/0.3.13-intel20.4\n')
        f.write('\nmpirun /projects/shared/Software/Simvascular/svSolver_build/svSolver-build/bin/svsolver solver.inp\n')
        f.write('\nbash postprocess.sh\n\n')


    # create results (.vtu, .vtp) generating script that calls the svpost (postprocessing pipeline)
    postprocess_path = directory_path + 'postprocess.sh'
    period = round(1 / hr * 60, 2) 

    results_path = directory_path + 'results/'
    utils.CreateFolder(results_path)

    with open(postprocess_path, 'w') as f:
        # '''
        # Note to user. Please change the file path to where simvascular's svpost is built in your HPC file network.
        # '''

        f.write('svpost=/projects/shared/Software/Simvascular/svSolver_build/svSolver-build/bin/svpost\n')

        
        f.write('\nindir=' + str(num_procs*num_nodes) + '-procs_case')
        f.write('\noutdir=results')
        f.write("\nstart=" + str(round(period * (total_periods-1) / time_step)))
        f.write("\nstop=" + str(round(period * (total_periods) / time_step)))
        f.write("\ninc=" + str(increment) + "\n\n")

        f.write(r'''
$svpost -all  \
    -indir ${indir}   \
    -outdir ${outdir}  \
    -start ${start}  \
    -stop ${stop}  \
    -incr ${inc}  \
    -vtp all_results  \
    -vtu all_results        
''')

    return None
