import os
import vtk
import sys
from pathlib import Path

script_path = Path(os.path.realpath(__file__)).parent # This is defined differently than LCAmodel.py
sys.path.append(script_path)


################################################ Graphics
try:
    sys.path.append(str(script_path / 'Graphics/'))
    import graphics as gr
except:
    print("Can't find the /Graphics package.")

win_width = 600
win_height = 600
renderer, renderer_window = gr.init_graphics(win_width, win_height)



def DisplayRender(polydata, segmentations = None):
    gr.add_geometry(renderer, polydata, color=[1.0, 1.0, 1.0], edges=True)

    camera = renderer.GetActiveCamera();
    camera.Zoom(0.5)
    cont1 = segmentations[10]
    center = cont1.get_center()
    camera.SetFocalPoint(center[0], center[1], center[2])
    gr.display(renderer_window)
    return None


################################################ Folders

def CreateFolder(file_path, sim_folder = '/Smulations/'):
    if not os.path.exists(file_path):       
        print("file path doesn't exist. Trying to make")
        print(file_path)
        os.makedirs(file_path)

    # The next if statement checks to see if folder to be created is a root folder storing all ofhe geometry files.
    # If that is the case, it is usually cleared as it is assumed that the original files are unwanted.
    # However, that is not always the case, so the user can input whether they wish to clear or not.  

    elif os.path.exists(file_path) and str(file_path) == str(script_path) + sim_folder: #'/Simulations/'
        response = True
        response = input("\n\nThe {0} folder exists. Clearing existing files. If you do not want to clear files, type no. Any other input will register as a yes.\n".format(sim_folder))

        if response == 'no' or response == "No" or response == "NO" or response == "nO":
            return None
        else:
            for root, dirs, files in os.walk(file_path, topdown=False):
                for file in files:
                    if 'gitignore' not in file:
                        os.remove(os.path.join(root, file))
                # Add this block to remove folders
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))

    return None



                


################################################ VTP Operations

def WriteOutVTP(data, path_out):    
    out = vtk.vtkXMLPolyDataWriter()
    out.SetInputData(data)
    out.SetFileName(path_out)
    out.Update()
    out.Write()
    return None


def CreateNewVTP(new_points, save_path, add_branchID = False):
    new_data = vtk.vtkPolyData()
    points = vtk.vtkPoints()

    NoT = len(new_points)

    if add_branchID == True:
        branchIDarray = vtk.vtkIntArray()
        branchIDarray.SetName('BranchIDTmp')
        branchIDarray.SetNumberOfComponents(1)
        branchIDarray.SetNumberOfTuples(NoT)

    for i in range(0,NoT):
        points.InsertNextPoint(new_points[i,0:3])
        if add_branchID == True:
            branchIDarray.SetTuple1(i,new_points[i,3])

    new_data.SetPoints(points)
    
    if add_branchID == True:
        new_data.GetPointData().AddArray(branchIDarray)

    WriteOutVTP(new_data, save_path)

    return new_data


def ModifyVTP(modified_points, path_in, save_path):
    read_data = vtk.vtkXMLPolyDataReader()
    read_data.SetFileName(path_in)
    read_data.Update()
    data = read_data.GetOutput()
    NoP = data.GetNumberOfPoints()
    
    # Replace the points of the original data array with the modified ones
    for i in range(0,NoP):
        data.GetPoints().SetPoint(i, modified_points[i,0:3])

    WriteOutVTP(data, save_path)
    return data


################################################ Misc functions

def RadiiTaperScale(branchID):
    # Medrano-Gracia P, Ormiston J, Webster M, Beier S, Young A, Ellis C, Wang C, Smedby Ö, Cowan B. 
    # A computational atlas of normal coronary artery anatomy. EuroIntervention. 2016 Sep 18;12(7):845-54. 
    # doi: 10.4244/EIJV12I7A139. PMID: 27639736.

    
    slope = {}

    slope['1']  = -0.015
    slope['5']  = -0.016
    slope['8']  = -0.024
    slope['9']  = -0.024
    slope['10'] = -0.018
    slope['11'] = -0.018

    # the slope is divided by 2.0 to account for the use of radii, not diameters, in generating vessels

    slope_val = slope[str(int(branchID))] / 2.0

    return slope_val
