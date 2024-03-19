import sv
import os
import re
import sys
import vtk
import csv
import random
import numpy as np
from pathlib import Path
from scipy import interpolate
from scipy.special import binom

from vtk.util.numpy_support import vtk_to_numpy


############################################################################################## AdWd Directory Paths
script_path = Path(os.path.realpath(__file__)).parent

import utilsVTK as utils


############################################################################################## Graphics
try:
    sys.path.append(str(script_path / 'Graphics/'))
    import graphics as gr
except:
    print("Can't find the /Graphics package.")

win_width = 600
win_height = 600
renderer, renderer_window = gr.init_graphics(win_width, win_height)


############################################################################################## Define Constants
angle_surface = 60


############################################################################################## Main Functions
def GenerateFinalModel(data, radii, radii_mainbranch, save_path, num, name, phase_2, msize, edge, bl, eccentricity = False, 
                       radius_meshing = True, phase_5 = None):
    
    branches = set(data[:,3])
    models_ls = []
    paths_ls  = []

    if phase_5:
        plaque_burden = []

    for i in branches:
        name_branch = name + '_'+str(int(i))
        path_filter = data[np.in1d(data[:,3], i)]
        path_points = path_filter[:,0:3]
        path_radius = radii[np.in1d(radii[:,-1], i)][0]
        path_radius = np.array([path_radius[0], path_radius[1]])

        print('------------------------ START CREATING PATHS FOR BRANCH {0} ------------------------'.format(int(i)))
        path_sv = CreatePath(name_branch, path_points, 0, save_path, True)
        paths_ls.append(path_sv)
        print('------------------------ DONE GENERATING PATHS FOR BRANCH {0} -------------------------'.format(int(i)))
        
        
        print('------------------------ STARTING SEGMENTATIONS FOR BRANCH {0} -----------------------'.format(int(i)))
        if not phase_5:
            segmentations, radii_updated = CreateGroup(path_sv, name_branch, path_radius, i, radii, radii_mainbranch, 
                                                    save_path, phase_2, phase_5, seg_eccentricity = eccentricity)
        else:
            segmentations, radii_updated, pb_index, pb_max, sten = CreateGroup(path_sv, name_branch, path_radius, i, radii, 
                                                                               radii_mainbranch, save_path, phase_2, phase_5, 
                                                                               seg_eccentricity = eccentricity)
            if pb_index == 0 and pb_max == 0:       #this is only true when no stenosis is present
                pass
            else:
                plaque_burden.append([num, i, pb_index, pb_max, sten])
        print('------------------------ DONE GENERATING SEGMENTS FOR BRANCH {0} ----------------------'.format(int(i)))

        
        print('------------------------ STARTING MODEL GEN FOR BRANCH {0} ------------------------'.format(int(i)))
        models_ls.append(GenerateModel(segmentations, name_branch, save_path, msize, True))
        print('------------------------ DONE GENERATING MODELS FOR BRANCH {0} ------------------------'.format(int(i)))


    print('------------------------ STARTING TO UNION MODEL ------------------------')
    unioned_model = UnionVessels(models_ls, radii_mainbranch, segmentations, name, save_path, msize)
    if unioned_model == 'SV':
        return 'SV'
    print('------------------------ DONE GENERATING UNION MODEL ------------------------')


    print('------------------------ STARTING TO MESH MODEL ------------------------')
    MeshModel(unioned_model, save_path, radii_updated, radii_mainbranch, paths_ls, phase_5, edge_s = edge, BL_mesh = bl, 
              radius_based_meshing = radius_meshing)


    if not phase_5:
        return None
    else:
        return None, plaque_burden

############################################################################################## Meshing

def CalcArea(surface):
    surface_mass = vtk.vtkMassProperties()
    surface_mass.SetInputData(surface)
    surface_mass.Update()
    area = surface_mass.GetSurfaceArea()

    return area






def CheckMeshQuality(mesh):
    meshQuality = vtk.vtkMeshQuality()
    meshQuality.SetInputData(mesh)
    meshQuality.RatioOn()
    meshQuality.SetTriangleQualityMeasureToAspectRatio()
    meshQuality.SetTetQualityMeasureToAspectRatio()
    meshQuality.Update()
    meshQualityOutput = meshQuality.GetOutput()
    NoP = meshQualityOutput.GetNumberOfPoints()

    meshAspectRatioRange = meshQualityOutput.GetCellData().GetArray("Quality").GetRange()
    aRatio_avg = 0

    for i in range(NoP):
        aspectRatio = meshQualityOutput.GetCellData().GetArray("Quality").GetTuple(i)[0]
        aRatio_avg += aspectRatio
    aRatio_avg = aRatio_avg / NoP

    print("Mesh Quality Range, Aspect Ratio: " + str(meshAspectRatioRange))
    print("Mesh Quality Average, Aspect Ratio: " + str(aRatio_avg))

    meshQuality.SetTriangleQualityMeasureToScaledJacobian()
    meshQuality.SetTetQualityMeasureToScaledJacobian()
    meshQuality.Update()
    meshQualityOutput1 = meshQuality.GetOutput()

    meshJacobian = meshQualityOutput1.GetCellData().GetArray("Quality").GetRange()
    jRatio_avg = 0

    for i in range(NoP):
        ScJacobian = meshQualityOutput1.GetCellData().GetArray("Quality").GetTuple(i)[0]
        jRatio_avg += ScJacobian
    jRatio_avg = jRatio_avg / NoP

    print("Mesh Quality Range, Scaled Jacobian: " + str(meshJacobian))
    print("Mesh Quality Average, Scaled Jacobian: " + str(jRatio_avg))

    return None





def MeshingSlopeCheck(edge_s, start_radius):

    slope_estimation =  (edge_s - 0.0079) / start_radius

    if slope_estimation <= 0 and edge_s > 0:
        print('This edge size results in a smaller mesh than the alloted sloping function. not refining mesh by radius.')
        return False
    else:
        print('This edge size still adheres to the selected mesh scaling. will refine by radius')
        return True






def LocalEdgeSizeF(local_radius, edge_s, start_radius):
    # this relationship was emperically determined
    y_intercept = 0.0079
    local_edge_size  = y_intercept + 0.1*local_radius


    if edge_s != 'auto':
        '''
        This condition is used to estimate the slope the mesh tapering with radius. 
        This loop assumes that the slope will not be negative, given the precondition check in the MeshModel() function for
        MeshingSlopeCheck()
        '''

        slope_rescaled = (edge_s - y_intercept) / start_radius
        og_edge_size   = y_intercept + 0.1*start_radius
        y_intercept_sc = y_intercept * abs(edge_s / og_edge_size)

        if y_intercept_sc < 0:
            raise ValueError('please double check the prescribed edge size. Currently, it may be too small.')

        local_edge_size  = y_intercept_sc + slope_rescaled*local_radius

        print('interscept:', y_intercept_sc)
        print('slope:', slope_rescaled)


    return local_edge_size






def LocalSphereRemeshing(path_data, local_radius, nominal_radius, options, index, edge_s, start_radius):
    point_in_path = path_data.get_curve_point(index)
    center_sphere = [point_in_path[0], point_in_path[1], point_in_path[2]]
    local_edge_s  = LocalEdgeSizeF(local_radius, edge_s, start_radius)

    sphere_mesh_refinement = { 'edge_size':local_edge_s, 'radius':nominal_radius * 2.0 * 1.25, 'center':center_sphere} # oct 22 used to be 2*1.23, this used to be 2 * 2, going to change
    options.sphere_refinement.append(sphere_mesh_refinement)
    print('index: {0}, nominal rad: {1}, mean rad: {2}, edge size: {3}'.format(index, np.round(nominal_radius,4), 
          np.round(local_radius,4), np.round(local_edge_s,6)))
    
    return




def extract_numbers(filename):  
    temp = re.findall(r'\d+', filename)
    nums_in_file = list(map(int, temp))

    return nums_in_file[0], nums_in_file[1]






def PlaqueRefinement(phase_5, save_path, options, path_num, edge_s, start_radius, run=False):

    if path_num == 1:
        plaque_file = 'plaque_1.csv'
        plaque_num = '1'
    elif path_num == 5:
        plaque_file = 'plaque_5.csv'
        plaque_num = '5'
    else:
        return None, None


    if os.path.isfile(save_path + 'measurements/' + plaque_file):
        with open(save_path + 'measurements/' + plaque_file) as f:
            plaque_data = np.loadtxt(save_path + 'measurements/' + plaque_file, delimiter=",")
        
        plaque_length = phase_5[1][plaque_num]
        plaque_start_index = plaque_data[0,2]
        plaque_endin_index = plaque_data[-1,2]



        if run:
            print('_______conducting plaque region mesh refinement_______')   
            print('refining ', plaque_file, '  plaque length: ', plaque_length)
            print('plaque indices')
            print(plaque_data[:,2])


        if plaque_length < 1.5:
            # for short plaques, refinement is done at the center with consecutively smaller and finer spheres

            edge_s_plaque = np.mean([plaque_data[1,0], np.mean(plaque_data[1,0:2])])


            plaque_es = LocalEdgeSizeF(edge_s_plaque, edge_s, start_radius) 


            plaque_center = [plaque_data[1,3],plaque_data[1,4],plaque_data[1,5]]
            refinement_steps = [plaque_es*1.2, plaque_es*1.10, plaque_es, plaque_es*0.875, plaque_es*0.8]

            refinement_radii = [plaque_data[0,0]*3.2, plaque_data[0,0]*2.5, plaque_data[0,0]*1.9,
                                plaque_data[0,0]*1.5, plaque_data[0,0]*1.25]
            
            refinement_coord = [plaque_center, plaque_center, plaque_center, plaque_center, plaque_center]
        
        else:
            # when plaques are long, the .csv file is structured to include the transition points
            # as such, refinement is only done to the mesh at the tail end, since the original sphere refinement
            # will miss the required refinement at the end of the plaque

            edge_s_plaque = np.mean([plaque_data[2,0], np.mean(plaque_data[2,0:2])])


            plaque_es = LocalEdgeSizeF(edge_s_plaque, edge_s, start_radius)
            plaque_center = [plaque_data[2,3],plaque_data[2,4],plaque_data[2,5]]


            refinement_steps = [plaque_es, plaque_es*1.11, plaque_es*1.25]

            refinement_radii = [plaque_data[3,1]*2.95, plaque_data[3,1]*2.7, plaque_data[4,1]*1.9]
            refinement_coord = [[plaque_data[3,3],plaque_data[3,4],plaque_data[3,5]],
                                [np.mean(plaque_data[3:,3]),np.mean(plaque_data[3:,4]),np.mean(plaque_data[3:,5])],
                                [plaque_data[4,3],plaque_data[4,4],plaque_data[4,5]]]


        if run:             # only add the spheres to refinement when run == True
            for es, rad, coord in zip(refinement_steps, refinement_radii, refinement_coord):
                print('plaque refinement edge size: {0}, radius {1}'.format(round(es,4), round(rad,3)))
                sphere_refinement = { 'edge_size':es, 'radius':rad, 'center':coord}
                options.sphere_refinement.append(sphere_refinement)

            print('_______done  plaque mesh refinement_______')   

            return

        else:
            return refinement_radii, refinement_coord

    else:
        return None, None





def CustomRadiusMeshing(paths, radii, rad_mainbranch, options, save_path, phase_5, edge_s, start_radius):
    dict_path_num = {'0': 1, '1': 5, '2': 8, '3': 9, '4': 10, '5': 11}

    j = 0
    for path_data in paths:
        polydata = path_data.get_curve_polydata()               # get the path's polydata
        n_pts = polydata.GetPoints().GetNumberOfPoints() - 1    # subtract npts by 1, last point is 0,0,0 by default
        starting_radii = radii[j,0]                             # choose the start rad to define the total points num
        target_path = dict_path_num[str(j)]                     # mapping the path number to the actual branch num

        print('path:', target_path, '  num of points:', n_pts)


        if j == 0:
            bifurc_mainbranch = [rad_mainbranch[0,3:], rad_mainbranch[1,3:], rad_mainbranch[2,3:]]
        elif j == 1:
            bifurc_mainbranch = [rad_mainbranch[3,3:], rad_mainbranch[4,3:]]
        else:
            bifurc_mainbranch = [rad_mainbranch[j-1,3:]]

        # obtain measurements about the plaque mesh refinement zone
        if phase_5 and (target_path == 1 or target_path == 5):            
            plaque_length    = phase_5[1][str(target_path)]
            refinement_radii, refinement_coord = PlaqueRefinement(phase_5,save_path,options,target_path, edge_s, start_radius, run=False)


        target_index = 0
        current_point = path_data.get_curve_point(0)
        

        while target_index < n_pts:
            path_point = path_data.get_curve_point(target_index)
            norm_to_point = np.linalg.norm(np.array(path_point) - np.array(current_point))
            

            near_plaque_refinement = False
            if phase_5 and refinement_radii != None and refinement_coord != None:
                
                for plaque_rad, plaque_coord in zip(refinement_radii, refinement_coord):
                    norm_to_plaque = np.linalg.norm(np.array(path_point) - np.array(plaque_coord))
                    
                    if (norm_to_plaque <= plaque_rad):      # do not refine if the normal refinement is near the plaque          
                        near_plaque_refinement = True
                        break


            if near_plaque_refinement:
                print('skipping index: ', target_index, '  as it is close to plaque. Norm to plaque: ', norm_to_plaque)

            elif (norm_to_point >= starting_radii * 2 * 1.4) or target_index == 0:
                current_point = path_point              # set the current point as path point as the given index
                
                # find the starting radius based on the actual segmentation radii
                segmentations_folder = save_path + 'segmentations/'

                closest_file = None
                closest_distance = float('inf')
                
                for filename in os.listdir(segmentations_folder):
                    if filename.endswith('.vtp'):
                        path, index = extract_numbers(filename)

                        # Calculate the Euclidean distance between target and file values
                        distance = ((index - target_index) ** 2) ** 0.5
                        if distance < closest_distance and path == target_path:
                            closest_distance = distance
                            closest_file     = filename
                            closest_index    = index

                
                if closest_file is None:
                    raise ValueError('no close file found. check if segmentations were correctly created')
                
                seg_reader = vtk.vtkXMLPolyDataReader()
                seg_reader.SetFileName(segmentations_folder + closest_file)
                seg_reader.Update()
                seg_output = seg_reader.GetOutput()

                n_seg_points = seg_output.GetNumberOfPoints()
                segmentation_center = path_data.get_curve_point(closest_index)

                rad_seg = []
                for k in range(n_seg_points):
                    rad_seg.append(np.linalg.norm(np.array(seg_output.GetPoint(k)) - np.array(segmentation_center)))

                # use the mean radius found (particularly near noncircular segmentations, near plaques)
                new_radius = np.mean([min(rad_seg), np.mean(rad_seg)])


                # uuse the nominal radius (usually the largest) to set the sphere size
                nominal_radius = max(rad_seg)

                LocalSphereRemeshing(path_data, new_radius, nominal_radius, options, target_index, edge_s, start_radius)
                starting_radii = new_radius

            target_index += 1


        # refine the mesh near the plaque, only when run = True
        if phase_5:
            PlaqueRefinement(phase_5, save_path, options, target_path, edge_s, start_radius, run=True)


        # do the last point separately to ensure full refinement at the end of the branch
        if target_index != (n_pts - 1):
            LocalSphereRemeshing(path_data, new_radius, nominal_radius * 1.7, options, n_pts-1, edge_s, start_radius)

        j += 1

    return





def MeshModel(model, save_path, radii, radii_mainbranch, paths, phase_5, edge_s=False, BL_mesh=False, radius_based_meshing = True):

    mesher = sv.meshing.TetGen()
    
    ######################## set the mesh geometric parameters from the unioned model
    mesher.set_model(model)
    caps = model.identify_caps()
    wall_ids = [i + 1 for i in range(len(caps)) if not caps[i]]
    
    print("Wall ids are {}".format(wall_ids))
    mesher.set_walls(wall_ids)


    mesher.compute_model_boundary_faces(angle = angle_surface)
    face_ids = mesher.get_model_face_ids()
    print("Mesh face ids: " + str(face_ids))


    ######################## setting mesh options
    print("Set meshing options ... ")
    check_radius_meshing = True
    if edge_s == None:
        edge_size = compute_scale(model.get_polydata()) * 1.25
    else:
        if edge_s == 'auto':
            edge_size  = LocalEdgeSizeF(radii[0,0], edge_s, radii[0,0])
            print("edge size automatically set, written during : {}".format(edge_size))
        else:
            if edge_s > 0:
                edge_size = edge_s      # keep the user-defined edge size
                check_radius_meshing = MeshingSlopeCheck(edge_s, radii[0,0])
            else:
                raise ValueError('edge size must be greater than 0. please correct')

    print("edge size {}".format(edge_size))



    options = sv.meshing.TetGenOptions(global_edge_size=edge_size, surface_mesh_flag=True, volume_mesh_flag=True)

    ######################## perform radius based meshing if the flags are true
    ## two flags are used because even if it is requested, in some cases the requested edge size is already small enough
    ## to not warrant having additional radial refinement.
    if radius_based_meshing == True and check_radius_meshing == True:
        options.use_mmg = False         # this is a "rapid meshing option" it must be false to enable radius based meshing

        print('starting radius based mesh refinement ... ')
        CustomRadiusMeshing(paths, radii, radii_mainbranch, options, save_path, phase_5, edge_s, radii[0,0])
        options.sphere_refinement_on = True 
    else:
        options.use_mmg = True         # this is a "rapid meshing option" it must be false to enable radius based meshing



    ######################## include an inflation mesh near the walls
    if BL_mesh == True:
        print("Set boundary layer meshing options ... ")
        options.boundary_layer_inside = True
        mesher.set_boundary_layer_options(number_of_layers=3, edge_size_fraction=0.5, 
                                        layer_decreasing_ratio=0.6, constant_thickness=False)
    else:
        print('not adding boundary layer')



    ######################## generate the mesh
    mesher.generate_mesh(options)       # Initiate the mesh generation
    mesh = mesher.get_mesh()            # Get the mesh as a vtkUnstructuredGrid

    print("Mesh:")
    print("Number of nodes: {0:d}".format(mesh.GetNumberOfPoints()))
    print("Number of elements: {0:d}".format(mesh.GetNumberOfCells()))
    print("Face ids are {}".format(face_ids))
    print("Wall ids are {}".format(wall_ids))
    
    np.savetxt(save_path + '/measurements/mesh_data.txt', np.array([edge_size, mesh.GetNumberOfCells()]), fmt='%.5f')

    CheckMeshQuality(mesh)

    # # surface mesh
    meshes_area = []
    for i in face_ids:
        face_mesh = mesher.get_face_polydata(i)
        meshes_area.append(CalcArea(face_mesh))
        print('face: ' + str(i) + ' area: ' + str(CalcArea(face_mesh)))

    
    areas_sorted = sorted(meshes_area)

    out_mesh = save_path + 'mesh/'                      # Write the mesh
    out_mesh_surfaces = out_mesh + 'mesh-surfaces/'     # Write the mesh
    utils.CreateFolder(out_mesh)
    utils.CreateFolder(out_mesh_surfaces)


    print('--------Writing Mesh--------')
    for i in face_ids:
        face_mesh = mesher.get_face_polydata(i)
        if CalcArea(face_mesh) == max(meshes_area):
            wall_id = i
            mesh_surf_path = out_mesh_surfaces + 'wall_' + str(i) + '.vtp'
            utils.WriteOutVTP(face_mesh, mesh_surf_path)
        elif CalcArea(face_mesh) == areas_sorted[-2]:
            inlet_id = i
            mesh_surf_path = out_mesh_surfaces + 'inlet_' + str(i) + '.vtp'
            utils.WriteOutVTP(face_mesh, mesh_surf_path)            
        else:
            mesh_surf_path = out_mesh_surfaces + 'cap_' + str(i) + '.vtp'
            utils.WriteOutVTP(face_mesh, mesh_surf_path)


    mesh_path = out_mesh + '/' + 'mesh-complete.mesh.vtu'
    mesher.write_mesh(mesh_path)

    ext_path = out_mesh + '/' + 'mesh-complete.exterior.vtp'
    utils.WriteOutVTP(mesher.get_model_polydata(), ext_path)

    wall_path = out_mesh + '/' + 'walls_combined.vtp'
    utils.WriteOutVTP(mesher.get_surface(), wall_path)

    print('--------Mesh Built--------')

    return


############################################################################################## Modeling

def compute_scale(surface):
    #Compute a length scale for a vtkPolyData surface.
    num_cells = surface.GetNumberOfCells()
    points = surface.GetPoints()
    min_area = 1e6
    max_area = -1e6
    avg_area = 0.0

    for i in range(num_cells):
        cell = surface.GetCell(i)
        cell_pids = cell.GetPointIds()
        pid1 = cell_pids.GetId(0)
        pt1 = points.GetPoint(pid1)
        pid2 = cell_pids.GetId(1)
        pt2 = points.GetPoint(pid2)
        pid3 = cell_pids.GetId(2)
        pt3 = points.GetPoint(pid3)

        area = vtk.vtkTriangle.TriangleArea(pt1, pt2, pt3)
        avg_area += area

        if area < min_area:
            min_area = area
        elif area > max_area:
            max_area = area

    avg_area /= num_cells
    length_scale = np.sqrt(2.0 * avg_area)
    return length_scale 



def RemeshModels(model, msize):
    #the larger the msize (edge size), the coarser each mesh. smoothing needs to be really fine. inverse effect
    
    # msize = 0.014
    print("remesh grid size {}".format(msize))
    remesh_model = sv.mesh_utils.remesh(model.get_polydata(), hmin = msize, hmax = msize)
    model.set_surface(surface=remesh_model)
    model.compute_boundary_faces(angle = angle_surface)
    return model





def SmoothModel(model, radii_mainbranch, msize):
    print('------ BEGIN SMOOTHING ------')
    to_be_smoothed = model.get_polydata()
    for i in range(len(radii_mainbranch)):

        for s_size, c_factor in zip([4.25], [0.8]):
            radius = radii_mainbranch[i,0].tolist() * s_size # make the smoothing sphere larger
            center = radii_mainbranch[i,3:].tolist()
            print("branch #: {0}, radius: {1}, center: {2}, constrain: {3}".format(radii_mainbranch[i,2], np.round(radius,4),
                                                                                   np.round(center,4), c_factor))
            s_params = {'method':'constrained', 'num_iterations':4, 'constrain_factor':c_factor, 'num_cg_solves':10}
            smoothedVTK = sv.geometry.local_sphere_smooth(to_be_smoothed, radius, center, s_params)
            to_be_smoothed = smoothedVTK

        if i < len(radii_mainbranch) - 1:
            # msize = 0.014
            print('msize smooth model:', msize)
            to_be_smoothed = sv.mesh_utils.remesh(smoothedVTK, hmin = msize, hmax = msize)

    # msize = 0.014
    print('msize smooth model, final out of loop:', msize)
    remesh_model = sv.mesh_utils.remesh(to_be_smoothed, hmin = msize, hmax = msize)
    model.set_surface(surface=remesh_model)
    model.compute_boundary_faces(angle = angle_surface)

    return model




def GenerateModel(segmentations, name, save_path, msize, m_vtp = False):
    modeler = sv.modeling.Modeler(sv.modeling.Kernel.OPENCASCADE)

    curve_segmentations = []
    use_distance = False
    cid = 0
    start_cid = 0
    
    for seg in segmentations:
        if cid == start_cid:
            seg_align = seg.get_polydata()
        else:
            seg_align = sv.geometry.align_profile(last_seg_align, seg.get_polydata(), use_distance)

        last_seg_align = seg_align
        curve = modeler.interpolate_curve(seg_align)
        curve_segmentations.append(curve) 
        
        cid += 1
  
    # Generate geometry, opencascade
    surf = modeler.loft(curve_segmentations)
    print(surf.get_face_ids())
    surf_wcaps = modeler.cap_surface(surf)

    # Convert surface to PolyData
    model = sv.modeling.PolyData(surf_wcaps.get_polydata())
    model.compute_boundary_faces(angle = angle_surface)

    print('individual model surface face ID:')
    print(model.get_face_ids())


    # Remesh each individual branch model
    model = RemeshModels(model, msize)
    model.compute_boundary_faces(angle = angle_surface)

    # utils.DisplayRender(model.get_polydata(), segmentations)
    if m_vtp == True:
        out_path = save_path +  'model/' 
        utils.CreateFolder(out_path)
        utils.WriteOutVTP(model.get_polydata(), out_path + 'model_' + name + '.vtp')

    return model





def UnionVessels(model, radii_mainbranch, segmentations, name, save_path, msize):
    base_model = model[0]
    NoM = len(model)
    
    union_model = None

    modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA)

    # Write out the file
    out_union = save_path + 'model/'
    utils.CreateFolder(out_union)

    for i in range(1,NoM):
        print('unioning: ', i, ' out of: ', NoM-1)
        union_model = modeler.union(base_model, model[i])
        base_model = union_model


    rev_model = RemeshModels(union_model, msize)
    # utils.DisplayRender(rev_model.get_polydata(), segmentations)

    out_file = out_union +  name + '_unsmoothed.vtp'
    utils.WriteOutVTP(rev_model.get_polydata(), out_file)


    # Smooth the model
    smooth_model = SmoothModel(rev_model, radii_mainbranch, msize)
    # utils.DisplayRender(smooth_model.get_polydata(), segmentations)

    print('Remeshed, smoothed, union model surface face ID:')
    print(smooth_model.get_face_ids())
    

    out_file = out_union +  name + '_final.vtp'
    utils.WriteOutVTP(smooth_model.get_polydata(), out_file)

    if len(smooth_model.get_face_ids()) > 8:
        msg = "The number of face IDs exceeds the required 8. Please double check the radii values, and model remeshing edge size, and angles."
        print(msg)
        return 'SV'
    elif len(smooth_model.get_face_ids()) < 8:
        msg = "The number of face IDs is less than the required 8. Please double check the radii values, and model remeshing edge size, and angles."
        print(msg)
        return 'SV'

    return smooth_model



############################################################################################## Path

def CreatePath(path_name, path_points, num_modes, save_path, p_vtp):
    path = sv.pathplanning.Path()

    for k in range(len(path_points)):
        x, y, z = path_points[k]
        path.add_control_point([x,y,z])

    if num_modes != 0:    
        path = path.smooth(1, num_modes, smooth_control_pts=True)

    if p_vtp == True:
        out_path = save_path +  'paths/' 
        utils.CreateFolder(out_path)
        utils.WriteOutVTP(path.get_curve_polydata(), out_path + 'path_' + path_name + '.vtp')

    # gr.create_path_geometry(renderer, path)
    # gr.display(renderer_window)

    return path



############################################################################################## Segmentations
def CalcTotalLength(path_points):
    num_path_points = len(path_points)
    diff = np.diff(path_points, axis = 0)
    d = [np.linalg.norm(diff[i]) for i in range(num_path_points - 1)]
    total_path_length = sum(d)
    return total_path_length





def bezier(t, control_points):
    n = len(control_points) - 1
    return sum(binom(n, i) * t**i * (1 - t)**(n - i) * control_points[i] for i in range(n + 1))





def CalcPlaneVectors(normal):
    v1 = np.array([-normal[1], normal[0], 0])

    if np.linalg.norm(v1) < 1e-6:
        v1 = np.array([0, -normal[2], normal[1]])
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    return v1, v2





def AppendSegmentation(c, segmentations, out_seg, group_name, s_vtp, i):
    segmentations.append(c)
    if s_vtp:
        utils.WriteOutVTP(c.get_polydata(), out_seg + 'seg_' + group_name + '_' + str(i) + '.vtp')

    return





def CreateAtheromicSegmentations(i, branchID, plaque_features, approx_point_spacing, path, out_seg, PolyData_path, 
                                 segmentations, s_vtp, save_path, group_name, c_old):

    # identify if this is a branch to impose plaque on
    print(plaque_features)
    plaque_lad = plaque_features[0][str(1)]
    plaque_lcx = plaque_features[0][str(5)]
    
    
    if (int(branchID) == 1 and plaque_lad == 1) or (int(branchID) == 5 and plaque_lcx == 1):
        length     = plaque_features[1][str(int(branchID))]
        width_frac = plaque_features[2][str(int(branchID))]
        narrowing  = plaque_features[3][str(int(branchID))] - 0.05   # the -5% is to accomodate a consistently 5% larger stenosis than prescribed.
        max_plaque_angle = np.pi * width_frac
    else:
        done_plaque = True
        print('not generating plaque in branch: ', branchID)
        AppendSegmentation(c_old, segmentations, out_seg, group_name, s_vtp, i)
        return i, done_plaque, 0, 0, 0

    print("plaque length:", length, "      prescribed stenosis:", narrowing + 0.05)


    #################################################### START PLAQUE GENERATION
    total_i = int(round(length / approx_point_spacing)) 
    print(total_i, length, approx_point_spacing)

    num_segmentation_points = 27                        # number of points along the circumference


    if int(branchID) == 1:                              # LAD, no need to shift the angular starting position
        start_ang = np.pi / 2 - max_plaque_angle
    else:                                               # LCx, shifting the angle by -pi/2
        start_ang = - max_plaque_angle
    angles = list(np.linspace(start_ang, 2*np.pi + start_ang, num_segmentation_points))
    angles = angles[:-1]                                # to avoid an additional point when angle = 0 = 2*pi

    print('starting angle', int(180 /np.pi * start_ang))



    ########### use bezier curve to define the plaque narrowing, length wise
    bezier_narrow_before = np.array([1, 1, (1-narrowing), (1-narrowing)])
    bezier_narrow_after  = np.array([(1-narrowing), (1-narrowing), 1, 1])


    increment_segmentation_plaque = 2                   # spacing of segmentations
    starting_index_before_plaque  = i


    # write a list that contains the branch, point indices, and corresponding coordinates at those areas.
    if length < 1.0:
        coord_indices_arr = np.zeros((3,6))
        array_plaque_cent = 1
    else:
        coord_indices_arr = np.zeros((5,6))      # two more rows to count the transition point to stable stenosis
        array_plaque_cent = 2

        coord_indices_arr[1,1] = PolyData_path.GetPointData().GetArray('radius').GetTuple(i+int(total_i/3))[0]
        coord_indices_arr[1,2] = i + int(total_i/3)
        coord_indices_arr[1,3:] = path.get_curve_point(i+int(total_i/3))

        coord_indices_arr[3,1] = PolyData_path.GetPointData().GetArray('radius').GetTuple(i+int(2*total_i/3))[0]
        coord_indices_arr[3,2] = i + int(2*total_i/3)
        coord_indices_arr[3,3:] = path.get_curve_point(i+int(2*total_i/3))



    # Sometimes the indexing skips over the actual minimum segmentation region. use a finer approximation.
    coord_indices_arr[array_plaque_cent,1]  = PolyData_path.GetPointData().GetArray('radius').GetTuple(i+int(total_i/2))[0]
    coord_indices_arr[array_plaque_cent,2]  = i + int(total_i/2)
    coord_indices_arr[array_plaque_cent,3:] = path.get_curve_point(i+int(total_i/2))


    coord_indices_arr[-1,1]  = PolyData_path.GetPointData().GetArray('radius').GetTuple(i+total_i)[0]
    coord_indices_arr[-1,3:] = path.get_curve_point(i+total_i)


    segmentation_areas = []
    non_stenosed_areas = []
    act_stenosis_perct = []
    

    # this outer loop generates length wise plaque segmentations
    for j in range(0, total_i, increment_segmentation_plaque):
        center  = path.get_curve_point(i)
        seg_rad = PolyData_path.GetPointData().GetArray('radius').GetTuple(i)[0]

        # Normalize the normal vector
        normal = path.get_curve_tangent(i)
        normal = normal / np.linalg.norm(normal)
        
        # Find two orthogonal vectors in the plane
        v1, v2 = CalcPlaneVectors(normal) 

        local_points = []
        local_points_not_stenosed = []

        angle_counter = 0


        # Use the bezier interpolation to find the narrowing percentage along the length of the plaque
        if length < 1.0:
            if j <= int(total_i/2):
                narrow_percent = bezier(j/int(total_i/2), bezier_narrow_before)
            else:
                narrow_percent = bezier((j-int(total_i/2))/int(total_i/2), bezier_narrow_after)
        else:
            if j <= int(total_i/3):
                narrow_percent = bezier(j/int(total_i/3), bezier_narrow_before)
            elif j >= int(2*total_i/3):
                narrow_percent = bezier((j-int(2*total_i/3))/int(total_i/3), bezier_narrow_after)
            else:
                narrow_percent = bezier(0, bezier_narrow_after)

        print(j,i, total_i, narrow_percent)


        bezier_stenosis  = np.array([1, 1, narrow_percent, narrow_percent, 1, 1])

        radii_stenosis = []
        total_area = 0

        '''
        new change: shifting center of the segmentation down if needed to achieve a large stenois coverage.
        '''
        seg_rad_og = seg_rad
        center_og  = center
        center = center - seg_rad * v2 * (1-narrow_percent)
        seg_rad *= np.abs(narrow_percent)


        # main loop where the points along the segmentation with plaque are defined
        for ind, angle in enumerate(angles):

            '''
            new change: using angle fraction, not circumference.
            '''
            if ind/len(angles) <= width_frac:
                multiplication_factor = bezier(ind/len(angles)/width_frac, bezier_stenosis)

            else:
                multiplication_factor = 1



            # calculate the next point first attempt.
            x = seg_rad * np.cos(angle) * multiplication_factor
            y = seg_rad * np.sin(angle) * multiplication_factor
            
            x_normal = seg_rad_og * np.cos(angle)
            y_normal = seg_rad_og * np.sin(angle)
            
            local_point = [center[0] + x*v1[0] + y*v2[0], center[1] + x*v1[1] + y*v2[1], center[2] + x*v1[2] + y*v2[2]]
            local_point_not_stenosed = [center_og[0] + x_normal*v1[0] + y_normal*v2[0], 
                                        center_og[1] + x_normal*v1[1] + y_normal*v2[1], 
                                        center_og[2] + x_normal*v1[2] + y_normal*v2[2]]
            local_points.append(local_point)
            local_points_not_stenosed.append(local_point_not_stenosed)


            # calculate the total area of this stenosis
            if angle_counter > 0:
                p1 = np.array(local_point)-np.array(center)
                p2 = np.array(local_points[angle_counter-1]) - np.array(center)
                total_area += 0.5 * np.linalg.norm(np.cross(p1, p2))


            radii_stenosis.append(np.linalg.norm(np.asarray(local_point) - np.asarray(center_og)))
            angle_counter += 1


        # connect the first and last point to compute the final "triangle" that approximates the segmentation area
        p1 = np.array(local_point)-np.array(center)
        p2 = np.array(local_points[0]) - np.array(center)
        total_area += 0.5 * np.linalg.norm(np.cross(p1, p2))


        c = sv.segmentation.SplinePolygon(local_points)
        min_rad = min(radii_stenosis)



        '''
        new change: defining stenosis based on the min radius and direction of midpoint vector.
        '''
        mid_point = local_points[round(len(local_points)/2)]
        if np.dot(np.asarray(mid_point) - np.asarray(center_og), -v2) > 0 and narrowing >= 0.45:
            stenosis_percentage = (seg_rad_og + min_rad) / (seg_rad_og*2)

        elif np.dot(np.asarray(mid_point) - np.asarray(center_og), v2) > 0 and narrowing < 0.45:
            stenosis_percentage = 1 - (seg_rad_og + min_rad) / (seg_rad_og*2)
        else:
                stenosis_percentage = 1 - (seg_rad_og + min_rad) / (seg_rad_og*2)



        act_stenosis_perct.append(stenosis_percentage)
        segmentation_areas.append(total_area)
        non_stenosed_areas.append(np.pi*seg_rad_og**2)


        if j == 0:
            coord_indices_arr[0,0] = min_rad
            coord_indices_arr[0,1] = seg_rad_og
            coord_indices_arr[0,2] = i
            coord_indices_arr[0,3:] = center
        
        elif np.abs(j - int(total_i/2)) == 1 or np.abs(j - int(total_i/2)) == 0:
            print('middle')
            coord_indices_arr[array_plaque_cent,0] = min_rad

        elif np.abs(j - total_i) == 1 or np.abs(j - total_i) == 0 or np.abs(j - total_i) == 2:
            print('end')
            coord_indices_arr[-1,0] = min_rad    # store the actual final segmentation radius
            coord_indices_arr[-1,2]  = i         # store the actual final segmentation index

        if length > 1.0:
            if np.abs(j - int(total_i/3)) == 1 or np.abs(j - int(total_i/3)) == 0:
                coord_indices_arr[1,0] = min_rad # store the actual final segmentation radius
                coord_indices_arr[1,2] = i       # store the actual final segmentation index
            
            elif np.abs(j - int(2*total_i/3)) == 1 or np.abs(j - int(2*total_i/3)) == 0:
                coord_indices_arr[3,0] = min_rad # store the actual final segmentation radius
                coord_indices_arr[3,2] = i       # store the actual final segmentation index

        

        AppendSegmentation(c, segmentations, out_seg, group_name, s_vtp, i)
        # gr.create_segmentation_geometry(renderer, c)

        # create new segmentation as long the increment is closer to the end
        if j < total_i - increment_segmentation_plaque:
            i += increment_segmentation_plaque


    done_plaque = True

    # calculate the plaque burden
    plaque_burdens = (np.array(non_stenosed_areas) - np.array(segmentation_areas)) / np.array(non_stenosed_areas)
    index_max_plaque_burden = np.argmax(plaque_burdens) * increment_segmentation_plaque  +  starting_index_before_plaque
    value_max_plaque_burden = max(plaque_burdens)


    '''
    new change: not dividing narrowing by 2
    '''

    actual_stenosis_amount = max(act_stenosis_perct)
    plaque_details = [length, width_frac, narrowing]


    print('max plaque burden: ', index_max_plaque_burden, round(value_max_plaque_burden,3))
    print('actual stenosis %: ', round(actual_stenosis_amount * 100,3))
    print(act_stenosis_perct)


    plaque_headers = ['length', 'width', 'narrowing']

    np.savetxt(save_path + 'measurements/' + 'stenosis_acc_' + str(int(branchID)) + '.csv', [actual_stenosis_amount], delimiter=",")
    np.savetxt(save_path + 'measurements/' + 'plaque_' + str(int(branchID)) + '.csv', coord_indices_arr, delimiter=",")
    np.savetxt(save_path + 'measurements/pburden_' + str(int(branchID)) + '.csv', [index_max_plaque_burden,
                                                                                   value_max_plaque_burden], delimiter=",")


    print(plaque_details)
    with open(save_path + 'measurements/plaque_' + str(int(branchID)) + '_details.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(plaque_headers) 
        writer.writerow(plaque_details)


    return i, done_plaque, index_max_plaque_burden, value_max_plaque_burden, actual_stenosis_amount






def CreateGroup(path, group_name, radius, branchID, radii, radii_mainbranch, save_path, phase_2, phase_5, 
                r_interp = True, s_vtp = True, seg_eccentricity = False):    


    def EccentricPoints(center, normal, radius, interval, final_n, incr_spacing):
        
        normal = normal / np.linalg.norm(normal)            # Normalize the normal vector
        v1, v2 = CalcPlaneVectors(normal)                   # Find two orthogonal vectors in the plane
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]             # Define the angles for the four points (in radians)
        probability = random.random()                       # use a 50% probability split to assign diameter asymmetry

        

        # describe the diameter assymetry to form an eccentric segmentation, do not form at the start and ends of branches
        if  probability < 0.5 and interval > (incr_spacing) and interval < (final_n - incr_spacing):
            # Jihoon Kweon and others, Impact of coronary lumen reconstruction on the estimation of endothelial shear stress
            # Volume 19, Issue 10, October 2018, Pages 1134–1141, https://doi.org/10.1093/ehjci/jex222 

            diameter_ratio_asymmetry = random.uniform(0.8552, 0.9524) # based on the standard deviations of the inverse of dmax/dmin from Table 2
            print('diameter asymmetry, ', diameter_ratio_asymmetry, ' probablility ', probability)

        else:
            # treat the cross section as a circle
            diameter_ratio_asymmetry = 1


        # Calculate the local coordinates of the four points
        local_points = []
        eccentric_counter = 0

        for angle in angles:
            probability_rad_change = random.random()

            if probability_rad_change < 0.5 and eccentric_counter < 2:
                # only induce the eccentricity in two points and if the probability is < 0.5
                x = radius * np.cos(angle) * diameter_ratio_asymmetry
                y = radius * np.sin(angle) * diameter_ratio_asymmetry
                eccentric_counter +=1
            else:
                x = radius * np.cos(angle) * 1
                y = radius * np.sin(angle) * 1

            # local_points.append(center + x * v1 + y * v2)
            local_points.append([center[0] + x * v1[0] + y * v2[0], center[1] + x * v1[1] + y * v2[1],
                                center[2] + x * v1[2] + y * v2[2]])        

        return local_points


    path_total_length = CalcTotalLength(path.get_curve_points())
    
    segmentations = list()
    PolyData_path = path.get_curve_polydata()
    

    n = path.get_num_curve_points()
    print('points in the curve', n)
    print('points in the polydata_path', PolyData_path.GetNumberOfPoints())
    PolyData_path.GetPoints().SetNumberOfPoints(n)


    points = PolyData_path.GetPoints()
    point_array = vtk_to_numpy(points.GetData())
    point_array = point_array.reshape(-1, 3)


    # Define new arrays for path parametric coordinate and radius
    parametric_coord = vtk.vtkDoubleArray()
    parametric_coord.SetName('parametric_coord')
    parametric_coord.SetNumberOfComponents(1)
    parametric_coord.SetNumberOfTuples(n)

    local_radius = vtk.vtkDoubleArray()
    local_radius.SetName('radius')                                                                                                                                                                                                                                                                                                                                          
    local_radius.SetNumberOfComponents(1)
    local_radius.SetNumberOfTuples(n)


    if s_vtp:
        out_seg = save_path +  'segmentations/'
        utils.CreateFolder(out_seg)


    # radius interpolation that accounts for diameter biases
    if int(branchID) == 1:
        lcx_origin = radii_mainbranch[0,3:]
        index_of_intersection = np.argwhere(np.all(np.isclose(point_array, lcx_origin), axis=1))[0][0]

    if r_interp == True:                                                                     
        if int(branchID) == 1 and radii_mainbranch[0,0] < radii[1,0]:
            # the LAD is smaller than the LCx
            print('the LAD is smaller than the LCx')
            print(radii_mainbranch[0,0], radii[1,0])

            sb_coords = point_array[0:index_of_intersection+1,0:3]
            sb_length = CalcTotalLength(sb_coords)

            print(radii)
            print(index_of_intersection, sb_length, path_total_length)
            slope = utils.RadiiTaperScale(branchID)

            radius_lad_prox = radii_mainbranch[0,0]

            radii[0,0] = radii[1,0]*1.05 - slope * sb_length         # Make the starting radius based on LCx
            print(radii)
            print(radius_lad_prox)

            # At the start, interpolate the radius such that it runs from the LCx corrected start to 5% > LCx value
            f_start_lad = interpolate.interp1d(np.array([0,sb_length]), 
                                            np.array([radii[0,0], radii[1,0]*1.05]), fill_value = 'extrapolate')
            
            # for the middle, set the control points for the bezier curve
            d1_origin = radii_mainbranch[1,3:]
            d1_index  = np.argwhere(np.all(np.isclose(point_array, d1_origin), axis=1))[0][0]
            d1_coords = point_array[0:d1_index+1,0:3]
            d1_length = CalcTotalLength(d1_coords)


            bezier_len    = (d1_length - sb_length) * 0.5
            middle_bez_len= bezier_len / 2.0 + sb_length
            total_bez_len = bezier_len + sb_length

            bezier_radius = np.array([radii[1,0]*1.05, radii[1,0]*1.05, radius_lad_prox, radius_lad_prox])

            # after the bezier
            f_end_lad = interpolate.interp1d(np.array([total_bez_len,path_total_length]), 
                                            np.array([radius_lad_prox, radii[0,1]*1.05]), fill_value = 'extrapolate')
            
            starting_rad = radii[0,0]

        elif int(branchID) == 5 and radii_mainbranch[0,0] * 0.85 > radii[1,0]:
            om1_origin = radii_mainbranch[3,3:]
            om1_index  = np.argwhere(np.all(np.isclose(point_array, om1_origin), axis=1))[0][0]
            om1_coords = point_array[0:om1_index+1,0:3]
            om1_length = CalcTotalLength(om1_coords)
            

            slope = utils.RadiiTaperScale(branchID)

            radii_lcx_prox = radii_mainbranch[0,0] * 0.85

            bezier_len    = om1_length * 0.35
            bezier_radius = np.array([radii_lcx_prox, radii_lcx_prox, radii[1,0], radii[1,0]])
            

            f_end_lcx  = interpolate.interp1d(np.array([bezier_len,path_total_length]), 
                                            np.array([radii[1,0], radii[1,1]]), fill_value = 'extrapolate')

            starting_rad = radii[1,0]

        else:
            print('using stock radius interpolator')
            print(radius)
            f = interpolate.interp1d(np.array([0,path_total_length]), np.array([radius[0], radius[-1]]), fill_value = 'extrapolate')
            starting_rad = radius[0]

    else:
        print('using stock radius interpolator')
        f = interpolate.interp1d(np.array([0,path_total_length]), np.array([radius[0], radius[-1]]), fill_value = 'extrapolate')
        starting_rad = radius[0]


    # main radius setting loop
    path_curve_length = np.zeros(n)
    x0, y0, z0 = path.get_curve_point(0)
    parametric_coord.SetTuple1(0,0.0)
    local_radius.SetTuple1(0,starting_rad)

    approx_point_spacing = path_total_length / n        # this is used when determining the number of plaque segmentations
    print('total length of the path:',path_total_length)

    for i in range(1,n):
        x, y, z = path.get_curve_point(i)

        path_curve_length[i] = path_curve_length[i-1] + np.linalg.norm([x-x0, y-y0, z-z0])
        parametric_coord.SetTuple1(i,path_curve_length[i])

        if r_interp == 1:
            if int(branchID) == 1 and radii_mainbranch[0,0] < radii[1,0]:
                    if path_curve_length[i] < sb_length:
                        local_radius.SetTuple1(i, f_start_lad(path_curve_length[i]))

                    elif sb_length <= path_curve_length[i] < total_bez_len:
                        local_radius.SetTuple1(i, bezier((path_curve_length[i] - sb_length)/bezier_len, bezier_radius))

                    elif path_curve_length[i] >= total_bez_len:
                        local_radius.SetTuple1(i, f_end_lad(path_curve_length[i]))

            elif int(branchID) == 5 and radii_mainbranch[0,0] * 0.85 > radii[1,0]:
                    if path_curve_length[i] < bezier_len:
                        local_radius.SetTuple1(i, bezier(path_curve_length[i]/bezier_len, bezier_radius))
                    elif path_curve_length[i] >= bezier_len:
                        local_radius.SetTuple1(i, f_end_lcx(path_curve_length[i]))

            else:
                local_radius.SetTuple1(i,f(path_curve_length[i]))
        else:
            local_radius.SetTuple1(i,f(path_curve_length[i]))
        
        x0, y0, z0 = x, y, z

    PolyData_path.GetPointData().AddArray(parametric_coord)
    PolyData_path.GetPointData().AddArray(local_radius)


    # empirically determined incremental spacing
    if seg_eccentricity == False:
        if branchID == 1 or branchID == 5:
            incr = 5
        else:
            incr = 7
    else:
        if branchID == 1 or branchID == 5:    
            incr = 5
            incr_spacing = incr * 6
        else:
            incr = 7
            incr_spacing = incr * 2
    print('segmentation increment: ', incr)


    # Store the radius and segment frame in a sv segmentation file
    i = 0
    changed_incr = False
    done_plaque  = False


    # Generate segmentations
    while i < n:
        print(i, incr, n-1, n - incr - 1)

        if i >= (n - incr - 1) and not changed_incr:
            if (n - i - 1) % 2 == 0:
                incr = 2
            elif (n - i - 1) % 3 == 0: 
                incr = 3
            else:
                incr = 1
            
            incr = (n-1) - i

            print('changing increment to: ', incr, i + incr)
            changed_incr = True


        seg_normal = path.get_curve_tangent(i)
        seg_center = path.get_curve_point(i)
        seg_frame = path.get_curve_frame(i)
        seg_radius = PolyData_path.GetPointData().GetArray('radius').GetTuple(i)[0]
       
        # impose eccentricity on the lumen cross sectional areas
        if seg_eccentricity == False:
            c = sv.segmentation.Circle(radius=seg_radius, frame=seg_frame)
        else:
            circle_points = EccentricPoints(seg_center, seg_normal, seg_radius, i, n, incr_spacing)
            c = sv.segmentation.SplinePolygon(circle_points)


        # generate plaque
        if phase_5 != None and done_plaque == False:
            print(' plaque being complete is ', done_plaque)
            if (int(branchID) == 5):
                if i == 0:
                    print('LCX plaque generation')
                    i, done_plaque, index_pb, pb, sten = CreateAtheromicSegmentations(i, branchID, phase_5, approx_point_spacing, 
                                                                                      path, out_seg, PolyData_path,
                                                                                      segmentations, s_vtp, save_path, 
                                                                                      group_name, c)
            elif int(branchID) == 1:
                if i >= index_of_intersection:
                    print('LAD Plaque generation', index_of_intersection, i)
                    i, done_plaque, index_pb, pb, sten = CreateAtheromicSegmentations(i, branchID, phase_5, approx_point_spacing, 
                                                                                      path, out_seg, PolyData_path,
                                                                                      segmentations, s_vtp, save_path, 
                                                                                      group_name, c)
                else:
                    AppendSegmentation(c, segmentations, out_seg, group_name, s_vtp, i)
            else:
                AppendSegmentation(c, segmentations, out_seg, group_name, s_vtp, i)
        else:
            AppendSegmentation(c, segmentations, out_seg, group_name, s_vtp, i)

        # #### Add segmentation contour to contour window.
        # gr.create_segmentation_geometry(renderer, c)

        i += incr


    # ##### Display window with paths and segmentation contours.
    # if int(branchID) == 1:
    #     gr.create_path_geometry(renderer, path)
    #     gr.display(renderer_window)
    

    radii_updated = radii   # because the radius is updated for the LAD, this is stored in a new variable and passed


    if phase_5 and done_plaque == True:
        return segmentations, radii_updated, index_pb, pb, sten
    elif phase_5:
        return segmentations, radii_updated, 0, 0, 0
    else:
        return segmentations, radii_updated