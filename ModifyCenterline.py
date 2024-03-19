import os
import vtk
import sys
import math
import numpy as np
from pathlib import Path
from scipy.interpolate import splprep, splev, interp1d, make_interp_spline


################################################ Add Directory Paths
path_root = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Path(os.path.realpath(__file__)).parent)

import utilsVTK as utils


temp_testing = True



################################################ Main Functions
def ReadCenterlineVTP(centerline_name):
    if not os.path.isfile(centerline_name):
        raise NameError("Could not find file: %s" % centerline_name)

    centerline_reader = vtk.vtkXMLPolyDataReader()
    centerline_reader.SetFileName(centerline_name)
    centerline_reader.Update()
    centerline = centerline_reader.GetOutput()
    NoP = centerline.GetNumberOfPoints()

    X = list()
    Y = list()
    Z = list()
    B = list()
    P = list()
    R = list()

    for i in range(0,NoP):
        # BranchID correlates to each individual branch, but assigns ID -1 to a split
        # BranchIDTmp is the number of the branch (correlates to the total number of paths)
        # MaximumInscribedSphereRadius corresponds to the radius at a given point
        point  = centerline.GetPoints().GetPoint(i)
        branch = centerline.GetPointData().GetArray('BranchId').GetTuple(i)
        path   = centerline.GetPointData().GetArray('BranchIdTmp').GetTuple(i)
        radius = centerline.GetPointData().GetArray('MaximumInscribedSphereRadius').GetTuple(i)
        
        X.append(point[0])
        Y.append(point[1])
        Z.append(point[2])
        B.append(branch[0])
        P.append(path[0])
        R.append(radius[0])

    x = np.reshape(X, (NoP,1))
    y = np.reshape(Y, (NoP,1))    
    z = np.reshape(Z, (NoP,1))
    b = np.reshape(B, (NoP,1))
    p = np.reshape(P, (NoP,1))
    r = np.reshape(R, (NoP,1))

    centerline_points = np.concatenate((x,y,z,b,p,r), axis=1)

    # Fix the misprescribed -1 in some data entries
    for i in set(centerline_points[:,4]):
        branch_data = centerline_points[np.in1d(centerline_points[:, 4], i)]
        if branch_data[-1,3] == -1 and i != 0 and i < 12:
            index = np.argwhere(centerline_points[:,0:3] == branch_data[-1, 0:3])[0,0]
            centerline_points[index, 3] = centerline_points[index-1,3]
    
    return centerline_points



def ReadBifurcPoints(data):
    #extract all the indices that are bifurcations, i.e. equal to -1
    bifurc = [i for i in range(len(data)) if data[i,3] == -1]
    points = np.zeros((len(bifurc),5))

    i = 0
    for j in bifurc:
        points[i,0:3] = data[j,0:3]
        points[i,3] = data[j,4]         # Data column 5 is the BranchIdTmp
        points[i,4] = j                 # points column 5 is the tuple index
        i += 1
        
    return points



def BifurcLines(data):
    # this function is used to calculate the vectors that form bifurcations.

    num_branches = len(set(data[:,3]))
    
    # only needs 2 points per branch (first and last)
    # first three columns are coordinates, 4th is branchID, 5th is corresponding pair, 6th is index in vtp file 
    points = np.zeros((num_branches*2,6)) 

    # End points
    points[0,0:4] = data[0,0:4]
    points[-1,0:4] = data[-1,0:4]

    i = 1
    # for loop to check the first and last point of a given bifurcation branch
    for j in range(len(data)):
        if j < len(data) - 1:
            if data[j+1,3] != data[j,3] and ((13 > data[j,3] >= 0) or (data[j,3] > 14)):
                points[i,0:4] = data[j,0:4]
                points[i+1,0:4] = data[j+1,0:4]
                points[i,5] = data[j,4]
                points[i+1,5] = data[j+1,4]
                i += 2
            elif data[j+1,3] != data[j,3] and (data[j,3] == 13):
                points[i,0:4] = data[j-1,0:4]   #value at j is a skipped one
                points[i+1,0:4] = data[j+1,0:4]
                points[i,5] = data[j-1,4]
                points[i+1,5] = data[j+1,4]
                i += 2
            elif data[j+1,3] != data[j,3] and data[j,3] == 14:
                points[i,0:4] = data[j-2,0:4]   #value at j is a skipped two
                points[i+1,0:4] = data[j+1,0:4]
                points[i,5] = data[j-2,4]
                points[i+1,5] = data[j+1,4]
                i += 2
    
    # pair calculator
    for i in np.arange(len(points), step = 2):
        points[i,4] = AnglePairRelations(points[i,3])
        points[i+1,4] = AnglePairRelations(points[i,3])

    return points



def CalcDistances(coords):
    diff = np.diff(coords, axis = 0)
    d_s  = [np.linalg.norm(diff[k,:]) for k in range(len(coords) -1)]
    total_length = sum(d_s)
    return total_length


'''
The following functions prescribe a set of fixed relationships between branchIDs and their relative downstrem counterparts,
pairs that form angles, and upstream counterparts. They were measured based on the original left coronary model.
'''


def DownstreamPoints(val):
    downstream_branches = []
    downstream_branches.append(np.arange(0,19)) # 0 
    downstream_branches.append(np.arange(1,12)) # 1
    downstream_branches.append(np.array([2,3,4,8,9])) # 2
    downstream_branches.append(np.array([3,4,9])) # 3
    downstream_branches.append(np.array([4])) # 4
    downstream_branches.append(np.array([5,6,7,10,11])) # 5
    downstream_branches.append(np.array([6,7,11])) # 6
    for i in range(7,12):
        downstream_branches.append(np.array([i]))
    downstream_branches.append(np.array([12,13,14,15,16,17])) # 12
    downstream_branches.append(np.array([13,14,15,16,17])) # 13
    downstream_branches.append(np.array([14,15,17])) # 14
    for i in range(15,19):
        downstream_branches.append(np.array([i]))
    ds_vals = downstream_branches[int(val)]
    return ds_vals




def AnglePairRelations(val):
    relation = {}
    relation['0']  = 0
    relation['1']  = 0
    relation['2']  = 5
    relation['3']  = 8
    relation['4']  = 9
    relation['5']  = 2
    relation['6']  = 10
    relation['7']  = 11
    relation['8']  = 3
    relation['9']  = 4
    relation['10'] = 6
    relation['11'] = 7
    relation['12'] = 0
    relation['13'] = 12
    relation['14'] = 16
    relation['15'] = 17
    relation['16'] = 14
    relation['17'] = 15
    relation['18'] = 0

    rel_val = relation[str(int(val))]
    return rel_val



def SideMainBranchRelations(val):
    relation = {}
    relation['5']  = np.array([1,2])
    relation['8']  = np.array([2,3])
    relation['9']  = np.array([3,4])
    relation['10'] = np.array([5,6])
    relation['11'] = np.array([6,7])
    relation['16'] = np.array([13,14])
    relation['17'] = np.array([14,15])

    rel_val = relation[str(val)]
    return rel_val



def UpstreamBranchRelations(val):
    relation = {}
    relation['1']  = 0
    relation['2']  = 1
    relation['3']  = 2
    relation['4']  = 3
    relation['5']  = 1
    relation['6']  = 5
    relation['7']  = 7
    relation['8']  = 2
    relation['9']  = 3
    relation['10'] = 5
    relation['11'] = 6
    relation['12'] = 0
    relation['13'] = 12
    relation['14'] = 13
    relation['15'] = 15
    relation['16'] = 13
    relation['17'] = 14
    relation['18'] = 0

    rel_val = relation[str(val)]
    return rel_val



def UpstreamBranchRelationsSpaced(val):
    relation = {}

    relation['1']  = 0
    relation['5']  = 1
    relation['8']  = 1
    relation['9']  = 1
    relation['10'] = 5
    relation['11'] = 5

    rel_val = relation[str(val)]
    return rel_val



def MainBranchCheck(val):
    relation = {}

    relation['1'] = 1
    relation['0']  = 0
    relation['2']  = 1
    relation['3']  = 1
    relation['4']  = 1
    relation['5']  = 5
    relation['6']  = 5
    relation['7']  = 5
    relation['8']  = False
    relation['9']  = False
    relation['10'] = False
    relation['11'] = False
    relation['12'] = 12
    relation['13'] = 12
    relation['14'] = 12
    relation['15'] = 12
    relation['16'] = False
    relation['17'] = False
    relation['18'] = 0

    rel_val = relation[str(val)]
    return rel_val




def SphereCheck(data, mb_vector, branchID, LAD_LCX_angle):
    '''
    This function checks to see if the side branches exist within a spherical zone near the heart.
    These spheres were emperically determined.
    '''


    branch_data   = np.asarray(data[np.in1d(data[:, 4], branchID)])
    origin_vector = -1*branch_data[0,0:3]
    tolerance     = 0.02    # [cm]


    if branchID == 11:
        return data


    # These values for the baseline heart's sphere 'center' and 'radius' were empirically determined
    if branchID == 10:
        neutral_centroid = [-9.4485299363665, 7.23165420289725, -3.71765728454136]
        branch_5_center  = np.asarray(data[np.in1d(data[:, 4], 5)])[0,0:3]

        adjustment_vector = branch_5_center - neutral_centroid


        center = [-8, 9,-8.6] + adjustment_vector
        rad_f  = interp1d(np.array([35.2,105.2]), np.array([7.10, 4.5]), fill_value = 'extrapolate')

        radius = rad_f(LAD_LCX_angle)
    else:
        center = [-8, 9,-9.5]
        radius = 6.9
    
    # Do not change the position of one of these branch segments
    if branchID == 1 or branchID == 5 or branchID == 6 or branchID == 2 or branchID == 3:
        return data 

    # Check if the coordinate lies within the sphere or outside:
    distance = np.linalg.norm(center - branch_data[-1,0:3])
    changed_data = np.copy(data)


    if distance > radius:
        print('branch {0} is erroneously far from its original position. Correcting'.format(branchID))

        end_point_data = branch_data[-3:,:]
        potential_angles = []


        for angle in np.arange(np.pi, -1*np.pi, -np.pi/180):                 
            point_rot = AffineTransformation([branchID], angle, end_point_data, mb_vector, origin_vector, fix_alignment= True)
            distance_new = np.linalg.norm(center - point_rot[-1,:])

            if distance_new < distance and np.abs(distance_new - radius) < tolerance:
                potential_angles.append(angle)

        
        if len(potential_angles) == 0:
            raise ValueError('The angle/position combination results in nonrealistic centerline coordinates. Please double check and change this combination')
        


        angle_final = min(potential_angles, key=abs)
        

        data_to_change, points_rot = AffineTransformation([branchID], angle_final, data, mb_vector, origin_vector)
        changed_data = ChangeDataPoints(data_to_change, changed_data, points_rot)


        # print('final', np.linalg.norm(center - points_rot[-1,:]) , distance, angle_final)
        if np.linalg.norm(center - points_rot[-1,:]) > distance:
            raise ValueError('Rotation correction did not bring the branch within the defined sphere.')

        return changed_data

    elif distance < radius*0.9:
        print('branch {0} is towards the center of the sphere. Correcting'.format(branchID))
        print(distance, radius)

        end_point_data = branch_data[-3:,:]
        potential_angles = []

        for angle in np.arange(-np.pi, np.pi, np.pi/180):            
            point_rot = AffineTransformation([branchID], angle, end_point_data, mb_vector, 
                                             origin_vector, fix_alignment = True)
            
            distance_new = np.linalg.norm(center - point_rot[-1,:])

            if distance_new > distance and np.abs(distance_new - radius) < tolerance:
                potential_angles.append(angle)


        if len(potential_angles) == 0:
            raise ValueError('The angle/position combination results in nonrealistic centerline coordinates. Change this combination')
        

        angle_final = min(potential_angles, key=abs)


        data_to_change, points_rot = AffineTransformation([branchID], angle_final, data, mb_vector, origin_vector)
        changed_data = ChangeDataPoints(data_to_change, changed_data, points_rot)

        if np.linalg.norm(center - points_rot[-1,:]) < distance:
            raise ValueError('Rotation correction did not bring the branch within the defined sphere.')

        return changed_data
    

    else:
        # Branch is close to the sphere radius, no need to rotate it
        print('branch {0} is close to the sphere perimeter. No correction applied'.format(branchID))
        print(distance, radius)
        return data



def AngleCalculator(centerline_points):
    bifurc_points = ReadBifurcPoints(centerline_points)
    line_points   = BifurcLines(bifurc_points)

    data_angles  = line_points[:,:]
    num_branches = int(len(line_points)/2.0)
    vectors = np.zeros((num_branches,6)) # 6th column is the angle

    # make the first coordinate the same 
    for i in np.arange(len(line_points), step = 2):
        for j in np.arange(len(line_points), step = 2):
            if data_angles[j,3] == data_angles[i,4] and data_angles[j,3] != 0 and data_angles[j,4] != 0:
                data_angles[i,0:3] = data_angles[j,0:3]

    # calculate the vectors for each branch segment
    for i in range(num_branches):
        # determine the unit vectors for changing
        vectors[i,0:3] = data_angles[(i*2+1), 0:3] - data_angles[i*2, 0:3]
        vectors[i,3] = data_angles[i*2, 3]
        vectors[i,4] = data_angles[i*2, 4]

    # angle calculating loop
    for i in range(num_branches):
        for j in range(num_branches):        
            if vectors[i,3] == vectors[j,4]: #i.e. if they are a pair
                vectors[i,5] = math.degrees(np.arccos(np.dot(vectors[i,0:3],vectors[j,0:3]) 
                                                      / np.linalg.norm(vectors[i,0:3]) 
                                                      / np.linalg.norm(vectors[j,0:3])))
                
    return vectors, data_angles



def BranchLengthCalculator(data, spaced = False):
    if spaced:
        column = 3  # in the smoothed models, the branchID is stored in column 4
    else:
        column = 4  # branchIDTmp is stored in column 5, from VTK raw data
    
    branches = set(data[:,column])
    branch_length = np.zeros((len(branches), 2))
    
    j = 0
    for i in branches:
        branch_data = data[np.in1d(data[:, column], i)]
        length = CalcDistances(branch_data[:,0:3])
        branch_length[j,0] = int(i) 
        branch_length[j,-1] = length
        j += 1
    
    return branch_length



def DataToChange(ds, data, column):
    # determine all of the points in the original vtp data that have to be modified, by row
    data_to_change = []
    for branch in ds:
        for i in range(len(data)):
            if int(data[i,column]) == branch: #the 5th column is the BranchIDTmp, which does not have -1 (all points)
                data_to_change.append(np.append(data[i,0:3], i)) #store the index in the original array to change
    data_to_change = np.asarray(data_to_change)
    
    return data_to_change



def ChangeDataPoints(data_to_change, changed_data, altered_points, data_structure_orig = True):
    # store the transformed coordinates in the original data (changed_data) array
    if data_structure_orig == True:
        column = 3
    else:
        column = 4

    i = 0
    for j in data_to_change[:,column]:
        changed_data[int(j),0:3] = altered_points[i,0:3]
        i += 1

    return changed_data



def PointsRotation(rot_matrix, x_orig, y_orig, z_orig, data_to_change):
    X = data_to_change[:,0:3].T
    A = np.array([[0, 0, 0, (x_orig*rot_matrix[0,0] + y_orig*rot_matrix[0,1] + z_orig*rot_matrix[0,2]) + -1*x_orig],
                [0, 0, 0, (x_orig*rot_matrix[1,0] + y_orig*rot_matrix[1,1] + z_orig*rot_matrix[1,2]) + -1*y_orig],
                [0, 0, 0, (x_orig*rot_matrix[2,0] + y_orig*rot_matrix[2,1] + z_orig*rot_matrix[2,2]) + -1*z_orig],
                [0, 0, 0, 0]]) + rot_matrix
    B = np.r_[X, [np.ones(len(data_to_change))]]

    points_rotated = A @ B
    points_rotated = points_rotated[0:-1,:].T # remove the final row with the constant 1 values
    return points_rotated



def RotationMatrix(theta, x_rot, y_rot, z_rot):
    rot_mat = np.array([[x_rot*x_rot*(1-np.cos(theta)) + np.cos(theta),y_rot*x_rot*(1-np.cos(theta)) 
                            - z_rot*np.sin(theta),z_rot*x_rot*(1-np.cos(theta)) + y_rot*np.sin(theta),0],
                        [x_rot*y_rot*(1-np.cos(theta)) + z_rot*np.sin(theta),y_rot*y_rot*(1-np.cos(theta)) 
                            + np.cos(theta),z_rot*y_rot*(1-np.cos(theta)) - x_rot*np.sin(theta),0],
                        [x_rot*z_rot*(1-np.cos(theta)) - y_rot*np.sin(theta),y_rot*z_rot*(1-np.cos(theta)) 
                            + x_rot*np.sin(theta),z_rot*z_rot*(1-np.cos(theta)) + np.cos(theta),0],
                        [0,0,0,1]])
    return rot_mat



def AffineTransformation(ds, theta, data, rot_vector, origin_vector, fix_alignment = False):
    x_rot, y_rot, z_rot = rot_vector[0], rot_vector[1], rot_vector[2]               # vector normal to rotation plane
    x_orig, y_orig, z_orig = origin_vector[0], origin_vector[1], origin_vector[2]   # point the rotation is about


    data_to_change  = DataToChange(ds, data, 4)
    rotation_matrix = RotationMatrix(theta, x_rot, y_rot, z_rot)
    points_rotated  = PointsRotation(rotation_matrix, x_orig, y_orig, z_orig, data_to_change)

    if fix_alignment == False:
        return data_to_change, points_rotated
    else:
        return points_rotated



def AngleManipulate(branchID, angle_change, data, type_of_change):
    vectors, origin = AngleCalculator(data)

    # change the angle of row 6
    # assign the origin to the 12th row (correlating to 6th branch, given 2 rows per branch for first and last point)
    angle_stock = vectors[branchID,5]
    origin_data = origin[branchID*2,:]

    if angle_stock != angle_change:
        b1 = origin_data[3]
        b2 = origin_data[4]

        ds1 = DownstreamPoints(b1)                  # downstream branch 1
        ds2 = DownstreamPoints(b2)                  # downstream branch 2
        ds = np.concatenate((ds1, ds2), axis = 0)
        ds.sort()                                   # sorts by shortest to longest
        
        # determine the unit vector about which to perform rotation
        planar_vectors = np.zeros((2,3))
        for i in range(len(vectors)):
            if vectors[i,3] == b1:
                planar_vectors[0,:] = vectors[i,0:3] / np.linalg.norm(vectors[i,0:3])
            elif vectors[i,3] == b2:
                planar_vectors[1,:] = vectors[i,0:3] / np.linalg.norm(vectors[i,0:3])

        # define vectors for origin (at bifurcation junction) and rotation matrices
        origin_vector = -1*origin_data[0:3]
        unit_vector = np.cross(planar_vectors[0,:],planar_vectors[1,:])
        unit_vector = unit_vector / np.linalg.norm(unit_vector)
        

        changed_data = np.zeros(data.shape)
        changed_data[:,:] = data[:,:]

        if type_of_change == 'symmetric':
            theta_sym = math.radians(angle_change - angle_stock) / 2.0
            print('SYMMETRIC ANGLE CHANGE')

            data_to_change1, points_rotated1 = AffineTransformation(ds1, -1*theta_sym, data, unit_vector, origin_vector)
            data_to_change2, points_rotated2 = AffineTransformation(ds2, theta_sym, data, unit_vector, origin_vector)
            changed_data = ChangeDataPoints(data_to_change1, changed_data, points_rotated1)
            changed_data = ChangeDataPoints(data_to_change2, changed_data, points_rotated2)
            return changed_data
        
        elif type_of_change == 'main_branch':
            # print('MAIN BRANCH ANGLE CHANGE')

            theta_mb = math.radians(angle_change - angle_stock)
            data_to_change_mb, points_rotated_mb = AffineTransformation(ds2, theta_mb, data, unit_vector, origin_vector)
            changed_data = ChangeDataPoints(data_to_change_mb, changed_data, points_rotated_mb)
            return changed_data
        
        elif type_of_change == 'side_branch':
            # print('SIDE BRANCH ANGLE CHANGE')

            theta_sb = math.radians(angle_change - angle_stock)
            data_to_change_sb, points_rotated_sb = AffineTransformation(ds1, -1*theta_sb, data, unit_vector, origin_vector)
            changed_data = ChangeDataPoints(data_to_change_sb, changed_data, points_rotated_sb)
            return changed_data
        
        else:
            msg = 'pass the string symmetric, main_branch, or side_branch'
            raise Exception(msg)
    else:
        print('please pass a different angle than the original value. Will not shift this value')
        return data




def ShiftBranch(branchID, shift_dict, data, maintain_angle = False, sphere_check=False, LAD_LCx_a = None, diams = None):
    vectors, origin = AngleCalculator(data)
    branch_lengths = BranchLengthCalculator(data)

    def DistanceShiftCorrection(branchID, shift_dict, branch_lengths):
        if branchID == 11:
            fraction_shift = shift_dict[str(branchID)] + shift_dict['10'] - branch_lengths[5,1]
        elif branchID == 9:
            fraction_shift = shift_dict[str(branchID)] + shift_dict['8'] - branch_lengths[2,1]
        else:
            fraction_shift = shift_dict[str(branchID)]
        return fraction_shift


    def CalcIndex(fraction_shift, coords, bifurc_index = False):
        nonlocal up_dist
        nonlocal down_dist

        # calculate the distance of the new loop relative to the current origin
        if fraction_shift > up_dist and bifurc_index == False: # shift downstream
            new_dist = abs(fraction_shift - up_dist)
        elif bifurc_index == True:
            new_dist = fraction_shift
        else: # shift upstream
            new_dist = fraction_shift
        
        index = 0
        x0, y0, z0 = coords[0,0:3]
        dist_prev = 0.0
        
        for i in range(len(coords)):
            x, y, z = coords[i,0:3]
            dist_current = dist_prev + np.linalg.norm([x-x0, y-y0, z-z0])
            if abs(dist_current - new_dist) < abs(dist_prev - new_dist):
                index = i
            dist_prev = dist_current
            x0, y0, z0 = x, y, z
        
        return index


    def UpdateBranchData(data, fraction_shift, coords, updown):
        nonlocal up_dist
        nonlocal down_dist
        nonlocal branchID

        bifurc_data = ReadBifurcPoints(data)

        bifurc_do_data = bifurc_data[np.in1d(bifurc_data[:, 3], updown[1])][:,:]
        bifurc_do_length = CalcDistances(bifurc_do_data[:,0:3])

        index_in_updown = CalcIndex(fraction_shift, coords)
        index_new_orig = np.argwhere(data[:,0:3] == coords[index_in_updown, 0:3])[0,0]

        if fraction_shift < up_dist:
            # shifting upstream
            index_old_orig = np.argwhere(data[:,0:3] == coords[-1, 0:3])[0,0]
    
            # replace the original bifurcation id values with downstream values
            if branchID == 10 or branchID == 8 or branchID == 5:
                for i in range(index_new_orig, index_old_orig+1):
                    data[i,4] = updown[1]
                    data[i,3] = updown[1]
                data[int(bifurc_do_data[0,-1]):int(bifurc_do_data[-1,-1])+1,3] = updown[1]
            else:
                for i in range(index_new_orig, index_old_orig+1):
                    data[i,4] = updown[1]
                    data[i,3] = updown[1]
                data[int(bifurc_do_data[0,-1]):int(bifurc_do_data[-1,-1])+2,3] = updown[1]
        else:
            # shifting downstream
            index_old_orig = np.argwhere(data[:,0:3] == coords[0, 0:3])[0,0]

            for i in range(index_old_orig, index_new_orig):
                data[i,4] = updown[0]
                data[i,3] = updown[0]

        # reassign bifurcation points downstream
        bifurc_index_new = CalcIndex(bifurc_do_length, data[np.in1d(data[:, 4], updown[1])][:,0:3], True)
        data[index_new_orig:index_new_orig+bifurc_index_new+1,3] = -1

        bifurc_data = ReadBifurcPoints(data)

        return data


    def TranslationOperation(fraction_shift, coords, data_to_change, origin):
        index = CalcIndex(fraction_shift, coords)
        
        nonlocal up_dist
        nonlocal down_dist
        nonlocal branchID

        if fraction_shift > up_dist:
            vector_shift = coords[index,0:3] - origin[branchID*2, 0:3]
        else:
            vector_shift = coords[index,0:3] - origin[branchID*2, 0:3]

        translation_matrix = np.array([[1, 0, 0, vector_shift[0]], [0, 1, 0, vector_shift[1]],
                                        [0, 0, 1, vector_shift[2]], [0, 0, 0, 1]])
        
        X = data_to_change[:,0:3].T
        B = np.r_[X, [np.ones(len(data_to_change))]]
        points_translated = translation_matrix @ B
        points_translated = points_translated[0:-1,:].T # remove the final row with the constant 1 values

        changed_data = np.zeros(data.shape)
        changed_data[:,:] = data[:,:]
        changed_data = ChangeDataPoints(data_to_change, changed_data, points_translated)
        return changed_data


    def MaintainAngle(fraction_shift, coords, ds, changed_data, branchID, mbID, sphere_check, LAD_LCx_a, diams = None):
        vectors_new, origin_unused = AngleCalculator(changed_data)

        nonlocal vectors
        nonlocal up_dist
        nonlocal down_dist
        nonlocal origin

        index = CalcIndex(fraction_shift, coords)
        vector_mainbranch = vectors_new[mbID, 0:3]
        vector_sidebranch = vectors_new[branchID, 0:3]

        angle_original = vectors[branchID,5]
        angle_new = math.degrees(np.arccos(np.dot(vector_mainbranch,vector_sidebranch)/np.linalg.norm(vector_sidebranch) 
                                    / np.linalg.norm(vector_mainbranch)))
        
        # print('new angle after shifting:{0:0.2f}  prior angle to maintain: {1:0.2f}'.format(angle_new, angle_original))

        if np.abs(angle_new - angle_original) > 0.1: #i.e. the angle is significantly different
            theta = math.radians(angle_original - angle_new)

            origin_vector = -1*coords[index, 0:3]
            unit_vector = np.cross(vector_mainbranch, vector_sidebranch)
            unit_vector = unit_vector / np.linalg.norm(unit_vector)

            data_to_change, points_rot = AffineTransformation(ds, theta, changed_data, unit_vector, origin_vector)
            changed_data_angle = ChangeDataPoints(data_to_change, changed_data, points_rot)

            # Make sure that the shifted points are within the sphere
            if sphere_check == True:
                changed_data_angle = SphereCheck(changed_data_angle, vector_mainbranch, branchID, LAD_LCx_a)

            return changed_data_angle
        else:
            # Make sure that the shifted points are within the sphere
            if sphere_check == True:
                changed_data_angle = SphereCheck(changed_data, vector_mainbranch, branchID, LAD_LCx_a)
                return changed_data_angle
            
            return changed_data
    



    try:
        ds = DownstreamPoints(branchID)
        updown = SideMainBranchRelations(branchID)
    except KeyError:
        msg = 'Branch ' + str(branchID) + ' cannot be shifted. Please pass branchID 5, 8, 9, 10, or 11 for the LCA.'
        raise Exception(msg)    

 
    data_to_change = DataToChange(ds, data, 4)
    up_coords = DataToChange([updown[0]], data, 4)[:,0:3]
    down_coords = DataToChange([updown[1]], data, 4)[:,0:3]

    up_dist = branch_lengths[(updown[0]), 1]
    down_dist = branch_lengths[(updown[1]), 1]

    fraction_shift = DistanceShiftCorrection(branchID, shift_dict, branch_lengths)
    

    if down_dist + up_dist >= fraction_shift > up_dist:    #0.0: #i.e. negative
        print('shifting downstream')
        changed_data = TranslationOperation(fraction_shift, down_coords, data_to_change, origin)
        changed_data = UpdateBranchData(changed_data, fraction_shift, down_coords, updown)
        if maintain_angle == True:
            print('maintaining angle for branch {0}'.format(branchID))
            changed_data_angle = MaintainAngle(fraction_shift, down_coords, ds, changed_data, 
                                               branchID, updown[1], sphere_check, LAD_LCx_a, diams)
            return changed_data_angle
        else:
            return changed_data
        
    elif fraction_shift <= up_dist:
        print('shifting upstream')
        changed_data = TranslationOperation(fraction_shift, up_coords, data_to_change, origin)
        changed_data = UpdateBranchData(changed_data, fraction_shift, up_coords, updown)
        if maintain_angle == True:
            print('maintaining angle for branch {0}'.format(branchID))
            changed_data_angle = MaintainAngle(fraction_shift, up_coords, ds, changed_data, 
                                               branchID, updown[1], sphere_check, LAD_LCx_a, diams)
            return changed_data_angle
        else:
            return changed_data
    else:
        msg = ('pass a valid distance for shifting the origin, less than the upstream + downstream length of ' \
                + str(up_dist+down_dist))
        print(msg)
        return 'python'




def ChangeBranchLength(branchID, new_dist, data):
    data_to_change = np.zeros(data.shape)
    data_to_change = data[:,:]
    
    if MainBranchCheck(int(branchID)) == False or branchID == 1 or branchID == 4 or branchID == 7:
        branch_data = data[np.in1d(data[:, 4], branchID)]
        branch_dists = BranchLengthCalculator(data)
        base_dist = branch_dists[np.in1d(branch_dists[:, 0], branchID)][0,-1]

        print(branchID, 'new distance', new_dist, 'original distance', base_dist)
        if base_dist > new_dist:
            print('shortening vessel length')
            index = 0
            dist_prev = 0

            if branchID == 1:
                loop_range = range(len(branch_data)-1, -1, -1)
                x0, y0, z0 = branch_data[-1,0:3]
            else:
                loop_range = range(len(branch_data))
                x0, y0, z0 = branch_data[0,0:3]

            for i in loop_range:
                x, y, z = branch_data[i,0:3]
                dist_current = dist_prev + np.linalg.norm([x-x0, y-y0, z-z0])
                if abs(dist_current - new_dist) < abs(dist_prev - new_dist):
                    index = i
                dist_prev = dist_current
                x0, y0, z0 = x, y, z
            
            if branchID == 1:
                global_start = np.argwhere(data[:,0:3] == branch_data[0, 0:3])[0,0]
                global_end = np.argwhere(data[:,0:3] == branch_data[index, 0:3])[0,0]
            else:
                global_start = np.argwhere(data[:,0:3] == branch_data[index, 0:3])[0,0]
                global_end = np.argwhere(data[:,0:3] == branch_data[-1, 0:3])[0,0]

            data_to_change = np.delete(data_to_change, np.arange(global_start, global_end+1), axis=0)


            return data_to_change
        
        elif base_dist < new_dist and branchID != 1:
            print('increasing vessel length')

            inc_dist = new_dist - base_dist
            delta = np.linalg.norm(branch_data[-1,0:3] - branch_data[-2,0:3])
            n_points = int(inc_dist/delta)
            points_new = np.zeros((n_points, 6))
            extend_vector = branch_data[-1,0:3] - branch_data[-2,0:3]  # intentially offset from prior point -2
            vector_length = np.linalg.norm(extend_vector)

            final_point = branch_data[-1,0:3] + extend_vector/vector_length*inc_dist # unit vector * new distance1
            
            new_y = np.vstack([branch_data[-10,1:3], branch_data[-1,1:3], final_point[1:]])
            new_x = np.hstack([branch_data[-10,0], branch_data[-1,0], final_point[0]])

            sorted = np.argsort(new_x)
            x_sorted = new_x[sorted]
            y_sorted = new_y[sorted]

            xx = np.linspace(branch_data[-1,0], final_point[0], n_points)
            bspl1 = make_interp_spline(x_sorted, y_sorted, k=3, bc_type='natural')
            yy = np.asarray([*bspl1(xx)])


            for i in range(n_points):
                points_new[i,0] = xx[i]
                points_new[i,1:3] = yy[i,:]
                points_new[i,3] = branchID
                points_new[i,4] = branchID
                points_new[i,5] = branch_data[-1,-1] #assume the radius is the same as the last index

            global_insert_index = np.argwhere(data[:,0:3] == branch_data[-1, 0:3])[0,0] + 1
            data_to_change = np.insert(data_to_change, [global_insert_index], points_new, axis = 0)

            branch_data1 = data_to_change[np.in1d(data_to_change[:, 4], branchID)]
            
            return data_to_change
        else:
            print('pass a distinctly length that is different from the current branch length')
            return 'python'
    else:
        msg = str(branchID) + ' Cannot change the branch length for a main branch. Only applies to side and terminal branches'
        print(msg)
        return 'python'




def Tortuosity(branchID, data, n_inflections, fraction_length, fraction_amplitude):
    
    def RotatePoints(ds, theta, rot_vector, origin_vector):
        x_rot, y_rot, z_rot = rot_vector[0], rot_vector[1], rot_vector[2]             # vector normal to rotation plane
        x_orig, y_orig, z_orig = origin_vector[0], origin_vector[1], origin_vector[2] # point the rotation is about

        rotation_matrix = RotationMatrix(theta, x_rot, y_rot, z_rot)
        points_rotated  = PointsRotation(rotation_matrix, x_orig, y_orig, z_orig, ds)

        return points_rotated
    

    def TranslatePoints(origin, new_origin, data_to_change):
        vec_move = new_origin - origin

        translation_matrix = np.array([[1, 0, 0, vec_move[0]], [0, 1, 0, vec_move[1]], [0, 0, 1, vec_move[2]], [0, 0, 0, 1]])
        
        X = data_to_change[:,0:3].T
        B = np.r_[X, [np.ones(len(data_to_change))]]
        points_translated = translation_matrix @ B
        points_translated = points_translated[0:-1,:].T # remove the final row with the constant 1 values

        return points_translated


    def RotatedPlane(curve_points):
        V = curve_points[-1] - curve_points[0]              # Step 1: Calculate the vector V from the first to the last point
        M = np.mean(curve_points, axis=0)                   # Step 2: Find the midpoint M of the curve
        U_vectors = curve_points - M                        # Step 3: Calculate the vectors from the midpoint to each point on the curve
        dot_products = np.dot(U_vectors, V)                 # Step 4: Calculate the dot product between each vector and V
        closest_vector = U_vectors[np.argmin(dot_products)] # Step 5: Find the vector with the smallest dot product
        cross_product = np.cross(V, closest_vector)         # Step 6: Calculate the cross product between V and the closest vector
        N = cross_product / np.linalg.norm(cross_product)   # Step 7: Normalize the cross product, unit normal vector N
        plane = (V, N)                                      # Step 8: Define the plane by the vector V and the normal vector N

        return plane, M


    ################################## Extract the branch data
    print('__________ Imposing tortuosity on branch', branchID, 'with N =', n_inflections, '__________')
    branch_data = data[np.in1d(data[:, 4], branchID)]
    curve_points = branch_data[:,0:3]
    n_points_curve = len(curve_points)

    if n_inflections == 0:
        print('passing, no inflections')
        return data, 0
    
    
    ################################## Define the rotated plane that exists on the start and end of the prescribed branch.
    plane, M = RotatedPlane(curve_points)
    V, N = plane

    plane_normal = np.cross(V / np.linalg.norm(V), N / np.linalg.norm(N))
    plane_d_val  = -1*np.dot(plane_normal, curve_points[0,:])
    path_length = np.linalg.norm(V)
    path_vector = (curve_points[-1,:] - curve_points[0,:]) / np.linalg.norm(curve_points[-1,:] - curve_points[0,:])


    ################################## Define indices to prescribe the sine function on
    num_of_indices= int(n_points_curve*fraction_length)
    starting_index= round((n_points_curve - num_of_indices)/2)
    ending_index  = starting_index + num_of_indices
    derivative_sine = np.zeros(num_of_indices)

    # define the baseline points for the sine function, which have not been transformed
    xs = np.linspace(curve_points[0,0] , curve_points[0,0] - path_length, n_points_curve)
    ys = np.zeros(n_points_curve)
    zs = np.full(n_points_curve,curve_points[0,2])


    ################################## Calculate the dependent values of the sine curve
    if num_of_indices == n_points_curve:
        ys = np.sin((xs - xs[starting_index])/path_length*n_inflections*np.pi)*path_length*fraction_amplitude + curve_points[0,1]
        derivative_sine = np.cos((xs - xs[starting_index])/path_length*n_inflections*np.pi)*n_inflections*np.pi*0.15
    else:
        # if the fraction of the sine curve length is less than 1, make the imposed sine curve symmetric in the branch
        for i in range(n_points_curve):
            if i < starting_index or i >= ending_index:
                # assign points at a baseline value, not changing
                ys[i] = curve_points[0,1]
            else:
                ys[i] = np.sin( (xs[i] - xs[starting_index])/path_length/fraction_length*n_inflections*np.pi)*path_length*fraction_amplitude + curve_points[0,1] 
                derivative_sine[i-starting_index] = np.cos((xs[i] - xs[starting_index])/path_length/fraction_length*n_inflections*np.pi)*fraction_amplitude*n_inflections*np.pi/fraction_length

    sine_points = np.vstack((xs, ys, zs)).T
    sine_normal = np.array([0,0,1])



    ################################## Define the angle of the vectors between the peaks of the sine curve 
    np.set_printoptions(precision=4)


    # Step 1: Create a list to store tuples containing the value and index of positive numbers
    positive_numbers_with_indices = [(num, index) for index, num in enumerate(derivative_sine) if num > 0]

    # Step 2: Sort the list of tuples based on the value of positive numbers
    sorted_positive_numbers_with_indices = sorted(positive_numbers_with_indices, key=lambda x: x[0])

    # Step 3: Take the first three tuples from the sorted list (containing value and index)
    three_smallest_positive_numbers_with_indices = sorted_positive_numbers_with_indices[:n_inflections]
    # print(sorted_positive_numbers_with_indices)

    # Step 4: Extract the indices from the tuples and return them
    filtered_arr1 = [index for _, index in three_smallest_positive_numbers_with_indices]
    filtered_arr1 = np.sort(filtered_arr1)

    # determine the vectors to calculate the angles between the peaks of the sine wave. changed based on number of inflections
    if n_inflections == 3:    
        vector1 = sine_points[filtered_arr1[0] + starting_index,:] - sine_points[filtered_arr1[1] + starting_index,:]
        vector2 = sine_points[filtered_arr1[2] + starting_index,:] - sine_points[filtered_arr1[1] + starting_index,:]
    if n_inflections == 2:
        vector1 = sine_points[starting_index,:] - sine_points[filtered_arr1[0] + starting_index,:]
        vector2 = sine_points[filtered_arr1[1] + starting_index,:] - sine_points[filtered_arr1[0] + starting_index,:]
    if n_inflections == 1:
        vector1 = sine_points[starting_index,:] - sine_points[filtered_arr1[0] + starting_index,:]
        vector2 = sine_points[ending_index,:] - sine_points[filtered_arr1[0] + starting_index,:]


    angle_between_peaks = math.acos(np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2))
    print(filtered_arr1, n_points_curve, num_of_indices, starting_index, ending_index)
    obtuse_angle_between_peaks = 180 - math.degrees(angle_between_peaks)
    print('angle between peaks of sine curve', obtuse_angle_between_peaks)


    ################################## transform the points
    axis_of_rot = (np.cross(sine_normal, plane_normal))
    axis_of_rot = axis_of_rot / np.linalg.norm(axis_of_rot)

    angle_between_planes = math.acos(np.dot(sine_normal, plane_normal))
    rotated_points = RotatePoints(sine_points, angle_between_planes, axis_of_rot, curve_points[0,:])
    translated_points = TranslatePoints(rotated_points[0,:], curve_points[0,:], rotated_points)

    translated_vector = (translated_points[-1,:] - translated_points[0,:]) / np.linalg.norm(translated_points[-1,:] - translated_points[0,:])
    angle_between_new_points = math.acos(np.dot(translated_vector, path_vector))

    if branchID == 6 or branchID == 7:
        angle_between_new_points = angle_between_new_points     # for branch 6 this angle is kept as positive
    else:
        angle_between_new_points = -angle_between_new_points    # for all other branches this angle is kept as negative

    rotated_points1 = RotatePoints(translated_points, angle_between_new_points, plane_normal, curve_points[0,:])
    translated_points1 = TranslatePoints(rotated_points1[0,:], curve_points[0,:], rotated_points1)

    ################################## move the original curve points and assign to the original data
    new_curve_points = np.copy(curve_points)

    for i in range(n_points_curve):
        sine_point = translated_points1[i,:]
        curr_point = new_curve_points[i,:]
        
        vector_shift = sine_point - curr_point
        norm_vector = np.abs(np.dot(plane_normal, curr_point) + plane_d_val) / np.linalg.norm(plane_normal) * plane_normal

        # First shift the point to the same location as the sine function, based on its index
        new_curve_points[i,:] += vector_shift
        # Second shift the point based on the length of the normal vector to the original current_point position
        new_curve_points[i,:] += norm_vector

    
    data_to_change = np.copy(data)
    branches_to_change = DataToChange([branchID], data, 4)
    data_to_change = ChangeDataPoints(branches_to_change, data_to_change, new_curve_points)

    return data_to_change, obtuse_angle_between_peaks





def StraightenBranch(branchID, data):

    if branchID == 1:

        data_to_change = np.copy(data)
        branch_data = data[np.in1d(data[:, 4], branchID)]
        points_new = np.zeros((len(branch_data), 3))
        base_length = CalcDistances(branch_data[:,0:3])

        new_vector = branch_data[-1,0:3] - branch_data[0,0:3]

        for j in range(0, len(branch_data)):
            points_new[j,:] = branch_data[0,0:3] + new_vector*(j/len(branch_data))
    
        branches_to_change  = DataToChange([branchID], data, 4)
        data_to_change = ChangeDataPoints(branches_to_change, data_to_change, points_new)
        return data_to_change

    else:
        bifurc_points = ReadBifurcPoints(data) #extract all the point data at bifurcations
        bifurc_line_points = BifurcLines(bifurc_points)

        data_to_change = np.copy(data)

        if MainBranchCheck(int(branchID)) == False or branchID == ( 4 or 7):
            branch_data = data[np.in1d(data[:, 4], branchID)]
            points_new = np.zeros((len(branch_data), 3))
            base_length = CalcDistances(branch_data[:,0:3])

            bifurc_line_points = bifurc_line_points[np.in1d(bifurc_line_points[:, 3], branchID)]
            bifurc_vector = bifurc_line_points[-1,0:3] - bifurc_line_points[0,0:3]
            bifurc_length = np.linalg.norm(bifurc_vector)
            
            for j in range(0, len(branch_data)):
                points_new[j,:] = branch_data[0,0:3] + bifurc_vector/bifurc_length*base_length*(j/len(branch_data))
        else:
            msg = 'Cannot straighten a main branch with this function. Please pass a side branch ID number'
            raise Exception(msg)

        branches_to_change  = DataToChange([branchID], data, 4)
        data_to_change = ChangeDataPoints(branches_to_change, data_to_change, points_new)
        return data_to_change




def TranslationGeneral(origin, new_origin, data, data_to_change):
    vector_shift = new_origin - origin
    
    translation_matrix = np.array([[1, 0, 0, vector_shift[0]], [0, 1, 0, vector_shift[1]],
                                    [0, 0, 1, vector_shift[2]], [0, 0, 0, 1]])
    
    X = data_to_change[:,0:3].T
    B = np.r_[X, [np.ones(len(data_to_change))]]
    points_translated = translation_matrix @ B
    points_translated = points_translated[0:-1,:].T # remove the final row with the constant 1 values
    
    changed_data = np.zeros(data.shape)
    changed_data[:,:] = data[:,:]

    if len(data) != len(data_to_change):
        changed_data = ChangeDataPoints(data_to_change, changed_data, points_translated)
    else:
        changed_data[:,0:3] = points_translated
    
    return changed_data




def AlignBifurcationPoints(data):
    bifurc_points = ReadBifurcPoints(data) #extract all the point data at bifurcations
    bifurc_line_points = BifurcLines(bifurc_points)

    # the 4th column contains the branchIDTmp, the 5th column contains the tuple index for the original array
    data_to_change = np.zeros(data.shape)
    data_to_change = data[:,:]

    branches = set(data[:,4])
    
    for i in branches:
        if MainBranchCheck(int(i)) == False or i == (4 or 7):
            branch_data = bifurc_points[np.in1d(bifurc_points[:, 3], i)]
            points_new = np.zeros((len(branch_data), 3))
    
            points_new[0,:] = branch_data[0,0:3]
            bifurc_vector = branch_data[-1,0:3] - points_new[0,:]

            if i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7:
                bifurc_vector = bifurc_line_points[np.in1d(bifurc_line_points[:, 3], i)][-1,0:3]- points_new[0,:]

            for j in range(1, len(branch_data)):
                points_new[j,:] = points_new[0,:] + bifurc_vector*j/len(branch_data)

            data_to_change = ChangeDataPoints(branch_data, data_to_change, points_new, False)
    
    return data_to_change



def EvenSpacing(centerline_data, h_space, diam_dict, branches_to_smooth = None, phase3 = False):
    """
    The values for the branchIDs are reassigned based on main/side branch relation.
    This is used to define the evenly spaced points module and radii.
    Doing so improves the union operation
    """

    def AlignSideBranchMainBranch(data):
        data_changing = np.copy(data)

        branches = set(data_changing[:,3])
        print('branches in set', branches)

        for i in branches:
            ds_val = DownstreamPoints(i)
            print(ds_val, i)
            upstream = UpstreamBranchRelationsSpaced(ds_val[0])

            if np.any(data_changing[:,-1] == upstream):
                branch_index_orig = [j for j in range(len(data_changing)) if data_changing[j,-1] == i][0]
                upstream_indices = [j for j in range(len(data_changing)) if data_changing[j,-1] == upstream]
                origin = data_changing[branch_index_orig,0:3]

                norms_orig = np.linalg.norm(data_changing[upstream_indices,0:3] - origin, axis = 1)

                smallest_norm_index = np.argmin(norms_orig)
                index_of_smallest_orig = upstream_indices[smallest_norm_index]
                new_origin = data_changing[index_of_smallest_orig, 0:3]

                data_to_change = DataToChange(ds_val, data_changing, 3)
                data_changing = TranslationGeneral(origin, new_origin, data_changing, data_to_change)


        
        return data_changing

    def RedefineMainBranch(data):
        branches = set(data[:,4])
        data_redef = data[:,:]

        for i in branches:
            mb_check = MainBranchCheck(int(i))
            if mb_check != False and mb_check != i:
                data_redef[:,4][data_redef[:,4]==i] = mb_check

        return data_redef

    def remove_duplicate_rows(arr):

        mylist = list(arr)

        unique_arr = []
        for x in mylist:
            if  np.any(unique_arr == x) == False:
                unique_arr.append(x)
            else:
                print('duplicate found', x)

        list_of_lists = [list(arr) for arr in unique_arr]
        array_converted = np.array(list_of_lists)
        return array_converted


    centerline_ad = AlignBifurcationPoints(centerline_data)
    centerline_mod = RedefineMainBranch(centerline_ad)


    branches = set(centerline_mod[:,4])
    NoP = len(centerline_mod)
    centerline_spaced = []
  

    # perform spline operation on each branchID separately
    for i in branches:
        Xs, Ys, Zs = [], [], []
        
        for j in range(NoP):
            if int(centerline_mod[j,4]) == i:
                Xs.append(centerline_mod[j,0])
                Ys.append(centerline_mod[j,1])
                Zs.append(centerline_mod[j,2])


        # Check for and remove duplicate points
        coordinates_branch = np.column_stack((Xs, Ys, Zs))
        unique_coords = remove_duplicate_rows(coordinates_branch)

        Xs_u, Ys_u, Zs_u = unique_coords[:,0], unique_coords[:,1], unique_coords[:,2]


        if len(Xs_u) != len(Xs):
            print('fixing coordinates due to duplicate values')
            Xs, Ys, Zs = [], [], []
            Xs, Ys, Zs = Xs_u, Ys_u, Zs_u
      

        # calculate the length of the given segment
        diff = np.diff([Xs,Ys,Zs], axis = 1)
        d_s  = [np.linalg.norm(diff[:,k]) for k in range(len(Xs) -1)]
        total_length = sum(d_s)
        nSpoints = int(total_length / h_space)
        
        ideal_s = np.array([nSpoints-np.sqrt(2*nSpoints),nSpoints+np.sqrt(2*nSpoints)])

        if not phase3:
            if np.any(branches_to_smooth == i):
                s = np.mean(ideal_s)
                print(i, 'smoothing', s)
            else:
                s = 5
                print(i, 'less significant smoothing', s)
        else:
            s = 0.75    # in phase 3, less aggressive smoothing is used to avoid misrepresenting the imposted tortuosity

        print(i, 'phase 3 smoothing', s)

        k_order = 3 # spline order, 3 is ideal and should be <= 5
        nest = -1   # estimate of number of knots needed (-1 = maximal)

        tckp,u = splprep([Xs,Ys,Zs],s=s,k=k_order)
        xs,ys,zs = splev(np.linspace(0,1,nSpoints),tckp)
        
        x = np.reshape(xs, (nSpoints,1))
        y = np.reshape(ys, (nSpoints,1))
        z = np.reshape(zs, (nSpoints,1))
        b = np.full((nSpoints,1), i)

        for k in range(nSpoints):
            centerline_spaced.append([x[k][0], y[k][0], z[k][0],i])
    
    centerline_spaced = np.array(centerline_spaced)
    centerline_spaced = centerline_spaced.reshape(-1, centerline_spaced.shape[-1])

    # assign the first point of a branch to be the same as the main branch it coincides with
    centerline_spaced = AlignSideBranchMainBranch(centerline_spaced)

    radii = np.zeros((len(branches),3))
    radii_mb = []
    sb_orig_list = []
    j = 0

    bigger_branch = 'lad'
    print('the bigger diam branch is ', bigger_branch)

    space_branch_lengths = BranchLengthCalculator(centerline_spaced, True)


    for i in branches:
        radii_original = centerline_mod[np.in1d(centerline_mod[:, 4], i)]
        radii[j,:] = np.array([radii_original[0,5], radii_original[-1, 5], i])
        
        # Set the end values for the MB 1 and 5 based on the average values from most simulations
        # I will keep this up until phase 1 is over, so that the simulations values are more comparable
        # In future simulations, this should be adjusted to match the actual smallest values (0.091 for LAD, 0.077 LCx)


        if str(int(i)) in diam_dict:
                radii[j,0] = diam_dict[str(int(i))] / 10 / 2 # converting from mm to cm, and from diameter to radius


        # # determine the end radii values for each branch
        slope = utils.RadiiTaperScale(i)

        radii[j,1] = radii[j,0] + slope * space_branch_lengths[np.in1d(space_branch_lengths[:, 0], i)][0][-1]

        if i == 1 or i == 5:
            # determine the main branch radii values at the bifurcations

            mb_coords = centerline_spaced[np.in1d(centerline_spaced[:, 3], i)][:,0:3]
            total_length = CalcDistances(mb_coords)
            f = interp1d(np.array([0,total_length]), np.array([radii[j,0], radii[j,1]]), fill_value = 'extrapolate') 

                
            if i == 1:
                sb_IDS = [5,8,9]
            else:
                sb_IDS = [10,11]


            for sb in sb_IDS:
                if np.all(np.in1d(centerline_spaced[:, 3], sb) == False) == True:
                    pass
                else:
                    sb_origin = centerline_spaced[np.in1d(centerline_spaced[:, 3], sb)][0,0:3]
                    sb_orig_list.append(np.asarray(sb_origin))

                    index = np.argwhere(mb_coords == sb_origin)[0,0]
                    sb_coords = mb_coords[0:index+1,0:3]
                    sb_length = CalcDistances(sb_coords)
                    
                    if sb == 5 and i == 1:
                        # If this is the main branch (LAD LCx)
                        # correct the starting radius based on the length of the upstream portion of the model.
                        # That way the typically larger radius of the LMB is captured naturally, based on the assigned
                        # and randomization of the LAD/LCx diameter is preserved
                        print('correcting the starting radius of the main branch, lad')

                        slope = utils.RadiiTaperScale(1)

                        radii[0,0] = radii[0,0] - slope * sb_length
                        radii[0,1] = radii[0,0] + slope * total_length

                        f = interp1d(np.array([0,total_length]), np.array([radii[0,0], radii[0,1]]), fill_value = 'extrapolate') 
                    
                    elif sb == 1 and i == 5:
                        print('correcting the starting radius of the main branch, lcx')
                        print(sb_length, sb, i, index)
                        # print(mb_coords)

                        slope = utils.RadiiTaperScale(5)

                        radii[1,0] = radii[1,0] - slope * sb_length
                        radii[1,1] = radii[1,0] + slope * total_length

                        f = interp1d(np.array([0,total_length]), np.array([radii[1,0], radii[1,1]]), fill_value = 'extrapolate')

                    radii_val = f(sb_length)
                    radii_mb.append(([np.asarray(radii_val), i, sb]))          
        j += 1


    print('interpolated radii')
    print(radii)

    radii_mb = np.hstack((np.asarray(radii_mb), np.asarray(sb_orig_list)))

    print('MB radius at start of a given SB')
    print(radii_mb)

    return centerline_spaced, radii, radii_mb




def CorrectRadius(radii, radii_mb, space_branch_lengths):
    radii_fixed = np.zeros((np.shape(radii)))
    radii_fixed[:,:] = np.copy(radii)
    scale_factor = 0.85

    # Correct the side branch radius values to prevent them from being larger than the main branch at their origin
    # do this for distal branches, not the upstream ones (lad/lcx, i.e. the first two rows)
    for i in range(2, len(radii)):
        radii_fixed[i,0] = radii_mb[i-1,0] * scale_factor

        branchID = radii_mb[i-1,2]            
        branch_length = space_branch_lengths [i,-1]

        slope = utils.RadiiTaperScale(branchID)

        radii_fixed[i,1] = radii_fixed[i,0] + slope * branch_length

    
    return radii_fixed





def RemoveBranches(branches_to_remove, data, column):
    data_skimmed = data[:,:]
    for i in branches_to_remove:
        data_skimmed = data_skimmed[np.logical_not(data_skimmed[:,column] == i)]
    return data_skimmed

