import sv
import vtk
from math import sqrt

class MouseInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, surface=None, picking_keys=None, event_table=None):
        #self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.AddObserver("KeyPressEvent", self.onKeyPressEvent)
        self.AddObserver("CharEvent", self.onCharEvent)
        self.window = None
        self.renderer = None
        self.surface = surface 
        self.event_table = event_table
        self.selected_points = []
        self.selected_node_ids = []
        self.selected_node_id = None
        self.selected_cell_id = None
        self.picking_keys = picking_keys 
        self.picked_actor = None
        self.last_picked_actor = None
        self.method_queue = []
        self.pick_glyph_actor_queue = []

    def select_event(self, renderer, key, obj, event):
        '''Process a select event. 
        '''
        clickPos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)

        # Get selecected geometry. 
        self.picked_actor = picker.GetActor()
        if self.picked_actor == None:
            #self.OnLeftButtonDown()
            return
        position = picker.GetPickPosition()
        cell_id = picker.GetCellId()

        if cell_id == -1: 
            return

        print(" ")
        print("Picked position: {0:g} {1:g} {2:g} ".format(position[0], position[1], position[2]))
        print("Cell id is: {0:d}".format(cell_id))

        min_i = -1
        min_d = 1e5
        min_p = 3*[0.0]
        surface = self.surface
        points = surface.GetPoints()

        for i in range(points.GetNumberOfPoints()):
            p = 3*[0.0]
            points.GetPoint(i,p)
            dx = p[0] - position[0]
            dy = p[1] - position[1]
            dz = p[2] - position[2]
            d = sqrt(dx*dx + dy*dy + dz*dz)
            if d < min_d:
                min_d = d
                min_p[0] = p[0]
                min_p[1] = p[1]
                min_p[2] = p[2]
                min_i = i  

        print("Picked node: {0:d} {1:g} {2:g} {3:g} ".format(min_i, min_p[0], min_p[1], min_p[2]))
        self.selected_node_ids.append(min_i)
        self.selected_node_id = min_i
        self.selected_cell_id = cell_id

        ## Show picked point.
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(1)
        points.SetPoint(0, position[0], position[1], position[2])
        geom = vtk.vtkPolyData()
        geom.SetPoints(points)
        glyphFilter = vtk.vtkVertexGlyphFilter()
        glyphFilter.SetInputData(geom)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyphFilter.GetOutputPort())
        actor = vtk.vtkActor()
        if key == 's':
            actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        elif key == 't':
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        else:
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetPointSize(10)
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)
        self.pick_glyph_actor_queue.append(actor)

        # Get picked face ID.
        data_name = "ModelFaceID" 
        cell_data = surface.GetCellData().GetArray(data_name)
        if cell_data != None:
            face_id = cell_data.GetValue(cell_id)
            print("Picked face: {0:d} ".format(face_id))

        #self.window.Render()

    def onKeyPressEvent(self, renderer, event):
        '''Process a key press event.
        '''
        key = self.GetInteractor().GetKeySym()

        if key == 'u' and len(self.method_queue) != 0:
            last_method = self.method_queue.pop()
            last_method(undo=True) 
            last_actor = self.pick_glyph_actor_queue.pop()
            last_actor.SetVisibility(False)

        if (key in self.picking_keys):
            self.select_event(renderer, key, None, event)

        if self.event_table == None:
            return

        if (key in self.event_table):
            value = self.event_table[key]
            if type(value) == tuple:
                method = value[0]
                data = value[1]
            else:
                method = value
                data = None

            # Call the method with selected node/cell and any data.
            print("###### self.selected_node_id: " + str(self.selected_node_id))
            method(surface=self.surface, node_id=self.selected_node_id, cell_id=self.selected_cell_id, data=data, renderer=self.renderer)

            # Store the last method so we can use it for an undo operation.
            self.method_queue.append(method)

        self.window.Render()

    def onCharEvent(self, renderer, event):
        '''Process an on char event.

        This is used to prevent passing the shortcut key 'w' to vtk which we use
        to write selected results and vtk uses to switch to wireframe display. 
        '''
        key = self.GetInteractor().GetKeySym()
        if (key != 'w'):
            self.OnChar()

def add_line(renderer, pt1, pt2, color=[1.0, 1.0, 1.0], width=2):
    line = vtk.vtkLineSource()
    line.SetPoint1(pt1);
    line.SetPoint2(pt2)
    line.Update()
    polydata = line.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    actor.GetProperty().SetLineWidth(width)
    renderer.AddActor(actor)

def create_segmentation_geometry(renderer, segmentation, color=[1.0, 1.0, 1.0], show_cpts=False, show_center=False):
    ''' Create geometry for the segmentation points and control points.
    '''
    #print("---------- gr.create_segmentation_geometry ----------")
    coords = segmentation.get_points()
    num_pts = len(coords)
    #print("[gr.create_segmentation_geometry] num_pts: {0:d}".format(num_pts))
    if num_pts == 0:
        return

    ## Create segmentation geometry points and line connectivity.
    #
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(num_pts)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(num_pts+1)
    n = 0
    for pt in coords:
        points.SetPoint(n, pt[0], pt[1], pt[2])
        lines.InsertCellPoint(n)
        n += 1
    #_for pt in coords
    lines.InsertCellPoint(0)

    geom = vtk.vtkPolyData()
    geom.SetPoints(points)
    geom.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(geom)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2.0)
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    renderer.AddActor(actor)

    ## Add center point.
    #
    if show_center:
        center = segmentation.get_center()
        #print("gr.create_segmentation_geometry] Center: {0:g} {1:g} {2:g}".format(center[0], center[1], center[2]))
        num_pts = 1
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        pid = points.InsertNextPoint(center)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pid)
        points_pd = vtk.vtkPolyData()
        points_pd.SetPoints(points)
        points_pd.SetVerts(vertices)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(points_pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        renderer.AddActor(actor)
        actor.GetProperty().SetPointSize(5)
        renderer.AddActor(actor)

    ## Add control points.
    #
    if show_cpts:
        try:
            coords = segmentation.get_control_points()
        except:
            coords = []
        num_pts = len(coords)
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for pt in coords:
            pid = points.InsertNextPoint(pt)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)
        #_for pt in coords
        points_pd = vtk.vtkPolyData()
        points_pd.SetPoints(points)
        points_pd.SetVerts(vertices)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(points_pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        actor.GetProperty().SetPointSize(5)
        renderer.AddActor(actor)

def create_path_geometry(renderer, path, line_color=[0.0, 0.6, 0.0], marker_color=[1.0,0.0,0.0], show_points=False):
    ''' Create geometry for the path curve and control points.
    '''
    coords = path.get_curve_points()
    num_pts = len(coords)

    # Create contour geometry points and line connectivity.
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(num_pts)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(num_pts)
    n = 0
    for pt in coords:
        points.SetPoint(n, pt[0], pt[1], pt[2])
        lines.InsertCellPoint(n)
        n += 1
    #_for pt in coords

    geom = vtk.vtkPolyData()
    geom.SetPoints(points)
    geom.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(geom)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2.0)
    actor.GetProperty().SetColor(line_color[0], line_color[1], line_color[2])
    renderer.AddActor(actor)

    ## Show curve points.
    if show_points:
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for pt in coords:
            pid = points.InsertNextPoint(pt)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)
        #_for pt in coords
        points_pd = vtk.vtkPolyData()
        points_pd.SetPoints(points)
        points_pd.SetVerts(vertices)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(points_pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.0, 1.0)
        actor.GetProperty().SetPointSize(5)
        renderer.AddActor(actor)

    ## Add control points.
    coords = path.get_control_points()
    num_pts = len(coords)
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    for pt in coords:
        pid = points.InsertNextPoint(pt)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pid)
    #_for pt in coords
    points_pd = vtk.vtkPolyData()
    points_pd.SetPoints(points)
    points_pd.SetVerts(vertices)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(points_pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(marker_color[0], marker_color[1], marker_color[2])
    actor.GetProperty().SetPointSize(8)
    renderer.AddActor(actor)

#_create_path_geometry(renderer, path)

def convert_ug_to_polydata(mesh):
    '''
    Convert mesh to polydata.
    '''
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(mesh)
    geometry_filter.Update()
    mesh_polydata = geometry_filter.GetOutput()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(mesh_polydata)
    triangle_filter.Update()
    polydata = triangle_filter.GetOutput()
    return polydata 

def create_contour_geometry(renderer, contour):
    """ Create geometry for the contour points and control points.
    """
    coords = contour.get_points()
    num_pts = len(coords)

    ## Create contour geometry points and line connectivity.
    #
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(num_pts)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(num_pts+1)
    n = 0
    for pt in coords:
        points.SetPoint(n, pt[0], pt[1], pt[2])
        lines.InsertCellPoint(n)
        n += 1
    #_for pt in coords
    lines.InsertCellPoint(0)

    geom = vtk.vtkPolyData()
    geom.SetPoints(points)
    geom.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(geom)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2.0)
    actor.GetProperty().SetColor(0.0, 0.6, 0.0)
    renderer.AddActor(actor)

    ## Add center point.
    #
    center = contour.get_center()
    num_pts = 1
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    pid = points.InsertNextPoint(center)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(pid)
    points_pd = vtk.vtkPolyData()
    points_pd.SetPoints(points)
    points_pd.SetVerts(vertices)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(points_pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    actor.GetProperty().SetPointSize(5)
    renderer.AddActor(actor)

    ## Add control points.
    #
    try:
        coords = contour.get_control_points()
        num_pts = len(coords)
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for pt in coords:
            pid = points.InsertNextPoint(pt)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)
        #_for pt in coords
        points_pd = vtk.vtkPolyData()
        points_pd.SetPoints(points)
        points_pd.SetVerts(vertices)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(points_pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        actor.GetProperty().SetPointSize(5)
    except:
        pass
    # renderer.AddActor(actor)

def display(renderer_win):
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(renderer_win)
    # Set the window title.
    renderer_win.Render()
    renderer_win.SetWindowName("SV Python API")
    interactor.Start()

def add_geometry(renderer, polydata, color=[1.0, 1.0, 1.0], line_width=1, wire=False, edges=False, array_name=None, scalar_range=None):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    if array_name == None:
        mapper.SetScalarVisibility(False)
    else:  
        mapper.SetArrayName(array_name)
        mapper.SelectColorArray(array_name)
        mapper.SetScalarVisibility(True)
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SetScalarRange(scalar_range)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    #actor.GetProperty().SetPointSize(5)

    if wire:
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(1.0)
    elif not edges:
        actor.GetProperty().SetLineWidth(line_width)

    if edges:
        actor.GetProperty().EdgeVisibilityOn();
    else:
        actor.GetProperty().EdgeVisibilityOff();

    renderer.AddActor(actor)
    return actor

def add_plane(renderer, center, normal, color=[1.0, 1.0, 1.0], wire=False):
    planeSource = vtk.vtkPlaneSource()
    planeSource.SetCenter(center);
    planeSource.SetNormal(normal)
    planeSource.Update()
    plane_pd = planeSource.GetOutput()
    add_geometry(renderer, plane_pd, color, wire)

def add_glyph_points(renderer, points, color=[1.0, 1.0, 1.0], size=2):
    geom = vtk.vtkPolyData()
    geom.SetPoints(points)
    glyphFilter = vtk.vtkVertexGlyphFilter()
    glyphFilter.SetInputData(geom)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyphFilter.GetOutputPort())
    actor = vtk.vtkActor()
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetPointSize(size)
    actor.SetMapper(mapper)
    renderer.AddActor(actor)

def add_points(renderer, points, color=[1.0, 1.0, 1.0], size=3):
    geom_points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    for pt in points:
        pid = geom_points.InsertNextPoint(pt)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pid)
    points_pd = vtk.vtkPolyData()
    points_pd.SetPoints(geom_points)
    points_pd.SetVerts(vertices)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(points_pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetPointSize(size)
    renderer.AddActor(actor)

def add_sphere(renderer, center, radius, color=[1.0, 1.0, 1.0], wire=False):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], center[1], center[2])
    sphere.SetRadius(radius) 
    sphere.SetPhiResolution(16)
    sphere.SetThetaResolution(16)
    sphere.Update()
    sphere_pd = sphere.GetOutput() 
    return add_geometry(renderer, sphere_pd, color, wire=wire)

def init_graphics(win_width, win_height):
    ''' Create renderer and graphics window.
    '''
    renderer = vtk.vtkRenderer()
    renderer_win = vtk.vtkRenderWindow()
    renderer_win.AddRenderer(renderer)
    renderer.SetBackground(0.8, 0.8, 0.8)
    renderer_win.SetSize(win_width, win_height)
    #renderer_win.Render()
    #renderer_win.SetWindowName("SV Python API")
    print("Create renderer and graphics window.")
    #print("---------- Alphanumeric Keys ----------")
    print("q - Quit")
    #print("s - Select a face.")
    return renderer, renderer_win 

def init_picking(window, renderer, surface, picking_keys, event_table=None):

    # Create a trackball interacter to transoform the geometry using the mouse.
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(window)

    face_ids = surface.GetCellData().GetArray("ModelFaceID")
    if face_ids == None:
        print("No ModelFaceID data.")
    else:
        face_ids_range = 2*[0]
        face_ids.GetRange(face_ids_range, 0)
        min_id = int(face_ids_range[0])
        max_id = int(face_ids_range[1])
        print("Face IDs range: {0:d} {1:d}".format(min_id, max_id))

    # Add the custom style.
    style = MouseInteractorStyle(surface, picking_keys, event_table)
    style.window = window
    style.renderer = renderer
    interactor.SetInteractorStyle(style)
    style.SetCurrentRenderer(renderer)
    return interactor


