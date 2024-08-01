import pandas as pd
import numpy as np

def divide_segment(length, number_points):
    segment_length = length / (number_points+1)
    return [segment_length + (n * segment_length) for n in range(number_points)]


def a1_grid_points(sensor_pts_xyz_arr):
    irrad_x_unique = pd.Series(sensor_pts_xyz_arr[:, 0]).unique().shape[0]
    irrad_z_unique = pd.Series(sensor_pts_xyz_arr[:, 2]).unique().shape[0]
    cell_dimension = round(sensor_pts_xyz_arr[:, 0][1] - sensor_pts_xyz_arr[:, 0][0],3)
    x_position = (irrad_x_unique / 2) * cell_dimension
    y_position = sensor_pts_xyz_arr[:, 1][0]
    z_position = (irrad_z_unique / 2) * cell_dimension
    grid_point = [x_position, y_position, z_position]
    return np.array([grid_point])

def a2_grid_points_count(sensor_pts_xyz_arr, x_n_points, z_n_points):
    cell_dimension = round(sensor_pts_xyz_arr[:, 0][1] - sensor_pts_xyz_arr[:, 0][0],3)
    y_position = sensor_pts_xyz_arr[:, 1][0]
    
    irrad_x_unique = pd.Series(sensor_pts_xyz_arr[:, 0]).unique().shape[0]
    x_points = divide_segment(irrad_x_unique, x_n_points)
    x_points = [x * cell_dimension for x in x_points]
    
    irrad_z_unique = pd.Series(sensor_pts_xyz_arr[:, 2]).unique().shape[0]
    z_points = divide_segment(irrad_z_unique, z_n_points)
    z_points = [z * cell_dimension for z in z_points]
    
    grid_points = []
    
    for x in x_points:
        for z in z_points:
            grid_points.append([x,y_position,z])
    
    return np.array(grid_points)

def a2_grid_points_distance(sensor_pts_xyz_arr, distance):
    cell_dimension = round(sensor_pts_xyz_arr[:, 0][1] - sensor_pts_xyz_arr[:, 0][0],3)
    y_position = sensor_pts_xyz_arr[:, 1][0]
    irrad_x_unique = pd.Series(sensor_pts_xyz_arr[:, 0]).unique().shape[0]
    x_length = irrad_x_unique * cell_dimension
    x_n_points = int(x_length // distance)
    if x_n_points==0:
        print("specified distance is too small in the x dimension, reverting to a single point")
        x_n_points=1
    irrad_z_unique = pd.Series(sensor_pts_xyz_arr[:, 2]).unique().shape[0]
    z_length = irrad_z_unique * cell_dimension
    z_n_points = int(z_length // distance)
    if z_n_points==0:
        print("specified distance is too small in the z dimension, reverting to a single point")
        z_n_points=1
    
    return a2_grid_points_count(sensor_pts_xyz_arr, x_n_points, z_n_points)
    
    
def a4_grid_points(sensor_pts_xyz_arr, num_points=200, lambda_param=0.35, margin=0.25, seed=42):
    np.random.seed(seed)
    
    cell_dimension = round(sensor_pts_xyz_arr[:, 0][1] - sensor_pts_xyz_arr[:, 0][0],3)
    y_position = sensor_pts_xyz_arr[:, 1][0]
    
    irrad_x_unique = pd.Series(sensor_pts_xyz_arr[:, 0]).unique().shape[0]
    x_length = irrad_x_unique * cell_dimension
    
    irrad_z_unique = pd.Series(sensor_pts_xyz_arr[:, 2]).unique().shape[0]
    z_length = irrad_z_unique * cell_dimension
    x_points = np.linspace(0, x_length - margin, num_points)
    x_points = np.clip(x_points, 0 + margin, x_points - margin)
    
    z_points = np.random.exponential(scale=1/lambda_param, size=num_points)
    z_points = np.clip(z_points, 0 + margin, z_length - margin)
    
    grid_points = [(x_points[i],y_position,z_points[i]) for i in range(num_points)]
    
    return np.array(grid_points)

def b1_grid_points(building, radiance_surface_key):
    module_centers = []

    modules = building.get_modules(radiance_surface_key)
    for module in modules:
        module_centers.append(building.get_dict_instance([radiance_surface_key,module])['Details']['panelizer_center_pt'])


    return np.array(module_centers)

def b2_grid_points(building, radiance_surface_key):
    cell_centers = []

    modules = building.get_modules(radiance_surface_key)
    for module in modules:
        for c in building.get_dict_instance([radiance_surface_key,module])['CellsXYZ']:
            cell_centers.append(c)


    return np.array(cell_centers)

def generate_grid_points(irrad_sensor_pts_xyz_arr, grid_code, building, radiance_surface_key):
    radiance_surface_key_curly = "{" + radiance_surface_key.replace("_", ";") + "}"
    # build the grid of points that will be used to extract irradiance and calculate power
    if grid_code == "a1":
        grid_points = a1_grid_points(irrad_sensor_pts_xyz_arr)
    elif grid_code == "a2":
        grid_points = a2_grid_points_count(irrad_sensor_pts_xyz_arr, 6, 2)
    elif grid_code == "a3":
        grid_points = a2_grid_points_count(irrad_sensor_pts_xyz_arr, 10, 10)
    elif grid_code == "a4":
        grid_points = a4_grid_points(
            irrad_sensor_pts_xyz_arr, num_points=100, lambda_param=0.45, seed=42
        )
    elif grid_code == "b1":
        grid_points = b1_grid_points(building, radiance_surface_key_curly)
    elif grid_code == "b2":
        grid_points = b2_grid_points(building, radiance_surface_key_curly)
    else:
        print("a1, a2, a3, a4, b1, b2 are the only grid codes that are accepted")
        return None
    
    return grid_points