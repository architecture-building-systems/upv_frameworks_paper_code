import shading_masks as sm
import power_calculations as pc
import global_vars as gv
import grid_points as gp
import numpy as np
from scipy.spatial.distance import cdist
import sys
import glob
import os
import time
import pandas as pd

module_path = "/Users/jmccarty/Data/221205_ipv_workbench/github/IPV_Workbench"
sys.path.insert(0, module_path)
import workbench
from workbench.utilities import general, config_utils, temporal, io
from workbench.manage import manage, host, results_writers

# from workbench.manage import host
from workbench.workflows import workflows
from workbench.simulations import (
    method_iv_solver,
    method_topology_solver,
    method_module_iv,
)
from workbench.visualize import plots as ipv_plot
from workbench.device import temperature


def collect_raw_irradiance(
    pv_cells_xyz_arr, sensor_pts_xyz_arr, sensor_pts_irradiance_arr, method="closest"
):

    cdist_arr = cdist(pv_cells_xyz_arr, sensor_pts_xyz_arr)

    if method == "closest":
        first = cdist_arr.argsort()[:, 0]
        irrad_cell_mean = sensor_pts_irradiance_arr.T[first]
    elif method == "mean":
        first = cdist_arr.argsort()[:, 0]
        second = cdist_arr.argsort()[:, 1]
        third = cdist_arr.argsort()[:, 2]
        irrad_cell_mean = (
            sensor_pts_irradiance_arr.T[first]
            + sensor_pts_irradiance_arr.T[second]
            + sensor_pts_irradiance_arr.T[third]
        ) / 3
    else:
        print(
            "The arg 'method' must be specified as either 'closest' or 'mean' (mean of nearest 3 points). "
            "Defaulting to closest."
        )
        first = cdist_arr.argsort()[:, 0]
        irrad_cell_mean = sensor_pts_irradiance_arr.T[first]

    return irrad_cell_mean.T


def calculate_power_simple(
    irrad_sensor_pts_xyz_arr,
    grid_points,
    G_eff_constant,
    mask_arr,
    mask_ratio,
    module_peak_power,
    module_gamma_ref,
    module_area,
    Tcell=25,
):

    # find the area per point
    sa_per_pt = gv.SURFACE_AREA / grid_points.shape[0]

    # build the basic irradiance grid
    G_eff_global = np.ones_like(mask_arr) * G_eff_constant

    # reduce the global irradiance using the mask and ratio
    G_eff_global = np.where(mask_arr == 1, G_eff_global * mask_ratio, G_eff_global)

    # gather irradiance from the masked array using the closest point(s) in the grid
    G_eff = collect_raw_irradiance(grid_points, irrad_sensor_pts_xyz_arr, G_eff_global)

    Tcell = np.ones_like(G_eff) + Tcell
    # calculate power of a single module
    power = np.vectorize(pc.pv_watts_method)(
        G_eff, Tcell, module_peak_power, module_gamma_ref, I_misc=0
    )

    # divide that by the area of the module to get normalised power at the point(s)
    power_norm = power / module_area

    # multiply this normalised production per point by the surface_area associated to each pt
    # in order to calculate power potential of entire surface
    surface_power = np.sum(power_norm * sa_per_pt)

    return round(surface_power, 2)


# solve initial iv curves
def modified_iv_solver(
    building, surface_curly, sensor_pts_xyz_arr, G_eff_global, Tcell
):
    modules = building.get_modules(surface_curly)
    for module_name in modules:
        module_dict = building.get_dict_instance([surface_curly, module_name])

        pv_cells_xyz_arr = np.array(building.get_cells_xyz(surface_curly, module_name))

        G_eff_global_mod = general.collect_raw_irradiance(
            pv_cells_xyz_arr, sensor_pts_xyz_arr, G_eff_global
        )
        Gmod = G_eff_global_mod.flatten()
        Tcell = np.ones_like(Gmod) + Tcell
        for hoy in building.analysis_period:
            Imod, Vmod = method_module_iv.solve_module_iv_curve_single_step(
                building, Gmod, Tcell, module_dict
            )
            module_dict["Curves"]["Imod"].update({hoy: np.array(Imod).flatten()})
            module_dict["Curves"]["Vmod"].update({hoy: np.array(Vmod).flatten()})

        Gmod = np.sum(
            G_eff_global_mod * module_dict["Parameters"]["param_one_cell_area_m2"]
        )
        module_dict["Yield"]["initial_simulation"]["irrad"][
            building.analysis_period[0]
        ] = np.round(Gmod, 3)


# solve actual power output at different topological levels
def modified_topology_solver(host_object):
    for topology in ["micro_inverter", "string_inverter", "central_inverter"]:
        workflows.run_topology_solver(host_object, topology)
        results_writers.write_building_results_timeseries(host_object, topology)


def calculate_power_complex_cells(
    building, surface, sensor_pts_xyz_arr, mask_arr, G_eff_constant, mask_ratio, Tcell
):
    # build the basic irradiance grid
    G_eff_global = np.ones_like(mask_arr) * G_eff_constant

    # reduce the global irradiance using the mask and ratio
    G_eff_global = np.where(mask_arr == 1, G_eff_global * mask_ratio, G_eff_global)

    # solve the iv curves for the modules
    modified_iv_solver(building, surface, sensor_pts_xyz_arr, G_eff_global, Tcell)

    # calculate power for all three electrical topologies using the G_eff_global
    modified_topology_solver(building)


def calculate_power_complex_modules(
    building,
    surface_curly,
    sensor_pts_xyz_arr,
    mask_arr,
    G_eff_constant,
    mask_ratio,
    Tmod,
):
    # build the basic irradiance grid
    G_eff_global = np.ones_like(mask_arr) * G_eff_constant

    # reduce the global irradiance using the mask and ratio
    G_eff_global = np.where(mask_arr == 1, G_eff_global * mask_ratio, G_eff_global)

    modules = building.get_modules(surface_curly)
    for module in modules:
        # load module_dict
        module_dict = building.get_dict_instance([surface_curly, module])
        # load module parameters
        module_params = module_dict["Parameters"]
        # grab center point
        module_ctr_pt = np.array([module_dict["Details"]["panelizer_center_pt"]])
        # find irradiance in grid
        G_eff_global_mod = collect_raw_irradiance(
            module_ctr_pt, sensor_pts_xyz_arr, G_eff_global
        )
        Gmod = G_eff_global_mod.flatten()
        for hoy in building.analysis_period:
            # calculate iv curve of module
            Imod, Vmod = method_iv_solver.solve_iv_curve(
                module_params, Gmod, Tmod, iv_curve_pnts=1000
            )
            # write this power to the dict of the module
            module_dict["Curves"]["Imod"].update({hoy: np.array(Imod).flatten()})
            module_dict["Curves"]["Vmod"].update({hoy: np.array(Vmod).flatten()})
        Gmod_module = Gmod * module_dict["Parameters"]["param_actual_module_area_m2"]
        module_dict["Yield"]["initial_simulation"]["irrad"].update({hoy: Gmod_module})
    # continue on like in the cell based workflow
    modified_topology_solver(building)


def read_electrical_topology_results(building):
    result_dict = {}
    # get results for each topology
    for result_file in glob.glob(
        os.path.join(building.project.GENERAL_RESULTS_DIR, "*csv")
    ):
        topology_name = "_".join(result_file.split(os.sep)[-1].split("_")[1:3])
        topology_surface_power_wh = (
            pd.read_csv(result_file)["electricity_gen_bulk_1111_0_south_kwh"][0] * 1000
        )
        result_dict[topology_name] = topology_surface_power_wh
    return result_dict


def primary_workflow(
    building,
    radiance_surface_key,
    shape_library,
    coverage_range,
    G_eff_constant,
    grid_codes,
    Tcell,
):
    # initiate result fiel
    # need to make a new version of the radiance key
    radiance_surface_key_simple = (
        radiance_surface_key.replace(";", "_").strip("{").strip("}")
    )
    log_file = f"/Users/jmccarty/Nextcloud/Projects/17_framework/notebooks/results/log_simulations_{G_eff_constant}W_{Tcell}C.txt"
    # load sensor pts
    sensor_pts_xyz_arr = io.load_grid_file(
        building.project, radiance_surface_key_simple
    )[["X", "Y", "Z"]].values

    # get important details about the module being used
    surface = building.get_surfaces()[0]
    module = building.get_modules(surface)[0]
    module_params = building.get_dict_instance([surface, module])["Parameters"]
    # get the peak power
    peak_power = module_params["performance_power_W_ref"]
    # extract the gamma value for the module
    gamma_ref = module_params["performance_temp_coe_gamma_pctC"]
    # get the module area
    module_area = module_params["param_actual_module_area_m2"]

    final_results_dict = {}

    # loop through all of the shapes
    for shape in shape_library:
        # loop through all of the shape coverage options
        for shape_coverage in coverage_range:
            # create key to describe shape for storing data
            shape_key = f"{shape}_{str(int(shape_coverage)).zfill(2)}"

            # load mask
            # mask_arr = sm.generate_mask_arr(
            #     sensor_pts_xyz_arr, shape, coverage_factor=shape_coverage
            # ).flatten()
            mask_arr = sm.load_shading_masks()[shape][shape_coverage].flatten()
            # build mask ratio
            mask_ratio = sm.calculate_ratio(shape, shape_coverage)
            # build the basic irradiance grid
            G_eff_global = np.ones_like(mask_arr) * G_eff_constant
            # reduce the global irradiance using the mask and ratio
            G_eff_global = np.where(
                mask_arr == 1, G_eff_global * mask_ratio, G_eff_global
            )

            # loop through the grid options for the simple power production
            for grid_code in grid_codes:
                current_log_data = pd.read_csv(log_file)
                
                result_key = f"{shape_key}_{grid_code}"
                if grid_code[0] != "c":
                    if len(current_log_data[current_log_data['result_key']==result_key]):
                        pass
                    else:
                        start_time = time.time()
                        grid_points = gp.generate_grid_points(
                            sensor_pts_xyz_arr,
                            grid_code,
                            building,
                            radiance_surface_key_simple,
                        )
                        surface_power_wh = calculate_power_simple(
                            sensor_pts_xyz_arr,
                            grid_points,
                            G_eff_constant,
                            mask_arr,
                            mask_ratio,
                            peak_power,
                            gamma_ref,
                            module_area,
                            Tcell,
                        )
                        final_results_dict[result_key] = round(surface_power_wh, 3)

                        # write_result and log time
                        total_time = round(time.time() - start_time, 2)
                        log_data(log_file, result_key, surface_power_wh, total_time)
                else:
                    
                    if len(current_log_data[current_log_data['result_key']==result_key+"c"]):
                        pass
                    else:
                        module_time_start = time.time()
                        # do module center point calculation
                        if grid_code=="c1":
                            calculate_power_complex_modules(
                                building,
                                radiance_surface_key,
                                sensor_pts_xyz_arr,
                                mask_arr,
                                G_eff_constant,
                                mask_ratio,
                                Tcell,
                            )
                            surface_power_wh_module_dict = read_electrical_topology_results(
                                building
                            )
                            for k,v in surface_power_wh_module_dict.items():
                                sub_result_key = result_key + k[0]
                                final_results_dict[sub_result_key] = round(
                                    v, 3
                                )
                            # write_result and log time
                            total_time_module = round(time.time() - module_time_start, 2)
                            for k, v in surface_power_wh_module_dict.items():
                                sub_result_key = result_key + k[0]
                                log_data(
                                    log_file,
                                    sub_result_key,
                                    v,
                                    total_time_module,
                                )
                        else:
                            cell_time_start = time.time()
                            # do cell calculation
                            # result_key = f"{shape_key}_{grid_code}"
                            calculate_power_complex_cells(
                                building,
                                radiance_surface_key,
                                sensor_pts_xyz_arr,
                                mask_arr,
                                G_eff_constant,
                                mask_ratio,
                                Tcell,
                            )
                            surface_power_wh_cell_dict = read_electrical_topology_results(
                                building
                            )
                            for k, v in surface_power_wh_cell_dict.items():
                                sub_result_key = result_key + k[0]
                                final_results_dict[sub_result_key] = round(
                                    v, 3
                                )
                            # write_result and log time
                            total_time_cell = round(time.time() - cell_time_start, 2)
                            for k, v in surface_power_wh_cell_dict.items():
                                sub_result_key = result_key + k[0]
                                log_data(
                                    log_file,
                                    sub_result_key,
                                    v,
                                    total_time_cell,
                                )

    return final_results_dict


def log_data(LOG_FILE, result_key, surface_power_wh, runtime):
    if os.path.exists(LOG_FILE):
        pass
    else:
        # create file
        first_line = "result_key,surface_power [wh],runtime [sec]\n"
        with open(LOG_FILE, "w") as fp:
            fp.writelines([first_line])

    entry = f"{result_key},{surface_power_wh},{runtime}\n"
    with open(LOG_FILE, "a") as fp:
        fp.write(entry)
        



def multi_func(irradiance, Tcell):
    # because we have restarted our notebook we will reactivate our project by reading in the config path
    # place your config path here
    project_name = f"cactus_framework_study_{irradiance}_{Tcell}"
    project_folder = f"/Users/jmccarty/Desktop/multi_func_sims"
    if os.path.exists(project_folder):
        pass
    else:
        os.mkdir(project_folder)
        
    project_epw = os.path.join("/Users","jmccarty","Downloads", "fluntern_2001-2017.epw")
    config_path = manage.initiate_project(project_folder, project_name, project_epw)

    # config_path = os.path.join(project_folder, "cactus_framework_study.config")
    project_manager = manage.Project(config_path)

    # rerun the setup in order to rebuild the entire project object and its attributes
    project_manager.project_setup()
    
    
    project_manager.edit_cfg_file("management", "host_name", "B1111")
    project_manager.edit_cfg_file("management", "raw_host_file", "D020_solar_glass_B1111_raw.pickle")
    project_manager.edit_cfg_file("analysis", "analysis_period", "4332-4333")
    project_manager.edit_cfg_file("analysis", "device_id", "D020")

    # load building
    building = host.Host(project_manager)

    radiance_surface_key_curly = "{1111;0}"
    shape_library = ["random_squares_small", "random_squares_large"]
    coverage_range = list(np.arange(10, 91, 5))
    grid_codes = ["a1", "a2", "a3", "a4", "b1", "b2", "c1", "c2"]

    primary_workflow(
        building,
        radiance_surface_key_curly,
        shape_library,
        coverage_range,
        irradiance,
        grid_codes,
        Tcell,
    )

