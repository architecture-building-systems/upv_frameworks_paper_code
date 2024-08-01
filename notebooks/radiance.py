"""
radiance_surface_key = project.analysis_active_surface

radiance_project_dir = project.RADIANCE_DIR
scenario_tmy = project.TMY_FILE

analysis_period = project.analysis_analysis_period
skyglow_template_file = project.skyglow_template
n_workers = project.irradiance_n_workers
rflux_rad_params = project.irradiance_radiance_param_rflux
rcontrib_rad_params = project.irradiance_radiance_param_rcontrib
"""

import time
import os
import glob
import sys

def current_time():
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())

def create_log_file(destination_path):
    first_line = "date,scenario,simulation_type,n_workers,n_points,rad_par,runtime [sec]\n"
    with open(destination_path, 'w') as fp:
        fp.writelines([first_line])

class Project:
    def __init__(self, project_dir, scenario, skyglow_template, epw_file, use_accelerad=False, rad_par_res='low'):
        self.TMY_FILE = epw_file
        self.rad_detail = rad_par_res
        self.RADIANCE_DIR = project_dir
        self.analysis_active_surface = scenario
        self.scenario_dir = os.path.join(project_dir, self.analysis_active_surface)
        self.radiance_model = os.path.join(self.scenario_dir, "model")
        self.IRRADIANCE_RESULTS_DIR = os.path.join(self.scenario_dir, "results")
        self.grid_file = glob.glob(os.path.join(self.radiance_model, "grid", "*.pts"))[0]
        self.use_accelerad = use_accelerad
        self.irradiance_use_accelerad = use_accelerad
        self.skyglow_template = skyglow_template
        if sys.platform=='darwin':
            self.irradiance_n_workers = os.cpu_count() - 1
        else:
            if self.use_accelerad==True:
                self.irradiance_n_workers = 1
            else:
                self.irradiance_n_workers = os.cpu_count() - 1        
            
        if self.rad_detail=="low":
            # -ab 6 -ad 25000 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15

            self.irradiance_radiance_param_rflux = "-lw 1.0e-3 -ab 3 -ad 2048"
            self.irradiance_radiance_param_rcontrib = "-ad 256 -lw 1.0e-3 -dc 1 -dt 0 -dj 0"
        elif self.rad_detail=="med":
            self.irradiance_radiance_param_rflux = "-lw 1.0e-5 -ab 6 -ad 4096"
            self.irradiance_radiance_param_rcontrib = "-ad 256 -lw 1.0e-5 -dc 1 -dt 0 -dj 0"
        elif self.rad_detail=="high":
            self.irradiance_radiance_param_rflux = "-lw 1.0e-3 -ab 6 -ad 8192"#"-ab 6 -ad 25000 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15"#"-lw 4.0e-7 -ab 6 -ad 25000"
            self.irradiance_radiance_param_rcontrib = "-ab 1 -ad 8192 -lw 1.0e-3 -dc 1 -dt 0 -dj 0" #"-ab 1 -ad 25000 -as 4096 -c 1 -dc 0.75 -dp 512 -dr 3 -ds 0.05 -dt 0.15 -lr 8 -lw 4e-07 -ss 1.0 -st 0.15"
            #"-ad 256 -lw 4.0e-7 -dc 1 -dt 0 -dj 0"
            
        self.LOG_FILE = os.path.join(project_dir, "log.txt")
        
        for fp_dir in [self.radiance_model, self.IRRADIANCE_RESULTS_DIR]:
            if os.path.exists(fp_dir):
                pass
            else:
                os.makedirs(fp_dir)
        
        
    
    def log(self, runtime, simulation_type):
        
        if simulation_type=='irradiance':
            rad_par = self.irradiance_radiance_param_rcontrib + " " + self.irradiance_radiance_param_rflux
            n_workers = self.irradiance_n_workers
            
            grid_file = glob.glob(os.path.join(self.radiance_model, "grid", "*.pts"))[0]
            # n_points = int(grid_file.split("_")[-1].split("s")[0])
            try:
                n_points = int(grid_file.split("_")[-1].split("s")[0])
            except ValueError:
                with open(grid_file, "r") as fp:
                    n_points = len(fp.readlines())
        entry = f"{current_time()},{self.analysis_active_surface}," \
                f"{simulation_type}," \
                f"{n_workers},{n_points},{rad_par},{runtime}\n"
        if os.path.exists(self.LOG_FILE):
            pass
        else:
            create_log_file(self.LOG_FILE)
            
        with open(self.LOG_FILE, "a") as fp:
            fp.write(entry)
            
    def get_irradiance_results(self):
        self.DIRECT_IRRAD_FILE = os.path.join(self.IRRADIANCE_RESULTS_DIR, "direct.lz4")
        self.DIFFUSE_IRRAD_FILE = os.path.join(self.IRRADIANCE_RESULTS_DIR, "diffuse.lz4")
        
        