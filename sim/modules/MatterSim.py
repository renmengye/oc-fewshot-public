"""MatterSim API.

Authors: 
 - Mengye Ren (mren@cs.toronto.edu)
 - Michael Iuzzolino (michael.iuzzolino@colorado.edu)
"""
from __future__ import division, print_function, unicode_literals
import os

import MatterSim
import math
import numpy as np
import modules.vis_utils

# https://github.com/peteanderson80/Matterport3DSimulator/blob/master/src/driver/driver.py

class MatterSimAPI(object):
    """MatterSim API."""
    def __init__(self, params):
        self.params = self._setup_params(params)
    
    def _setup_params(self, params):
        # Image params
        params['width'] = int(params['width'])
        params['height'] = int(params['height'])
        params['depth'] = bool(params['depth'])
        params['annotate_image'] = bool(params['annotate_image'])
        
        # FOV Params
        params['hfov_RAD'] = math.radians(params['hfov_DEG'])
        params['vfov_RAD'] = params['hfov_RAD'] / params['width'] * params['height']
        params['vfov_DEG'] = np.rad2deg(params['vfov_RAD'])
        
        # STDOUT
        print("Parameters")
        print("----------")
        for param_key, param_val in params.items():
            print(f'{param_key:15s} {param_val}')
        
        return params
    
    def add_to_params(self, new_params):
        for key, val in new_params.items():
            if key in self.params:
                print(f"{key} already exists in self.params!")
            else:
                self.params[key] = val
                
    def _reset_history(self):
        self.history = {
            "movement"  : [],
            "location"  : []
        }
     
    def update_history(self, key, vals):
        self.history[key].append(vals)
        
    def find_intersection_image_ids(self, scene_id, connectivity):
        self._init_sim_enviornment()
        
        image_ids = [ele['image_id'] for ele in connectivity]
        intersection_set = []
        for image_id in image_ids:
            try:
                self.sim.newEpisode([scene_id], [image_id], [0], [0])
                intersection_set.append(image_id)
            except Exception as e:
                pass
        
        self.close()
        
        return intersection_set
    
    def _init_sim_enviornment(self):
        self.sim = MatterSim.Simulator()
        self.sim.setCameraResolution(self.params['width'], self.params['height'])
        self.sim.setCameraVFOV(self.params['vfov_RAD'])
        self.sim.setDepthEnabled(self.params['depth'])
        self.sim.initialize()

    def initialize(self, scene_id, image_id, heading=0, elevation=0, reset_history=True):
        self.scene_id = scene_id
        self._init_sim_enviornment()
        successful_init = True
        try:
            self.sim.newEpisode([scene_id], [image_id], [heading], [elevation])
        except:
            successful_init = True
        
        if reset_history:
            self._reset_history()
           
        return successful_init
        
    def _read_sensor_data(self, state):
        sensor_data = {}
        # Get RGB image
        RGB_img = np.array(state.rgb, copy=False)[:,:, [2,1,0]]
           
        if self.params['annotate_image']:
            RGB_img = modules.vis_utils.annotate_locations_on_img(state.navigableLocations, RGB_img, params=self.params)
        sensor_data['RGB'] = RGB_img.copy()
        
        # Check for depth image
        if self.params['depth']:
            sensor_data['depth'] = np.array(state.depth, copy=False)
        
        return sensor_data
    
    def run(self, sim_step_i, location, image_id, heading_delta, elevation_delta, verbose=False, set_by='image_id'):
        if set_by == 'location':
            self.sim.makeAction([location], [heading_delta], [elevation_delta])
        elif set_by == 'image_id':
            if sim_step_i == 0:
                self.heading = heading_delta
                self.elevation = elevation_delta
            else:
                self.heading += heading_delta
                self.heading %= 2*np.pi
                self.elevation += elevation_delta

            _ = self.initialize(self.scene_id, image_id, self.heading, self.elevation, reset_history=False)
        else:
            assert False, "Invalid set_by argument"
            
        # Get state
        assert len(self.sim.getState()) == 1, "More than 1 state!"
        state = self.sim.getState()[0]
        
        # Update state history
        state_location = np.array([state.location.x, state.location.y, state.location.z])
        self.history['location'].append(state_location)
        
        # Process image
        sensor_data = self._read_sensor_data(state)

        # Stdout
        if verbose:
            stdout_str = f"\n\t[MatterSim] Location (x,y,z): "
            stdout_str += f"({state.location.x:0.6f}, "
            stdout_str += f"{state.location.y:0.6f}, "
            stdout_str += f"{state.location.z:0.6f})"
            print(stdout_str)
        
        return state, sensor_data
    
    def close(self, verbose=True):
        if verbose:
            print("\nClosing MatterportSim.")
        try:
            self.sim.close()
        except Exception as e:
            print(f"Error closing sim. \n{e}")
            