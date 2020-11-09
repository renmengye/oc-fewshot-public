"""Simulation Handler API: HabitatSim+MatterportSim

Authors: 
 - Michael Iuzzolino (michael.iuzzolino@colorado.edu)
 - Mengye Ren (mren@cs.toronto.edu)

"""
from __future__ import division, print_function, unicode_literals
import os
import sys
import glob
import json
import math
import numpy as np
import modules.utils

_MAX_DISTANCE = 10

class Episode(object):
    def __init__(self, matterport_sim, habitat_sim, params, rnd, args):
        self.matterport_sim = matterport_sim
        self.habitat_sim = habitat_sim
        self.setup_params(params)
        self._rnd = rnd
        self.args = args
        
        episode_params_to_add = {
            key : self.params[key] for key in ['max_next_loc_rel_heading', 'max_next_loc_rel_elevation']
        }
            
        self.matterport_sim.add_to_params(episode_params_to_add)
    
    def setup_params(self, params):
        self.params = {
            "elevation"                     : {
                "min"    : params.min_elevation_angle, 
                "max"    : params.max_elevation_angle, 
                "delta"  : params.elevation_delta, 
                "prob"   : params.elevation_delta_prob},
            "heading"                       : {
                "delta_mean" : params.heading_delta_mean, 
                "delta_var"  : params.heading_delta_var, 
                "prob"       : params.heading_delta_prob
            },
            "scan_refractory_period"        : params.scan_refractory_period,
            "movement_refractory_period"    : params.movement_refractory_period,
            "movement_prob"                 : params.movement_prob,
            "n_scan_steps"                  : params.n_scan_steps,
            "max_next_loc_rel_heading"      : params.max_next_loc_rel_heading,
            "max_next_loc_rel_elevation"    : params.max_next_loc_rel_elevation
        }

    def get_history(self, key):
        if 'matter' in key.lower():
            return self.matterport_sim.history
        elif 'habitat' in key.lower():
            return self.habitat_sim.history

    def _change_heading(self, heading, mean_shift=0.0):
        if self._rnd.uniform() <= self.params['heading']['prob']:
#             magnitude = self._rnd.normal(loc=self.params['heading']['delta_mean']+mean_shift, 
#                                          scale=self.params['heading']['delta_var'])
            magnitude = self.params['heading']['delta_mean']
            heading = self.heading_scan_direction * magnitude
        return heading
        
    def _change_elevation(self, elevation):
        if self._rnd.uniform() <= self.params['elevation']['prob']:
            elevation = self._rnd.choice([-1, 1]) * self.params['elevation']['delta']
            elevation = min(self.params['elevation']['max'], max(self.params['elevation']['min'], elevation))
        return elevation
        
    def _reset_scan_direction(self):
        self.heading_scan_direction = self._rnd.choice([-1, 1])
    
    def _sample_new_location(self, possible_locations, image_id):
        if self._rnd.uniform() >= self.params['movement_prob']:
            return False, 0, image_id
         
        # Place constraint on rel_heading
        good_locations = []
        for loc_i, loc_obj in enumerate(possible_locations):
            # Constrain positions by relative heading and elevations
            # --------------------------------------------------------------------------------
            if np.abs(loc_obj.rel_heading) > self.params['max_next_loc_rel_heading']:
                continue
            #if np.abs(loc_obj.rel_elevation) > self.params['max_next_loc_rel_elevation']:
            #    continue
            # --------------------------------------------------------------------------------
            
            good_locations.append((loc_i+1, loc_obj))
        
        # Check case for no good next locations
        if len(good_locations) == 0:
            return False, 0, image_id
        
        # Invert distances
        good_position_distances = _MAX_DISTANCE - np.array([ele[1].rel_distance for ele in good_locations])
        
        # Cast to probabilities
        location_probs = modules.utils.softmax(good_position_distances)

        # Randomly sample location given probabilities p
        idxs = range(len(good_locations))
        random_idx = self._rnd.choice(idxs, p=location_probs)

        next_location, next_location_obj = good_locations[random_idx]
        next_location_image_id = next_location_obj.viewpointId
            
        return True, next_location, next_location_image_id
    
    def _generate_next_movement(self, mode, step_i, state, image_id):
        # Extract navigable Locations
        possible_locations = state.navigableLocations[1:]
        
        # Set defaults
        # ------------------------------------------------
        location = 0
        heading_delta = 0
        elevation_delta = 0
        # ------------------------------------------------

        if mode == 'test_elevation':
            if step_i == 0:
                global elevation_test_direction
                elevation_test_direction = self._rnd.choice([-1, 1])
            elevation_delta = elevation_test_direction * math.radians(5.)
            
        elif mode == 'test_heading':
            if step_i == 0:
                global heading_test_direction
                heading_test_direction = self._rnd.choice([-1, 1])
            heading_delta = heading_test_direction * math.radians(5.0)
            
        elif mode == 'full':
            # Check to reset random scanning directions
            if step_i % self.params['n_scan_steps'] == 0:
                self._reset_scan_direction()
            
            # Check for movement
            rand_period = int(self._rnd.normal(loc=self.params['movement_refractory_period'], scale=1.0))
            rand_period = max(0, rand_period)
            if len(possible_locations) > 0 and self.n_since_move > rand_period:
                location_updated, location, image_id = self._sample_new_location(possible_locations, image_id)
                
                if location_updated:
                    # Resets
                    self.n_since_move = 0
                    self._reset_scan_direction()

                    heading_delta = 0 # state.navigableLocations[location].rel_heading
                    elevation_delta = -state.elevation
                    
            if self.n_since_move > self.params['scan_refractory_period']:
                # Scan room
                heading_delta = self._change_heading(heading_delta)
                elevation_delta = self._change_elevation(elevation_delta)
         
        self.n_since_move += 1
        return image_id, location, heading_delta, elevation_delta
    
    def _generate_episode(self, mode, scene_id, image_id, init_heading_DEG, n_sim_steps, verbose=False):
        # Initialize matterport sim
        _ = self.matterport_sim.initialize(scene_id, image_id)
        
        # Init
        # --------------------------------------
        location = 0
        heading_delta = math.radians(init_heading_DEG)
        elevation_delta = 0

        prev_matterport_elevation = 0
        
        self._reset_scan_direction()
        self.n_since_move = 0
        # --------------------------------------
        
        # Run!
        matterport_sensor_data = []
        for step_i in range(n_sim_steps):
            # Stdout
            stdout_str = f"\r[MatterSim] Generating {step_i+1:03d}/{n_sim_steps} "
            stdout_str += f"-- n_since_move: {self.n_since_move:02d} -- heading_delta: {heading_delta}"
            sys.stdout.write(stdout_str)
            sys.stdout.flush()
            
            # Take step
            state, sensor_data = self.matterport_sim.run(step_i, location, image_id, 
                                                         heading_delta, elevation_delta, 
                                                         verbose=verbose)
            
                    
            if self.args.fix_empty_matterport_frames and np.all(sensor_data['RGB']==0):
                self.matterport_sim.close()
                self.matterport_sim.initialize(scene_id, image_id, 
                                               heading=heading_delta, 
                                               elevation=elevation_delta, 
                                               reset_history=False)
                
                state, sensor_data = self.matterport_sim.run(step_i, location, image_id, 
                                                         heading_delta, elevation_delta, 
                                                         verbose=verbose)
                
            matterport_sensor_data.append(sensor_data)
            
            # Limit elevation to matterport sim elevation
            elevation_delta = state.elevation - prev_matterport_elevation
            prev_matterport_elevation = state.elevation
            
            # Store history
            self.matterport_sim.update_history('movement', [image_id, location, heading_delta, elevation_delta])
             
            # Reset
            location = 0
            heading_delta = 0
            elevation_delta = 0
            
            # Generate next movements
            image_id, location, heading_delta, elevation_delta = self._generate_next_movement(mode, step_i, state, image_id)
            
        # Release resources
        self.matterport_sim.close()
        
        return matterport_sensor_data
    
    def setup_scene(self, rnd_seed, max_tries=10):
        succesful_init = False
        attempt_i = 0
        while not succesful_init:
            if attempt_i >= max_tries:
                assert False, f"**FAILURE: Cannot initialize matterport sim after {max_tries} attempts."
            print(f"\nInitialization attempt {attempt_i}")
            succesful_init, vals = self._setup_scene_main(rnd_seed)
            attempt_i += 1
            
        return vals
        
    def _setup_scene_main(self, rnd_seed):
        print("Step 1: Setting up Scene...")
        print("---------------------------")
        succesful_init = True
        
        # Setup scene ids
        scenes_root = os.path.join(self.args.data_root, 'tasks/mp3d/')
        scene_ids = np.sort([os.path.basename(ele) for ele in glob.glob(f'{scenes_root}/*')])

        if self.args.sample_from_viewpoints:
            print("Sampling across viewpoints...")
            scene_to_vp_map = modules.utils.generate_scene_to_viewpoint_map(self.matterport_sim, 
                                                                            scene_ids, 
                                                                            self.args)
            all_viewpoints = [ele2 for ele in list(scene_to_vp_map.values()) for ele2 in ele]

            sampled_idx = rnd_seed.randint(len(all_viewpoints))
            image_id = all_viewpoints[sampled_idx]

            for scene_id, vps in scene_to_vp_map.items():
                if image_id in vps:
                    break

            connectivity = modules.utils.get_connectivity(self.args.data_root, scene_id)
            self.habitat_sim.connectivity = connectivity

        else:
            print("Sampling across scene ids...")
            # Randomly select scene id
            scene_id = rnd_seed.choice(scene_ids)

            # Setup connectivity and assign to HabitatSim
            connectivity = modules.utils.get_connectivity(self.args.data_root, scene_id)
            self.habitat_sim.connectivity = connectivity

            image_id_intersection = self.matterport_sim.find_intersection_image_ids(scene_id, connectivity)
            image_id = rnd_seed.choice(image_id_intersection)

        init_heading_DEG = rnd_seed.randint(360)
        print(f"Scene ID: {scene_id} -- Image ID: {image_id} -- Init Heading: {init_heading_DEG}")
        
        # Attemp initialization of matterport sim
        successful_init = self.matterport_sim.initialize(scene_id, image_id)
        
        state = self.matterport_sim.sim.getState()[0]
        
        if state.location is None:
            successful_init = False
        
        # Release resources
        self.matterport_sim.close()
        
        return successful_init, (scene_id, image_id, init_heading_DEG, connectivity)
    
    def run_sim(self, mode, scene_id, image_id, init_heading_DEG=0, n_sim_steps=100, verbose=False):
        # Generate episode
        print("Generating episode with MatterportSim...")
        matterport_sensor_data = self._generate_episode(mode, scene_id, image_id, init_heading_DEG, n_sim_steps, verbose)
        print("\n")
    
        # Generate habitatsims
        print("\nGenerating corresponding HabitatSim episode...")
        habitat_sensor_data, semantic_data = self.habitat_sim.run(scene_id, self.matterport_sim.history, verbose)
        print("\n")
        
        objects = []
        for obj in semantic_data.objects:
            try:
                region_id = obj.region.id
                region_name = obj.region.category.name()
            except:
                region_id = None
                region_name = None
                
            objects.append({
                "sizes"       : [float(f) for f in obj.aabb.sizes], 
                "center"      : [float(f) for f in obj.aabb.center], 
                "id"          : obj.id, 
                "category"    : obj.category.name(),
                "region_id"   : region_id,
                "region_name" : region_name
            })
        
        return {
            "matterport_sensor" : matterport_sensor_data, 
            "habitat_sensor"    : habitat_sensor_data,
            "semantic"          : objects
        }