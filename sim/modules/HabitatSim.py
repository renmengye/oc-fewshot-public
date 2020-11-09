"""HabitatSim API.

Authors: 
 - Mengye Ren (mren@cs.toronto.edu)
 - Michael Iuzzolino (michael.iuzzolino@colorado.edu)
# https://tmp.mosra.cz/habitat-sim/stereo-agent.html

# See this issue on agent position vs. agent_sensor_states position: https://github.com/facebookresearch/habitat-sim/issues/201
# https://github.com/facebookresearch/habitat-sim/issues/101
"""
from __future__ import division, print_function, unicode_literals
import os
import sys
import numpy as np
import habitat_sim

import modules.utils

_DEFAULT_GPU_DEVICE = 0

class HabitatSimAPI(object):
    def __init__(self, params):
        self.params = params
        self._setup_params(params)
    
    def _setup_params(self, params):
        # Setup sim settings
        self.sim_settings = {
            "width"           : int(params['width']),
            "height"          : int(params['height']),
            "default_agent"   : 0,
            "sensor_height"   : float(params['sensor_height']),
            "color_sensor"    : True,
            "semantic_sensor" : True,
            "depth_sensor"    : bool(params['depth']),
            "hfov"            : float(params['hfov']),
            "seed"            : 1
        }
        
        # STDOUT
        print("Parameters")
        print("----------")
        for param_key, param_val in self.sim_settings.items():
            print(f'{param_key:15s} {param_val}')

    def _reset_history(self):
        self.history = {
            "movement" : [],
            "states"   : [],
            "regions"  : []
        }
        
    def _init_sim_enviornment(self, scene_id):
        # Assign scene to sim settings
        self.sim_settings['scene'] = os.path.join(self.params['root'], f"tasks/mp3d/{scene_id}/{scene_id}.glb")  # Scene path
        
        # Make configuration
        self.cfg = self._make_cfg(self.sim_settings)
        
        # Setup sim
        self.sim = habitat_sim.Simulator(self.cfg)
        
        # Initialize agent
        self.agent = self.sim.initialize_agent(self.sim_settings["default_agent"])
        
    def _make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = int(self.params.get('gpu', _DEFAULT_GPU_DEVICE))
        sim_cfg.scene.id = settings["scene"]
        
        # Setup Sensors
        # ------------------------------------------------------------
        # Note: all sensors must have the same resolution
        sensors = {
            "color_sensor"    : {
                "sensor_type"    : habitat_sim.SensorType.COLOR,
                "resolution"     : [settings["height"], settings["width"]],
                "position"       : [0.0, settings["sensor_height"], 0.0],
            },
            "semantic_sensor" : {
                "sensor_type"    : habitat_sim.SensorType.SEMANTIC,
                "resolution"     : [settings["height"], settings["width"]],
                "position"       : [0.0, settings["sensor_height"], 0.0],
            }
        }
        
        if self.sim_settings['depth_sensor']:
            sensors["depth_sensor"] = {
                "sensor_type"    : habitat_sim.SensorType.DEPTH,
                "resolution"     : [settings["height"], settings["width"]],
                "position"       : [0.0, settings["sensor_height"], 0.0],
            }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = habitat_sim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]
                sensor_spec.parameters["hfov"] = str(settings["hfov"])
                sensor_specs.append(sensor_spec)
        # ------------------------------------------------------------
        
        # Setup Agent
        # ------------------------------------------------------------
        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {}
        # ------------------------------------------------------------
        
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def _get_pose(self, image_id):
        # Check for pose
        # ---------------------------------
        pose = None
        for node in self.connectivity:
            if node['image_id'] == image_id:
                pose = node['pose']
                pose = np.array(pose).reshape([4, 4])
                break
        # ---------------------------------
        
        # Check case where pose not found
        # ------------------------------------------------------
        if pose is None:
            assert False, f"Position ID {image_id} not found."
        # ------------------------------------------------------
        
        return pose

    def _extract_translation(self, pose):
        translation = pose[:3, 3]
        
        # Flip y and z, then invert z
        x, y, z = translation[0], translation[2], -translation[1]
        translation = np.array([x, y, z])
        
        return translation
        
    def teleport(self, agent, translation, rotation):
        """Teleport to a predefined location."""
        # Set Agent State
        # --------------------------------------------------------
        agent_state = habitat_sim.AgentState()
        agent_state.position = translation
        agent_state.rotation = rotation
        # See suggestion here: https://github.com/facebookresearch/habitat-sim/issues/201
        agent.state.sensor_states = {}
        agent.set_state(agent_state)
        # --------------------------------------------------------
        
    def take_snapshot(self, pose, heading=0.0, elevation=0.0, verbose=False):
        # Get transforms
        # --------------------------------------------------------
        cam_translation = self._extract_translation(pose)
        cam_rotation_quaternion = modules.utils.euler_to_quat(heading, elevation)
        # --------------------------------------------------------
        
        # Teleport agent
        self.teleport(self.agent, translation=cam_translation, rotation=cam_rotation_quaternion)
        
        # Get observations
        # --------------------------------------------
        expected_x, expected_y, expected_z = cam_translation
        pre_x, pre_y, pre_z = self.agent.state.position
        if verbose:
            stdout_str = f"\n\t[HabitatSim] EXPECTED Pose (x,y,z):   "
            stdout_str += f"({expected_x:0.6f}, "
            stdout_str += f"{expected_y:0.6f}, "
            stdout_str += f"{expected_z:0.6f})"
            print(stdout_str)
            
            stdout_str = f"\tBEFORE [HabitatSim] Location (x,y,z): "
            stdout_str += f"({pre_x:0.6f}, "
            stdout_str += f"{pre_y:0.6f}, "
            stdout_str += f"{pre_z:0.6f})"
            print(stdout_str)
        
        # Get observations
        # -------------------------------------------------------
        observations = self.sim.get_sensor_observations()
        
        # Extract data from observations
        data = { modules.utils.color2rgb_check(key.split("_")[0]) : val for key, val in observations.items() }
        # -------------------------------------------------------
        
        post_x, post_y, post_z = self.agent.state.position
        if verbose:
            stdout_str = f"\tAFTER [HabitatSim] Location (x,y,z):  "
            stdout_str += f"({post_x:0.6f}, "
            stdout_str += f"{post_y:0.6f}, "
            stdout_str += f"{post_z:0.6f})"
            print(stdout_str)
        
        # Check that we're in the expected location!
        assert np.abs(post_x - expected_x) < 1e-4, f"Unexpected X location! Expected: {expected_x} - Agent: {post_x}"
        assert np.abs(post_y - expected_y) < 1e-4, f"Unexpected Y location! Expected: {expected_y} - Agent: {post_y}"
        assert np.abs(post_z - expected_z) < 1e-4, f"Unexpected Z location! Expected: {expected_z} - Agent: {post_z}"
        
        return data
    
    def _print_semantic_scene(self, scene):
        print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
        for level_i, level in enumerate(scene.levels):
            print(f"Level {level_i}")
            print(
                f"Level id:{level.id}, center:{level.aabb.center},"
                f" dims:{level.aabb.sizes}"
            )
            for region in level.regions:
                print(
                    f"\tRegion id:{region.id}, category:{region.category.name()},"
                    f"\t center:{region.aabb.center}, dims:{region.aabb.sizes}"
                )
                for obj in region.objects:
                    print(
                        f"\t\tObject id:{obj.id}, category:{obj.category.name()},"
                        f"\t\t center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                    )
    
    def _compute_scene_object_positions(self, scene):
        obj_info = []
        for level_i, level in enumerate(scene.levels):
            for region in level.regions:
                for obj in region.objects:
                    obj_info.append([obj.id, obj.category.name(), 
                                     obj.region.id, obj.region.category.name(), 
                                     np.array(obj.aabb.center)])
        return obj_info
    
    def _find_closest_object(self, agent_pos, scene_obj_info):
        agent_pos = np.array(agent_pos)
        distances = [np.linalg.norm(agent_pos-obj[-1]) for obj in scene_obj_info]
        min_idx = np.argmin(distances)
        closest_obj = scene_obj_info[min_idx]
        return closest_obj
    
    def _find_agent_region(self, agent_pos, scene_obj_info):
        closest_object = self._find_closest_object(self.agent.state.position, scene_obj_info)
        agent_region_info = [closest_object[2], closest_object[3]] 
        return agent_region_info
    
    def run(self, scene_id, matterport_history, verbose=False):
        # Reset history
        self._reset_history()
        
        # Init
        heading = 0
        elevation = 0
            
        # Init sim environment
        self._init_sim_enviornment(scene_id)
        
        num_steps = len(matterport_history['movement'])
        sensor_data = []
        semantic_data = self.sim.semantic_scene
        
        scene_obj_info = self._compute_scene_object_positions(semantic_data)
        
        for i, (image_id, loc_i, heading_i, elevation_i) in enumerate(matterport_history['movement']):
            # Stdout
            sys.stdout.write(f'\r[HabitatSim] -- Generating {i+1}/{num_steps}...')
            sys.stdout.flush()
            
            # Get pose
            pose = self._get_pose(image_id)
            
            # Update heading and elevation
            # ----------------------------
            heading += heading_i
            heading %= 2*np.pi
            elevation += elevation_i
            # ----------------------------
            
            # Take Snapshot
            data_i = self.take_snapshot(pose, heading, elevation, verbose=verbose)
            sensor_data.append(data_i)
        
            # Determine agent's region_id and region name
            agent_region_info = self._find_agent_region(self.agent.state.position, scene_obj_info)
            
            # Update history
            # ---------------------------------------------------------------------
            self.history['movement'].append((image_id, loc_i, heading, elevation))
            self.history['states'].append(self.agent.state)
            self.history['regions'].append(agent_region_info)
            # ---------------------------------------------------------------------
            
        # Release resources
        self.close()
    
        return sensor_data, semantic_data
    
    def close(self, verbose=True):
        if verbose:
            print("\nClosing HabitatSim.")
        try:
            self.sim.close()
        except Exception as e:
            print(f"Error closing sim. \n{e}")