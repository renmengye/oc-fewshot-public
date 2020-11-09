"""Zoom Generator API.

Authors: 
 - Michael Iuzzolino (michael.iuzzolino@colorado.edu)
 - Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import division, print_function, unicode_literals
import os
import sys
import numpy as np
import modules.utils

class ZoomGenerator(object):
    def __init__(self, params, blacklist_categories, rnd):
        self._rnd = rnd
        self.params = params
        self.blacklist_categories = blacklist_categories
        
        # Helper Lambda Functions
        # =======================================================
        self.get_min_area_threshold = lambda img, r=0.004: int(np.prod(img.shape[:2])*r)
        self.get_max_area_threshold = lambda img, r=0.75: int(np.prod(img.shape[:2])*r)
        # =======================================================
        
    def _generate_zoom_bboxes(self, img_dim, n_zooms, centroid, original_bbox):
        # Extract h,w from image
        img_h, img_w = img_dim
        
        # Extract x,y from bbox
        y1, y2, x1, x2 = original_bbox
        
        # Extract cent x, y
        cent_x, cent_y = centroid
        
        # Init bbox h, w
        bbox_h = y2-y1
        bbox_w = x2-x1

        # Compute stepsize
        h_diff = img_h - bbox_h
        w_diff = img_w - bbox_h
        h_stepsize = h_diff // n_zooms
        w_stepsize = w_diff // n_zooms
    
        # Generate bboxes
        bboxes = []
        for zoom_i in range(n_zooms-1):
            new_bbox_h = bbox_h + h_stepsize
            new_bbox_w = bbox_w + w_stepsize

            new_x1 = int(max(0, cent_x - new_bbox_w//2))
            new_x2 = int(min(img_w, cent_x + new_bbox_w//2))
            new_y1 = int(max(0, cent_y - new_bbox_h//2))
            new_y2 = int(min(img_h, cent_y + new_bbox_h//2))
            new_bbox = [new_y1, new_y2, new_x1, new_x2]
            bboxes.append(new_bbox)

            bbox_h = new_y2 - new_y1
            bbox_w = new_x2 - new_x1
        
        # Prepend with original bbox
        bboxes = [original_bbox] + bboxes
        
        # Invert for far-to-close effect
        bboxes = bboxes[::-1]

        return bboxes
    
    def run(self, args, sim_data, image_id, region_history):
        print("\nStep 5: Processing Zooms...")
        n_frames = len(sim_data['matterport_sensor'])
        assert n_frames == len(region_history), 'Region history and Images do not align in timesteps!'
        
        # Generate object dictionary
        objects_dict = modules.utils.compute_objects_dict(sim_data)
        
        # Compute valid instance ids in episode conditioned on blacklist categories
        valid_ids_in_episode = modules.utils.get_valid_ids(objects_dict, self.blacklist_categories)
        
        all_imgs = []
        all_zoom_info = []
        for frame_i in range(n_frames):
            # Stdout
            sys.stdout.write(f"\rZooms on Frame {frame_i+1}/{n_frames}")
            sys.stdout.flush()
            
            # Get agent's region info at frame_i
            agent_region_id = region_history[frame_i][0]

            # Load frame data
            matterport_img = sim_data['matterport_sensor'][frame_i]['RGB']
            habitat_img = sim_data['habitat_sensor'][frame_i]['RGB']
            semseg_map = sim_data['habitat_sensor'][frame_i]['semantic']

            # Add baseline images for frame_i
            # [full_size RGB, zoomed object img, attention, zoomed attention, semseg, zoomed semseg]
            all_imgs += [[matterport_img, habitat_img, semseg_map]]

            # Get intersection between object ids in frame and valid ids in episode
            object_ids_in_frame = np.unique(semseg_map)
            valid_ids_in_frame = list(set(valid_ids_in_episode).intersection(set(object_ids_in_frame)))

            # Randomly sample number of objects to zoom within current frame
            # Cannot just sample valid ids a priori as some may be invalid (e.g., due to being too small of a fragment)
            num_objects_to_zoom_in_frame = self._rnd.randint(self.params.min_objects, self.params.max_objects)

            frame_zoom_info = {}
            num_objects_zoomed = 0
            for i, valid_idx in enumerate(valid_ids_in_frame):
                # Check that zooming is finished
                if not self.params.limit_objects_per_frame and (num_objects_zoomed >= num_objects_to_zoom_in_frame):
                    break
                
                # Extract region id
                object_region_id = objects_dict[valid_idx]['region_id']
                
                if args.filter_obj_by_region_id:
                    if object_region_id != agent_region_id:
                        continue
                
                # Build mask for target object
                obj_i_mask = semseg_map==valid_idx
                
                # Check area of object -- if too small, discard
                object_area = np.sum(obj_i_mask)
                if object_area < self.get_min_area_threshold(obj_i_mask):
                    continue

                # Create binary map from semantic segmentation for connected components
                binary_map = obj_i_mask.astype(np.uint8)
                
                # Compute primary bbox and bbox centroid
                primary_bbox = modules.utils.attention2bbox(binary_map)
                centroid = modules.utils.bbox2centroid(primary_bbox)
                
                # Limit centroids to objects within x% from center
                valid_centroid = modules.utils.limit_centroid_location(binary_map, centroid, limit=self.params.periphery_limit)
                if not valid_centroid:
                    continue
                
                # Randomly sample number of zoom steps
                n_zooms = self._rnd.randint(self.params.min_zooms, self.params.max_zooms)
                
                # Perform object zooming
                try:
                    zoom_bboxes = self._generate_zoom_bboxes(matterport_img.shape[:2], n_zooms, centroid, primary_bbox)
                except Exception as e:
                    print("\nE: ", e)
                    continue

                # Create object key
                key = f'object_{num_objects_zoomed}'
                
                # DEBUG: catch empty category; look into why this happens later
                category = objects_dict[valid_idx]['category']
                if category == '':
                    continue
                
                frame_zoom_info[key] = {
                    "instance_id" : int(valid_idx),
                    "category"    : objects_dict[valid_idx]['category'],
                    "region_id"   : object_region_id,
                    "region_name" : objects_dict[valid_idx]['region_name'],
                    "viewpoint"   : image_id,
                    "centroid"    : list(centroid), # Redundant - can retrieve from bbox
                    "zoom_bboxes" : [list(ele) for ele in zoom_bboxes]
                }

                # Increment counter
                num_objects_zoomed += 1

            all_zoom_info.append(frame_zoom_info)

        return all_imgs, all_zoom_info