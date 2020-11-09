#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys
import cv2
import numpy as np

parent_dir = os.getcwd()

# Add parent
vis_path = '/root/mount/oc-fewshot/sim/visualization'
os.chdir(vis_path)
import utils
os.chdir(parent_dir)

# Add parent
modules_path = '/root/mount/oc-fewshot/sim/'
os.chdir(modules_path)
import modules.utils
import modules.vis_utils
import config.vis_model as config_vis
import config.color_palette
os.chdir(parent_dir)

_NEW_BBOX_COLOR = utils.hex2rgb(config.color_palette.colors['gray'], reverse=True)
_SEEN_BBOX_COLOR = utils.hex2rgb(config.color_palette.colors['red'], reverse=True)
_UNLABELED_IDX = 40
_UNLABELED_KEY = 'Unknown'
# _ZOOM_ON = True
_SLOW_FACTOR = 4 # 4
_SLOW_FRAC = 1 / 4

def build_instance_lookup(y_full, batch, valid_idxs):
    y_full = y_full[valid_idxs]
    instance_ids = batch['instance_id'][0].numpy()[valid_idxs]
    lookup = {int(key) : int(val)  for key, val in zip(y_full, instance_ids)}
    return lookup

def build_class_lookup(batch, valid_idxs):
    instance_ids = batch['instance_id'][0].numpy()[valid_idxs]
    categories = batch['category'][0].numpy()[valid_idxs]
    lookup = {int(key) : val.decode()  for key, val in zip(instance_ids, categories)}
    return lookup
    
class EmptyBboxes(Exception):
    pass

class VideoGenerator:
    def __init__(self, args, zoom_on):
        self.args = args
        self.zoom_on = zoom_on
    
    def setup_directories(self):
        video_write_dir = os.path.join(self.args.output_root, self.args.output_dir)
        
        if not os.path.exists(video_write_dir):
            print(f"\nCreating directory @ {video_write_dir}")
            os.makedirs(video_write_dir)
        return video_write_dir
    
    def _slow_frames(self, frames, start_i, end_i, slow_factor=10):
        new_frames = []
        slow_frames = []
        for slow_frame in frames[start_i:end_i]:
            for _ in range(slow_factor):
                slow_frames.append(slow_frame)
        new_frames = slow_frames + frames[end_i:]
        return new_frames
    
    def process_episode(self, episode_data, flags, y_preds, y_gts, y_ss, y_full, vid_path):
        frames = []
        instance_bank = {}
        
        # Compute valid idxs
        valid_idxs = np.where(flags)

        # Generate lookup 
        instance_lookup = build_instance_lookup(y_full, episode_data, valid_idxs)
        class_lookup = build_class_lookup(episode_data, valid_idxs)
        class_lookup[_UNLABELED_IDX] = _UNLABELED_KEY

        # Read RGBA (RGB + attention)
        # --------------------------------------------------------
        rgba_imgs = episode_data['x_s'].numpy()[0]
        matterport_RGBs = rgba_imgs[:,:,:,:3]
        matterport_RGBs = np.array([utils.renorm_img(im) for im in matterport_RGBs])
        attention_maps = rgba_imgs[:,:,:,3]
        # --------------------------------------------------------

        # Parse categories, instance ids, and bboxs
        # --------------------------------------------------------
        categories = [ele.decode() for ele in episode_data['category'][0].numpy()]
        instance_ids = [ele.decode() for ele in episode_data['instance_id'][0].numpy()]
        all_bboxes = episode_data['bbox'][0].numpy()
        # --------------------------------------------------------

        # Process frames
        stack = [
            matterport_RGBs, 
            attention_maps,
            all_bboxes,
            categories,
            instance_ids
        ]
        for frame_i, (rgb_src, attention_map, bboxes, category, instance_id) in enumerate(zip(*stack)):
            if frame_i not in valid_idxs[0]:
                continue
                
            if np.all(bboxes==0):
                raise EmptyBboxes

            # Check for frame limits
            if self.args.frame_limit and frame_i >= self.args.frame_limit:
                break
            
            # Grab model prediction at frame_i
            y_gt = y_gts[frame_i]
            y_pred = y_preds[frame_i]
            y_s = y_ss[frame_i]

            # Check if prediction is correct
            correct_prediction = int(y_pred)==int(y_gt)
            
            # Actual label
            # ----------------------------
            y_s_instance_id = instance_lookup.get(y_s, _UNLABELED_IDX) 
            y_s_class = class_lookup[y_s_instance_id]
            y_s_label = f"{y_s_class.replace('_', ' ').capitalize()} {y_s_instance_id}"
            
            # Check if previously seen
            previously_seen = y_s_label in instance_bank
            
            # CHeck if label provided
            label_provided = y_s != _UNLABELED_IDX
            
            if not label_provided:
                actual_label = 'Label: None'
                bbox_color = _NEW_BBOX_COLOR
                linestyle = 'solid' #'dotted'
            else:
                actual_label = f"Label: {y_s_label.capitalize()}"
                bbox_color = _SEEN_BBOX_COLOR
                linestyle = 'solid'
            # ----------------------------
            
            # Predicted label
            # ----------------------------
            y_pred_instance_id = instance_lookup.get(y_pred, _UNLABELED_IDX) 
            y_pred_class = class_lookup[y_pred_instance_id]
            
            if y_pred_class == _UNLABELED_KEY:
                predicted_label = f'Prediction: {_UNLABELED_KEY}'
            else:
                predicted_label = f"Prediction: {y_pred_class.replace('_', ' ').capitalize()} {y_pred_instance_id}"
            # ----------------------------
      
            # stdout
            sys.stdout.write(f'\rProcessing frame {frame_i+1}/{attention_maps.shape[0]}...')
            sys.stdout.flush()

            zoom_frames = []
            if not self.zoom_on:
                for bbox in bboxes[::-1]:
                    if not np.all(bbox == 0):
                        break
                bboxes = [bbox]
                
            for zoom_i, bbox in enumerate(bboxes):
                # Catch empty bbox and continue (result of fixed size arrays in tf.data)
                if np.all(bbox == 0):
                    continue

                # Generate attention map
                # ---------------------------------------------------------------------------
                brightness_delta = self.args.highlight_delta if self.args.highlight_attended_object else 1.0
                rgb_src_dim = modules.utils.apply_attention(rgb_src, 
                                                            attention_map.astype(np.bool), 
                                                            brightness_delta=brightness_delta,
                                                            use_alpha=self.args.highlight_use_alpha)
                # ---------------------------------------------------------------------------

                # Transform bbox to correct coords
                if self.args.transform_bbox:
                    original_dim = (self.args.original_img_h, self.args.original_img_w)
                    bbox = modules.utils.transform_bbox(bbox, 
                                                        rgb_src_dim.shape[:2], 
                                                        original_dim=original_dim)

                # Pick zoom image to store for instance bank
                if not self.zoom_on or zoom_i == self.args.log_to_buffer_on_zoom_n:
                    _grid_thumbnail_dim = (self.args.grid_thumbnail_dim, self.args.grid_thumbnail_dim)
                    instance_bank_img = modules.utils.crop_img_from_bbox(rgb_src_dim.copy(), 
                                                                         bbox, 
                                                                         resize=True, 
                                                                         resize_dim=_grid_thumbnail_dim)

                # Process left frame
                # ------------------------------------------------------------------------------------------
                rgb_main = modules.vis_utils.draw_bbox(rgb_src_dim, 
                                                       bbox=bbox, 
                                                       bbox_thickness=config_vis._BBOX_THICKNESS, 
                                                       color=bbox_color,
                                                       style=linestyle,
                                                       gap=config_vis._LINE_GAP)
                rgb_main_upsample = modules.utils.upsample_frames([rgb_main], 
                                                                  dim=(self.args.resize_width,
                                                                       self.args.resize_height))[0]
                # ------------------------------------------------------------------------------------------

                # Create right frame
                grid_bank_img_base = (np.ones_like(rgb_main_upsample)*255).astype(np.uint8)

                # Add baseline text
                num_instances = len(list(instance_bank.keys()))
                if y_s_label in instance_bank:
                    num_occurences = len(instance_bank[y_s_label])
                else:
                    num_occurences = 0
                grid_bank_img_base = utils.add_grid_bank_baseline_text(modules.vis_utils,
                                                                       grid_bank_img_base, 
                                                                       num_instances, 
                                                                       num_occurences, 
                                                                       actual_label, 
                                                                       config_vis._TEXT_X, 
                                                                       config_vis._TEXT_Y, 
                                                                       config_vis._TEXT_Y2, 
                                                                       config_vis._FONT_THICKNESS, 
                                                                       config_vis._FONT_SCALE)

                # Add text (make sure after grid_bank_img_base to avoid incorrect text on right frame!)
                rgb_main_annot = utils.add_primary_baseline_text(modules.vis_utils,
                                                                 config_vis,
                                                                 rgb_main_upsample, 
                                                                 actual_label,
                                                                 predicted_label, 
                                                                 correct_prediction,
                                                                 config_vis._TEXT_X, 
                                                                 config_vis._TEXT_Y, 
                                                                 config_vis._LARGE_FONT_THICKNESS, 
                                                                 config_vis._LARGE_FONT_SCALE,
                                                                 config_vis._BLACKOUT_W,
                                                                 config_vis._BLACKOUT_H,
                                                                 _UNLABELED_KEY,
                                                                (_NEW_BBOX_COLOR, _SEEN_BBOX_COLOR))

                # Add previously seen cases (if exist)
                if previously_seen:
                    # Only compute on first zoom level
                    if zoom_i == 0:
                        # Generate instance row
                        _row_thumbnail_dim = (self.args.row_thumbnail_dim, self.args.row_thumbnail_dim)
                        instance_row = utils.generate_instance_row(instance_bank[y_s_label], 
                                                                   config_vis._ROW_PARAMS, 
                                                                   resize_dim=_row_thumbnail_dim)

                        # Generate grid bank
                        grid_bank = utils.generate_grid_bank(instance_bank,
                                                             config_vis, 
                                                             modules.vis_utils,
                                                             _SEEN_BBOX_COLOR,
                                                             instance_id=y_s_label, 
                                                             test_grid=self.args.test_grid)
                else:
                    # Generate grid bank
                    grid_bank = utils.generate_grid_bank(instance_bank, 
                                                         config_vis,
                                                         modules.vis_utils,
                                                         _SEEN_BBOX_COLOR,
                                                         instance_id=None, 
                                                         test_grid=self.args.test_grid)
                    instance_row = None

                grid_bank_img_base = utils.add_to_base_img(grid_bank_img_base, 
                                                           instance_row, 
                                                           config_vis._ROW_PARAMS)
                grid_bank_img_base = utils.add_to_base_img(grid_bank_img_base, 
                                                           grid_bank, 
                                                           config_vis._GRID_PARAMS)

                # Combine left and right frames
                combined_frame = np.concatenate([rgb_main_annot, grid_bank_img_base], axis=1)

                # Update stack
                zoom_frames.append(combined_frame)

            if self.args.pause_on_zoom:
                for _ in range(int(self.args.write_fps*self.args.zoom_pause_frac)):
                    zoom_frames.append(combined_frame)

            if self.args.zoom_out:
                n_actual_zooms = len(zoom_frames)
                idxs = list(range(n_actual_zooms)) + list(range(n_actual_zooms-1))[::-1]
                zoom_frames = [zoom_frames[idx] for idx in idxs]

            for zoom_frame in zoom_frames:
                frames.append(zoom_frame)

            # Update instance history
            if label_provided:
                if y_s_label not in instance_bank:
                    instance_bank[y_s_label] = [instance_bank_img]
                else:
                    instance_bank[y_s_label].append(instance_bank_img)
        
        # Slow frames
        # --------------------------------------------------
        end_i = int(len(frames) * _SLOW_FRAC)
        frames = self._slow_frames(frames, 
                                   start_i=0, end_i=end_i, 
                                   slow_factor=_SLOW_FACTOR
                                  )
        # --------------------------------------------------

        modules.vis_utils.imgs2vid(frames, vid_path, fps=self.args.write_fps)
        
class GeneratorArgs:
    def __init__(self, output_dir='sample_walks_with_zoom'):
        self.seed = 42
        self.output_root = '/root/mount/oc-fewshot/data/final_videos'
        self.output_dir = output_dir

        self.frame_limit = 0

        self.write_video = 1
        self.write_fps = 25 # 18

        self.zoom_out = 0
        self.test_grid = 0

        self.resize_height = 850
        self.resize_width = 1000

        self.grid_thumbnail_dim = 150
        self.row_thumbnail_dim = 100

        self.highlight_attended_object = 1
        self.highlight_use_alpha = 0
        self.highlight_delta = 0.7

        self.pause_on_zoom = 1
        self.zoom_pause_frac = 0.45
        self.log_to_buffer_on_zoom_n = 3
        
        self.transform_bbox = 0
        self.original_img_h = 600
        self.original_img_w = 800
