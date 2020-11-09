#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys
import cv2
import numpy as np
import utils

# Add parent
sys.path.insert(0, '/root/mount/oc-fewshot/sim/')

import modules.utils
import modules.vis_utils

import config.vis

class EmptyBboxes(Exception):
    pass
    
def process_episode(episode_data, args):
    frames = []
    instance_bank = {}
    
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
    # print('all boxes', all_bboxes)
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
        if np.all(bboxes==0):
            raise EmptyBboxes
        
        # Check for frame limits
        if args.frame_limit and frame_i >= args.frame_limit:
            break
        
        # Instance label for image
        instance_label = f"{category}-{instance_id}"
        
        # stdout
        sys.stdout.write(f'\rProcessing frame {frame_i+1}/{attention_maps.shape[0]}...')
        sys.stdout.flush()

        zoom_frames = []
        for zoom_i, bbox in enumerate(bboxes):
            # Catch empty bbox and continue (result of fixed size arrays in tf.data)
            if np.all(bbox == 0):
                continue
                
            # Check if instance is previously seen
            previously_seen = instance_id in instance_bank

            # Generate attention map
            # ---------------------------------------------------------------------------
            brightness_delta = args.highlight_delta if args.highlight_attended_object else 1.0
            rgb_src_dim = modules.utils.apply_attention(rgb_src, 
                                                        attention_map.astype(np.bool), 
                                                        brightness_delta=brightness_delta,
                                                        use_alpha=args.highlight_use_alpha)
            # ---------------------------------------------------------------------------

            # Pick zoom image to store for instance bank
            if zoom_i == args.log_to_buffer_on_zoom_n:
                _grid_thumbnail_dim = (args.grid_thumbnail_dim, args.grid_thumbnail_dim)
                instance_bank_img = modules.utils.crop_img_from_bbox(rgb_src_dim.copy(), bbox, 
                                                                     resize=True, 
                                                                     resize_dim=_grid_thumbnail_dim)

            # Process left frame
            # ----------------------------------------------------------------------------------------------------------------
            bbox_color = config.vis._SEEN_BBOX_COLOR if previously_seen else config.vis._NEW_BBOX_COLOR
            rgb_main = modules.vis_utils.draw_bbox(rgb_src_dim, 
                                                   bbox=bbox, 
                                                   bbox_thickness=config.vis._BBOX_THICKNESS, 
                                                   color=bbox_color)
            rgb_main_upsample = modules.utils.upsample_frames([rgb_main], 
                                                              dim=(args.resize_width, args.resize_height))[0]
            # ----------------------------------------------------------------------------------------------------------------

            # Create right frame
            grid_bank_img_base = (np.ones_like(rgb_main_upsample)*255).astype(np.uint8)

            # Add baseline text
            num_instances = len(list(instance_bank.keys()))
            num_occurences = 0 if not previously_seen else len(instance_bank[instance_id])
            grid_bank_img_base = utils.add_grid_bank_baseline_text(modules.vis_utils,
                                                                   grid_bank_img_base, 
                                                                   num_instances, 
                                                                   num_occurences, 
                                                                   instance_label, 
                                                                   config.vis._TEXT_X, 
                                                                   config.vis._TEXT_Y, 
                                                                   config.vis._TEXT_Y2, 
                                                                   config.vis._FONT_THICKNESS, 
                                                                   config.vis._FONT_SCALE)

            # Add text (make sure after grid_bank_img_base to avoid incorrect text on right frame!)
            rgb_main_annot = utils.add_primary_baseline_text(modules.vis_utils,
                                                             config.vis,
                                                             rgb_main_upsample, 
                                                             instance_label, 
                                                             previously_seen, 
                                                             config.vis._TEXT_X, 
                                                             config.vis._TEXT_Y, 
                                                             config.vis._FONT_THICKNESS, 
                                                             config.vis._FONT_SCALE)

            # Add previously seen cases (if exist)
            if previously_seen:
                # Only compute on first zoom level
                if zoom_i == 0:
                    # Generate instance row
                    _row_thumbnail_dim = (args.row_thumbnail_dim, args.row_thumbnail_dim)
                    instance_row = utils.generate_instance_row(instance_bank[instance_id], 
                                                               config.vis._ROW_PARAMS, 
                                                               resize_dim=_row_thumbnail_dim)

                    # Generate grid bank
                    grid_bank = utils.generate_grid_bank(instance_bank,
                                                         config.vis, 
                                                         modules.vis_utils,
                                                         instance_id=instance_id, 
                                                         test_grid=args.test_grid)
            else:
                # Generate grid bank
                grid_bank = utils.generate_grid_bank(instance_bank, 
                                                     config.vis,
                                                     modules.vis_utils,
                                                     instance_id=None, 
                                                     test_grid=args.test_grid)
                instance_row = None

            grid_bank_img_base = utils.add_to_base_img(grid_bank_img_base, 
                                                       instance_row, 
                                                       config.vis._ROW_PARAMS)
            grid_bank_img_base = utils.add_to_base_img(grid_bank_img_base, 
                                                       grid_bank, 
                                                       config.vis._GRID_PARAMS)

            # Combine left and right frames
            combined_frame = np.concatenate([rgb_main_annot, grid_bank_img_base], axis=1)

            # Update stack
            zoom_frames.append(combined_frame)

        if args.pause_on_zoom:
            for _ in range(int(args.write_fps*args.zoom_pause_frac)):
                zoom_frames.append(combined_frame)

        if args.zoom_out:
            n_actual_zooms = len(zoom_frames)
            idxs = list(range(n_actual_zooms)) + list(range(n_actual_zooms-1))[::-1]
            zoom_frames = [zoom_frames[idx] for idx in idxs]

        for zoom_frame in zoom_frames:
            frames.append(zoom_frame)

        # Update instance history
        if instance_id not in instance_bank:
            instance_bank[instance_id] = [instance_bank_img]
        else:
            instance_bank[instance_id].append(instance_bank_img)
    
    return frames

def run(dataset_iterator, video_write_dir, args):
    for file_i, episode_data in zip(range(args.n_episodes), dataset_iterator):
        print(f"\nProcessing file {file_i+1}/{args.n_episodes}...")
        
        # Setup video path and check for exist/overwrite
        vid_path = os.path.join(video_write_dir, f'sample_walk_{file_i:03d}.mp4')
        if not args.force and os.path.exists(vid_path):
            print(f"\n{vid_path} exists!")
            continue
        
        # Process frames
        try:
            frames = process_episode(episode_data, args)
        except EmptyBboxes:
            print("Bad episode.")
            continue

        # Write video
        if args.write_video:
            modules.vis_utils.imgs2vid(frames, vid_path, fps=args.write_fps)

def setup_dataset(args):
    # !pip install tensorflow
    # !pip install tensorflow-addons
    
    # Import needed objects
    from fewshot.data.datasets.matterport import MatterportDataset
    from fewshot.data.samplers.minibatch_sampler import MinibatchSampler
    from fewshot.data.preprocessors.normalization_preprocessor import NormalizationPreprocessor
    from fewshot.data.iterators.sim_episode_iterator import SimEpisodeIterator
    
    # Setup sampler, dataset, iterator
    sampler = MinibatchSampler(0)
    dataset = MatterportDataset(args.data_root, args.split_key, args.split_path)
    
    params = {
        "dataset"                : dataset,
        "sampler"                : sampler,
        "batch_size"             : 1, # Do not change
        "nclasses"               : args.n_classes,
        "preprocessor"           : NormalizationPreprocessor(),
        "fix_unknown"            : True,
        "maxlen"                 : 100,
        "semisupervised"         : False,
        "label_ratio"            : 0.1,
        "prefetch"               : True,
        "random_crop"            : True,
        "random_shuffle_objects" : True,
        "seed"                   : 0
    }
    iterator = SimEpisodeIterator(**params)
    
    return iterator

def setup_directories(args):
    video_write_dir = os.path.join(args.output_root, args.output_dir)
    if not os.path.exists(video_write_dir):
        print(f"\nCreating directory @ {video_write_dir}")
        os.mkdir(video_write_dir)
    return video_write_dir

def main(args):
    # Setup directory
    video_write_dir = setup_directories(args)
    
    # Setup dataset iterator
    dataset_iterator = setup_dataset(args)
    
    run(dataset_iterator, video_write_dir, args)
    
def setup_args():
    parser = argparse.ArgumentParser(description='Generate episodes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--data_root', type=str, default='sim/data/fewshot/h5_data')
    parser.add_argument('--split_key', type=str, default='train')
    parser.add_argument('--split_path', type=str, default='fewshot/data/matterport_split')
    parser.add_argument('--output_root', type=str, default='sim/data/fewshot')
    parser.add_argument('--output_dir', type=str, default='sample_walks_with_zoom')
    
    parser.add_argument('--n_classes', type=int, default=30)
    parser.add_argument('--force', type=int, default=0)
    parser.add_argument('--frame_limit', type=int, default=0)
    
    
    parser.add_argument('--write_video', type=int, default=1)
    parser.add_argument('--write_fps', type=float, default=18)
    
    parser.add_argument('--zoom_out', type=int, default=0)
    parser.add_argument('--test_grid', type=int, default=0)
    
    parser.add_argument('--resize_height', type=int, default=850)
    parser.add_argument('--resize_width', type=int, default=1000)
    
    
    parser.add_argument('--grid_thumbnail_dim', type=int, default=150)
    parser.add_argument('--row_thumbnail_dim', type=int, default=100)
    
    parser.add_argument('--highlight_attended_object', type=int, default=1)
    parser.add_argument('--highlight_use_alpha', type=int, default=0)
    parser.add_argument('--highlight_delta', type=float, default=0.7)
    
    parser.add_argument('--pause_on_zoom', type=int, default=1)
    parser.add_argument('--zoom_pause_frac', type=float, default=0.45)
    parser.add_argument('--log_to_buffer_on_zoom_n', type=float, default=3)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = setup_args()
    main(args)
