#!/usr/bin/env python
# coding: utf-8
import os
import sys
import glob
import cv2
import h5py
import argparse
import json
import multiprocessing
import numpy as np
from collections import defaultdict

import modules.utils

import config.zoom_params
import config.sim_params
import config.blacklist

from modules.EpisodeHandler import Episode
from modules.MatterSim import MatterSimAPI
from modules.HabitatSim import HabitatSimAPI
from modules.ZoomGenerator import ZoomGenerator


# Run Functions
def setup_simulators(args, sim_params):
    # Habitat Sim
    habitat_params = {
        "root"          : args.data_root, 
        "width"         : args.width, 
        "height"        : args.height, 
        "hfov"          : sim_params.habitat_sim_hfov_deg, 
        "depth"         : sim_params.depth_on,
        "sensor_height" : 0
    }
    HBT_sim = HabitatSimAPI(habitat_params)
    
    # Matterport Sim
    matterport_params = {
        "width"          : args.width,
        "height"         : args.height,
        "depth"          : sim_params.depth_on,
        "hfov_DEG"       : sim_params.matterport_hfov_deg,
        "annotate_image" : sim_params.annotate_navigable_locs
    }
    MAPT_sim = MatterSimAPI(matterport_params)
    
    # Episode Generation Handler
    episode_handler = Episode(matterport_sim=MAPT_sim, 
                              habitat_sim=HBT_sim, 
                              params=sim_params,
                              rnd=_SIM_RND,
                              args=args)
    
    return HBT_sim, MAPT_sim, episode_handler

def _step_1__setup_scene(HBT_sim, MAPT_sim, args):
    print("Step 1: Setting up Scene...")
    print("---------------------------")
    
    # Setup scene ids
    scenes_root = os.path.join(args.data_root, 'tasks/mp3d/')
    scene_ids = np.sort([os.path.basename(ele) for ele in glob.glob(f'{scenes_root}/*')])
    
    if args.sample_from_viewpoints:
        print("Sampling across viewpoints...")
        scene_to_vp_map = modules.utils.generate_scene_to_viewpoint_map(MAPT_sim, scene_ids, args)
        all_viewpoints = [ele2 for ele in list(scene_to_vp_map.values()) for ele2 in ele]
        
        sampled_idx = _SCENE_RND.randint(len(all_viewpoints))
        image_id = all_viewpoints[sampled_idx]
        
        for scene_id, vps in scene_to_vp_map.items():
            if image_id in vps:
                break
        
        connectivity = modules.utils.get_connectivity(args.data_root, scene_id)
        HBT_sim.connectivity = connectivity
        
    else:
        print("Sampling across scene ids...")
        # Randomly select scene id
        scene_id = _SCENE_RND.choice(scene_ids)

        # Setup connectivity and assign to HabitatSim
        connectivity = modules.utils.get_connectivity(args.data_root, scene_id)
        HBT_sim.connectivity = connectivity

        image_id_intersection = MAPT_sim.find_intersection_image_ids(scene_id, connectivity)
        image_id = _SCENE_RND.choice(image_id_intersection)

    init_heading_DEG = _SCENE_RND.randint(360)
    print(f"Scene ID: {scene_id} -- Image ID: {image_id} -- Init Heading: {init_heading_DEG}")
    
    return scene_id, image_id, init_heading_DEG, connectivity

def _step_2__run_simulation(sim_params, episode_handler, scene_id, image_id, init_heading_DEG):
    print("\nStep 2: Running Simulation...")
    print("-----------------------------")
    print("scene_id, image_id, init_heading_DEG: ", scene_id, image_id, init_heading_DEG)
    sim_data = episode_handler.run_sim(sim_params.mode, scene_id, image_id, 
                                       init_heading_DEG=init_heading_DEG, 
                                       n_sim_steps=args.n_frames_generate,
                                       verbose=args.verbose)
    
    # Update joint histories
    joint_histories = {
        "matterport" : episode_handler.get_history('matter'),
        "habitat"    : episode_handler.get_history('habitat'),
    }
    
    return sim_data, joint_histories
    
def _step_4__filter_empty_frames(all_imgs, all_zoom_info, joint_histories):
    print("\nStep 4: Filtering out empty frames...")
    # Compute valid frame idxs
    valid_frame_idxs = [idx for idx, frame in enumerate(all_zoom_info) if frame != {}]
    print(f"\t{len(valid_frame_idxs)}/{len(all_imgs)} valid frames.")
    
    # Filter images
    all_imgs_filtered = [imgs for idx, imgs in enumerate(all_imgs) if idx in valid_frame_idxs]
    
    # Filter zoom info 
    zoom_info_filtered = [ele for ele in all_zoom_info if ele]
    
    # Filter joint history
    filtered_joint_histories = {}
    for joint_key, joint_vals in joint_histories.items():
        filtered_joint_histories[joint_key] = {}
        for key, frames in joint_vals.items():
            filtered_joint_histories[joint_key][key] = [frames[idx] for idx in valid_frame_idxs]
    print("Complete.")
    return all_imgs_filtered, zoom_info_filtered, filtered_joint_histories

def _step_5__extract_sim_history_data(joint_histories):
    print("\nStep 5: Extracting sim history data...")
    sim_history_info = {
        "history" : {
            "image_id"   : np.array([ele[0].encode("ascii", "ignore") for ele in joint_histories['matterport']['movement']]),
            "location"   : np.array([ele[1] for ele in joint_histories['matterport']['movement']])
        },
        "heading"   : {
            "matterport" : modules.utils.extract_headings(joint_histories['matterport'], accumulate=True),
            "habitat"    : modules.utils.extract_headings(joint_histories['habitat'], accumulate=False)
        },
        "elevation" : {
            "matterport" : modules.utils.extract_elevations(joint_histories['matterport'], accumulate=True),
            "habitat"    : modules.utils.extract_elevations(joint_histories['habitat'], accumulate=False)
        }
    }
    
    return sim_history_info

def _step_6__process_images(all_imgs, args, spec_radius_bounds=(0.95, 1.15)):
    print("\nStep 6: Processing sensor images and align habitat to matterport...")
    
    # Alignment
    processed_imgs = []
    matterport_imgs = []
    habitat_imgs = []
    aligned_semantic_imgs = []
    for i, (mat_im, hab_img, sem_im) in enumerate(all_imgs):
        sys.stdout.write(f'\rProcessing frame {i+1}...')
        sys.stdout.flush()
        warped_sem_im, spec_rad = modules.utils.align_images(mat_im, hab_im, sem_im, spec_radius_bounds)
        
        matterport_imgs.append(mat_im)
        habitat_imgs.append(hab_im)
        aligned_semantic_imgs.append(warped_sem_im)
    
    # Resize images
    matterport_imgs = modules.utils.resize_imgs(matterport_imgs, args, interpolation=cv2.INTER_LINEAR)
    habitat_imgs = modules.utils.resize_imgs(habitat_imgs, args, interpolation=cv2.INTER_LINEAR)
    aligned_semantic_imgs = modules.utils.resize_imgs(aligned_semantic_imgs, args, interpolation=cv2.INTER_NEAREST)
    
    processed_imgs = list(zip(matterport_imgs, habitat_imgs, aligned_semantic_imgs))
    
    return processed_imgs

def _step_6__multiprocess_images(all_imgs, args, spec_radius_bounds=(0.95, 1.15)):
    print("\nStep 6: Processing sensor images and align habitat to matterport...")
    
    Q = multiprocessing.Queue()
    
    def worker(worker_num, idxs, all_imgs):
        print(f"Starting worker {worker_num}")
        return_dict = {}
        return_dict[worker_num] = {
            "matterport_imgs"       : [],
            "habitat_imgs"          : [],
            "aligned_semantic_imgs" : [],
        }
        for i, idx in enumerate(idxs):
            sys.stdout.write(f'\rWorker {worker_num} - Processing frame {i+1}/{len(idxs)}...')
            sys.stdout.flush()
            mat_im = all_imgs[idx][0]
            hab_im = all_imgs[idx][1]
            sem_im = all_imgs[idx][2]
            warped_sem_im, spec_rad = modules.utils.align_images(mat_im, hab_im, sem_im, spec_radius_bounds)
        
            return_dict[worker_num]['matterport_imgs'].append(mat_im)
            return_dict[worker_num]['habitat_imgs'].append(hab_im)
            return_dict[worker_num]['aligned_semantic_imgs'].append(warped_sem_im)
        
        Q.put(return_dict)
        print(f"Worker {worker_num} finished.")
        
    processes = []
    split_idxs = modules.utils.create_split_idxs(len(all_imgs), args.n_processes)
    for i, frame_idxs in enumerate(split_idxs):
        p = multiprocessing.Process(target=worker, args=(i, frame_idxs, all_imgs,))
        processes.append(p)
        p.start()
       
    return_dict = {}
    for p in processes:
        data = Q.get()
        for key, val in data.items():
            return_dict[key] = val
    
    for process in processes:
        process.join()
    
    # Recombine images in correct order
    # ----------------------------------------------------------------------
    print("Combining worker data...")
    matterport_imgs = []
    habitat_imgs = []
    aligned_semantic_imgs = []
    sorted_keys = np.sort(list(return_dict.keys()))
    for key in sorted_keys:
        matterport_imgs += return_dict[key]['matterport_imgs']
        habitat_imgs += return_dict[key]['habitat_imgs']
        aligned_semantic_imgs += return_dict[key]['aligned_semantic_imgs']
    
    # Check resize imgs
    matterport_imgs = modules.utils.resize_imgs(matterport_imgs, args, interpolation=cv2.INTER_LINEAR)
    habitat_imgs = modules.utils.resize_imgs(habitat_imgs, args, interpolation=cv2.INTER_LINEAR)
    aligned_semantic_imgs = modules.utils.resize_imgs(aligned_semantic_imgs, args, interpolation=cv2.INTER_NEAREST)
    
    processed_imgs = list(zip(matterport_imgs, habitat_imgs, aligned_semantic_imgs))
    # ----------------------------------------------------------------------
            
    print("Fin.")
    
    return processed_imgs

def _step_7__split_align_and_log_data(episode_basename, semantic_objs, processed_imgs, 
                                      zoom_info_filtered, sim_history_info, paths, args):
    
    # Compute splits
    n_total_frames = len(processed_imgs)
    n_save_groups = int(np.ceil(n_total_frames / args.n_frames_save))
    all_idxs = list(range(n_total_frames))
    
    print(f"\nStep 6-7: Splitting data into {n_save_groups} groups...")
    
    for group_i in range(n_save_groups):
        print(f"Processing group {group_i+1}/{n_save_groups}...")
        # Get idx slice
        slice_i = slice(group_i*args.n_frames_save, (group_i+1)*args.n_frames_save)

        # Group data into slices
        grouped_imgs = processed_imgs[slice_i]
        
        # Do not save if too few frames
        if len(grouped_imgs) < args.min_valid_frames:
            continue

        # Align images
        # ==============================================================================
        if args.multiproc:
            grouped_imgs = _step_6__multiprocess_images(grouped_imgs, args)
        else:
            grouped_imgs = _step_6__process_images(grouped_imgs, args)
        print("\n")
        # ==============================================================================
        
        # Clean up processed_imgs
        processed_imgs[slice_i] = [None for _ in range(len(processed_imgs[slice_i]))]
        
        # Group zoom info
        grouped_zoom_info = zoom_info_filtered[slice_i]

        # Group sim history
        grouped_sim_history = {}
        for sim_key, sim_vals in sim_history_info.items():
            grouped_sim_history[sim_key] = {}
            for key, val in sim_vals.items():
                grouped_sim_history[sim_key][key] = val[slice_i]

        # Log data
        episode_basename_i = f'{episode_basename}__group_{group_i}'
        modules.utils.log_data(episode_basename_i, semantic_objs, 
                               grouped_imgs, grouped_zoom_info, 
                               grouped_sim_history, args, paths=paths)
        
def main(args, paths):
    # Instantiate simulators
    HBT_sim, MAPT_sim, episode_handler = setup_simulators(args, config.sim_params)
    
    # Instantiate zoom handler
    zoom_generator = ZoomGenerator(config.zoom_params, config.blacklist.categories, _ZOOM_RND)
    
    try:
        start_i = args.start
        for episode_i in range(start_i, start_i+args.n_episodes):
            print(f"Generating episode {episode_i+1}/{args.n_episodes}")
            print("===============================")

            # STEP 1: Scene Setup
            # ==================================================================================
            scene_id, image_id, init_heading_DEG, connectivity = _step_1__setup_scene(HBT_sim, MAPT_sim, args)
            episode_basename = f"episode_{episode_i:06d}__scene_{scene_id}"
            # ==================================================================================

            # STEP 2: Run simulation
            # ==================================================================================
            sim_data, joint_histories = _step_2__run_simulation(config.sim_params, 
                                                                episode_handler, 
                                                                scene_id, 
                                                                image_id, 
                                                                init_heading_DEG)
            # ==================================================================================

            # STEP 3: Generate zooms
            # ==============================================================================
            all_imgs, all_zoom_info = zoom_generator.run(args, sim_data, image_id, 
                                                         region_history=joint_histories['habitat']['regions'])
            # ==============================================================================

            # STEP 4: Filter out frames with no objects
            # ==============================================================================
            images, zoom_annotations, joint_histories = _step_4__filter_empty_frames(all_imgs, 
                                                                                     all_zoom_info, 
                                                                                     joint_histories)
            # ==============================================================================

            if len(images) < args.min_valid_frames:
                print("\n**ALERT: Too few frames after filtering. Skipping to next episode generation.")
                continue

            # Step 5: Extracting data for use in annotations
            # ==============================================================================
            sim_history_info = _step_5__extract_sim_history_data(joint_histories)
            # ==============================================================================

            # ==============================================================================
            # STEP 6,7: Split data into groups, align images, and save data
            # ==============================================================================
            _step_7__split_align_and_log_data(episode_basename, 
                                              sim_data["semantic"], 
                                              images, 
                                              zoom_annotations, 
                                              sim_history_info, 
                                              paths, args)
            # ==============================================================================

            print('-'*40 + '\n')
        print("\nFin.")

    except KeyboardInterrupt:
        print("\nEnding early.")
        
def setup_args():
    parser = argparse.ArgumentParser(description='Generate episodes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--resize_imgs', type=int, default=1)
    parser.add_argument('--resize_width', type=int, default=160)
    parser.add_argument('--resize_height', type=int, default=120)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--n_frames_generate', type=int, default=600)
    parser.add_argument('--n_frames_save', type=int, default=200)
    parser.add_argument('--min_valid_frames', type=int, default=100)
    parser.add_argument('--data_root', type=str, default='data/v1')
    parser.add_argument('--output_root', type=str, default='data/fewshot')
    parser.add_argument('--output_dir', type=str, default='h5_data')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--multiproc', type=int, default=1)
    parser.add_argument('--n_processes', type=int, default=4)
    parser.add_argument('--save_h5', type=int, default=1)
    parser.add_argument('--filter_obj_by_region_id', type=int, default=1)
    parser.add_argument('--save_habitat_imgs', type=int, default=0)
    parser.add_argument('--save_sim_history_json', type=int, default=1)
    parser.add_argument('--sample_from_viewpoints', type=int, default=1)
    parser.add_argument('--add_timestamp_dir', type=int, default=1)
    parser.add_argument('--fix_empty_matterport_frames', type=int, default=1)
    
    args = parser.parse_args()
    return args

def set_random_seed(args):
    global _SCENE_RND, _SIM_RND, _ZOOM_RND
    _SCENE_RND = np.random.RandomState(args.seed)
    _SIM_RND = np.random.RandomState(args.seed)
    _ZOOM_RND = np.random.RandomState(args.seed)

def setup_paths(args):
    timestamp = modules.utils.generate_timestamp() if args.add_timestamp_dir else ''
    paths = {
        "root" : args.output_root,
        "h5"   : os.path.join(args.output_root, args.output_dir, timestamp)
    }
    
    for path in paths.values():
        if not os.path.exists(path):
            print(f"Creating dir: {path}")
            modules.utils.make_dir(path)
    
    return paths

if __name__ == '__main__':
    args = setup_args()
    set_random_seed(args)
    paths = setup_paths(args)
    main(args, paths)
