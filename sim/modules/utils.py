"""Basic utilities.

Authors: 
 - Mengye Ren (mren@cs.toronto.edu)
 - Michael Iuzzolino (michael.iuzzolino@colorado.edu)
"""
from __future__ import division, print_function, unicode_literals

import os
import sys
import glob
import cv2
import h5py
import json
import numpy as np
from collections import defaultdict
import torch
from torchvision.utils import make_grid
try:
    import habitat_sim.utils.common
except:
    print("Could not import habitat_sim.utils.common")
    
from datetime import datetime

color2rgb_check = lambda x : 'RGB' if x == 'color' else x

def transform_bbox(bbox, new_dim, original_dim=(600,800)):
    # Helpers
    get_new_dim = lambda x, axis : int(x / original_dim[axis] * new_dim[axis])
    get_w_ratio = lambda x : x / original_dim[1]
    
    # Unpack
    y1, y2, x1, x2 = bbox
    
    # Transform
    new_bbox = [
        int(get_new_dim(y1, axis=0)), 
        int(get_new_dim(y2, axis=0)), 
        int(get_new_dim(x1, axis=1)), 
        int(get_new_dim(x2, axis=1))
    ]
    
    return new_bbox

def generate_timestamp():
    obj = datetime.now()
    timestamp = f'run_{obj.year}Y_{obj.month}M_{obj.day}D_{obj.hour}h_{obj.minute}m_{obj.second}s'
    return timestamp

def generate_scene_to_viewpoint_map(MAPT_sim, scene_ids, args):
    scene_to_vp_map = defaultdict(list)
    
    connectivity_paths = glob.glob(f'connectivity/*.json')
    for map_i, connectivity_map_path in enumerate(connectivity_paths):
        scene_id = os.path.basename(connectivity_map_path).split("_connectivity")[0]
        sys.stdout.write(f'\rProcessing {scene_id}: {map_i+1}/{len(connectivity_paths)}')
        sys.stdout.flush()
        
        with open(connectivity_map_path, 'r') as infile:
            connectivity_map = json.load(infile)
        
        for node in connectivity_map:
            image_id = node['image_id']
            scene_to_vp_map[scene_id].append(image_id)
    print("\n")
            
    return scene_to_vp_map

def upsample_frames(imgs, dim=(800,600), interpolation=cv2.INTER_NEAREST):
    upsampled_imgs = [cv2.resize(im, dim, interpolation=interpolation) for im in imgs]
    return upsampled_imgs

def crop_img_from_bbox(img, bbox, resize=False, resize_dim=(50,50)):
    y1, y2, x1, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    print(y1, y2, x1, x2, img.shape)
    cropped_img = cv2.resize(cropped_img, resize_dim, interpolation=cv2.INTER_NEAREST)
    return cropped_img

def pad_image(img, padding, pad_val=255, direction='R'):
    h, w = img.shape[:2]
    pad_w = w + padding
    padded_img = np.ones((h, pad_w, 3), dtype=np.uint8)*pad_val
    if direction == 'R':
        padded_img[0:h,0:w] = img
    elif direction == 'L':
        padded_img[0:h,padding:w+padding] = img
    
    return padded_img

def attention2bbox(attention_map):
    lookup = np.where(attention_map)
    x1, x2 = int(np.min(lookup[1])), int(np.max(lookup[1]))
    y1, y2 = int(np.min(lookup[0])), int(np.max(lookup[0]))
    return [y1, y2, x1, x2]
    
def decode_png_to_np(png_data, png_data_len):
    pngs = np.array(png_data)
    idxs = [0] + list(np.cumsum(np.array(png_data_len)))
    decoded_np_imgs = np.array([cv2.imdecode(pngs[idxs[i]:idxs[i+1]], cv2.IMREAD_UNCHANGED) for i in range(len(idxs)-1)])
    return decoded_np_imgs

def zoom_in_on_img(bbox, matterport_img, binary_map):
    h, w = matterport_img.shape[:2]
    y1, y2, x1, x2 = bbox
    
    return_data = []
    
    # Matterport zooming
    # --------------------------------------------------
    zoomed_matterport_img = matterport_img[y1:y2, x1:x2]
    zoomed_matterport_img = cv2.resize(zoomed_matterport_img, (w, h))
    return_data += [zoomed_matterport_img]
    # --------------------------------------------------
    
    # Binary map zooming
    # --------------------------------------------------
    if binary_map is not None:
        zoomed_binary_map = binary_map[y1:y2, x1:x2]
        zoomed_binary_map = cv2.resize(zoomed_binary_map, (w, h), interpolation=cv2.INTER_NEAREST)
        return_data += [zoomed_binary_map]
    # --------------------------------------------------
        
    return return_data

def bbox2centroid(bbox, cast_int=True):
    # Helper
    cast2int = lambda x : int(x) if cast_int else x
    
    # Unpack
    y1, y2, x1, x2 = bbox
    
    # Retrieve centroid
    cent_x = cast2int(x1 + (x2-x1)//2)
    cent_y = cast2int(y1 + (y2-y1)//2)
    return cent_x, cent_y

def compute_objects_dict(sim_data):
    objects = sim_data["semantic"]
    obj_ids = [int(ele['id'].split("_")[-1]) for ele in objects]
    objects_dict = dict(zip(obj_ids, objects))
    return objects_dict

def semantic_seg_to_binary_map(semantic_seg_img, object_mask):
    filtered_img = np.copy(semantic_seg_img)
    filtered_img[np.where(~object_mask)] = 0
    binary_map = (filtered_img>0).astype(np.uint8)
    return binary_map

def get_valid_ids(objects_dict, blacklist_categories):
    valid_ids = []
    for key, val in objects_dict.items():
        if val['category'].lower() not in blacklist_categories:
            valid_ids.append(key)
    valid_ids = np.unique(valid_ids)
    return valid_ids

def change_brightness(img, value=0.5):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_img[...,2] = hsv_img[...,2] * value
    final_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return final_img

def apply_attention(img, target_obj_mask, brightness_delta, use_alpha=False):
    if use_alpha:
        attention_img = img.copy()
        if img.shape[-1] == 3:
            attention_img = (np.ones((*img.shape[:-1], 4))*255).astype(np.uint8)
            attention_img[...,:3] = img 
        attention_img[~target_obj_mask] = (attention_img*brightness_delta)[~target_obj_mask]
    else:
        attention_img = np.copy(img[:,:,:3])
        attention_img[~target_obj_mask] = change_brightness(np.copy(attention_img), value=brightness_delta)[~target_obj_mask]
    return attention_img

def get_centroid(binary_map, get_min_area_threshold, get_max_area_threshold):
    n_labels, labels, stats, centroids = run_connected_components(binary_map)
    areas = stats[:,-1]
    
    # Filter out centroid for region
    areas = areas[1:]
    centroids = centroids[1:]
    
    good_idxs = np.where((areas>get_min_area_threshold(binary_map)) & (areas<get_max_area_threshold(binary_map)))
    centroids = centroids[good_idxs]
    areas = areas[good_idxs]
    
    if len(areas) == 0:
        centroid = None
    else:
        max_area_idx = np.argmax(areas)
        centroid = centroids[max_area_idx]
    return centroid

def run_connected_components(binary_map, connectivity=4):
    # Perform the operation
    # Output: num_labels, label matrix, stat matrix, centroid matrix
    output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    return output

def create_split_idxs(n_frames, n_processes):
    n_per_process = n_frames // n_processes
    frame_idxs = list(range(n_frames))
    idxs = [i*n_per_process for i in range(n_processes)]
    idxs.append(None)
    split_idxs = [frame_idxs[idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
    return split_idxs

def extract_headings(history, accumulate):
    headings = np.array([ele[2] for ele in history['movement']])
    if accumulate:
        headings = np.cumsum(headings)
    headings %= (2*np.pi)
    headings = np.rad2deg(headings)
    return headings

def extract_elevations(history, accumulate):
    elevations = np.array([ele[3] for ele in history['movement']])
    if accumulate:
        elevations = np.cumsum(elevations)
    elevations = np.rad2deg(elevations)
    return elevations

def get_valid_centroid_limits(img, dim, limit):
    valid_center = img.shape[dim]*(1-limit)
    invalid_border = img.shape[dim]*limit / 2.0
    
    y1 = invalid_border
    y2 = invalid_border + valid_center

    return y1, y2

def limit_centroid_location(binary_map, centroid, limit):
    valid_centroid = True
    y1_lim, y2_lim = get_valid_centroid_limits(binary_map, dim=0, limit=limit)
    x1_lim, x2_lim = get_valid_centroid_limits(binary_map, dim=1, limit=limit)
    
    if centroid[0] < x1_lim or centroid[0] > x2_lim:
        valid_centroid = False
    if centroid[1] < y1_lim or centroid[1] > y2_lim:
        valid_centroid = False
    return valid_centroid

def resize_imgs(src_imgs, args, interpolation=cv2.INTER_NEAREST):
    # Check resize
    if args.resize_imgs:
        print("\nResizing images...")
        dtype = src_imgs[0].dtype
        convert = dtype != np.uint8
        
        if convert:
            unique_idxs = np.unique(src_imgs)
            src_imgs = [cv2.resize(idx2byte(ele, unique_idxs), 
                                   (args.resize_width, args.resize_height),
                                   interpolation=interpolation) 
                            for ele in src_imgs]
            src_imgs = [byte2idx(ele, unique_idxs).astype(dtype) for ele in src_imgs]
        else:
            src_imgs = [cv2.resize(ele, (args.resize_width, args.resize_height), 
                                   interpolation=interpolation).astype(dtype) 
                            for ele in src_imgs]
        print("Complete.")
    return src_imgs
    
def encode_to_png(src_imgs, args):
    encoded_imgs = [cv2.imencode('.png', ele)[1] for ele in src_imgs]
    encoded_imgs_len = np.array([len(im) for im in encoded_imgs]).astype(np.int64)
    encoded_imgs = np.concatenate(encoded_imgs, axis=0)
    return encoded_imgs, encoded_imgs_len

def log_data(episode_basename_src, object_data, all_imgs, all_zoom_info, sim_history_info, args, paths, check_uint16=False):
    # Check for previous versions
    existant_versions = glob.glob(f"{paths['h5']}/{episode_basename_src}__v*.h5")
    if len(existant_versions) == 0:
        version_num = 1
    else:
        max_version = max([int(ele.split('__v')[1].split("__")[0]) for ele in existant_versions])
        version_num = max_version+1
        
    episode_basename = f'{episode_basename_src}__v{version_num:03d}'
    
    # Write zoom info
    # ===========================================================================
    json_filepath = os.path.join(paths['h5'], f'{episode_basename}__annotations.json')
    assert not os.path.exists(json_filepath), "File already exists! Need to add mechanism to increment filename and not overwrite"
    print(f"\nLogging zoom data to {json_filepath}...")
    with open(json_filepath, 'w') as outfile:
        json.dump(all_zoom_info, outfile)
    # ===========================================================================
    
    # Image data
    # ===========================================================================
    if not args.save_h5:
        print("**WARNING: Not saving h5 data!")
    else:
        h5_save_path = os.path.join(paths['h5'], f'{episode_basename}__imgs.h5')
        assert not os.path.exists(h5_save_path), "File already exists! Need to add mechanism to increment filename and not overwrite"
        print(f"\nLogging image data to {h5_save_path}...")

        # Ensure maximum of uint16
        if check_uint16:
            for ele in all_imgs:
                assert np.max(ele[2]) < 65535 # Maximum of uint16.

        # Encode images
        # --------------------------------------------------------------------
        matterport_RGBs, matterport_RGBs_len = encode_to_png([ele[0] for ele in all_imgs], args)
        instance_segmentations, instance_segmentations_len = encode_to_png([ele[2].astype(np.uint16) for ele in all_imgs], args)
        # --------------------------------------------------------------------

        with h5py.File(h5_save_path, 'w') as outfile:
            outfile.create_dataset('matterport_RGB', data=matterport_RGBs)
            outfile.create_dataset('matterport_RGB_len', data=matterport_RGBs_len)
            outfile.create_dataset('instance_segmentation', data=instance_segmentations)
            outfile.create_dataset('instance_segmentation_len', data=instance_segmentations_len)
            outfile.create_dataset('objects', data=np.string_(json.dumps(object_data)))

            if args.save_habitat_imgs:
                habitat_RGBs, habitat_RGBs_len = encode_to_png([ele[1] for ele in all_imgs], args)
                outfile.create_dataset('habitat_RGB', data=habitat_RGBs)
                outfile.create_dataset('habitat_RGB_len', data=habitat_RGBs_len)
    # ===========================================================================
    
    # Simulation history
    # ===========================================================================
    def jsonify_data(info):
        jsonified_data = {}
        for key1, val in info.items():
            jsonified_data[key1] = {}
            for key2, vals in val.items():
                if vals.dtype in [np.int8, np.int32, np.int64]:
                    jsonified_data[key1][key2] = [int(ele) for ele in vals]
                elif vals.dtype in [np.float16, np.float32, np.float64]:
                    jsonified_data[key1][key2] = [float(ele) for ele in vals]
                else:
                    jsonified_data[key1][key2] = [str(vals) for ele in vals]
        return jsonified_data
    
    if args.save_sim_history_json:
        json_sim_history_info = jsonify_data(sim_history_info)
        sim_history_json_filepath = os.path.join(paths['h5'], f'{episode_basename}__sim_history.json')
        assert not os.path.exists(sim_history_json_filepath), "File already exists! Need to add mechanism to increment filename and not overwrite"
        print(f"\nLogging simulation history data to {sim_history_json_filepath}...")
        with open(sim_history_json_filepath, 'w') as outfile:
            json.dump(json_sim_history_info, outfile)
    # ===========================================================================
    
def np2pt(img_np):
    img_pt = torch.tensor(img_np).permute(2,0,1)
    return img_pt

def make_row_imgs(imgs):
    frame_stack = [np2pt(img) for img in imgs]
    grid_img_i = make_grid(frame_stack, padding=40, pad_value=255).permute(1,2,0).numpy()
    return grid_img_i

def idx2byte(semseg_map, unique_idxs):
    byte_map = np.zeros(semseg_map.shape, dtype=np.uint8)
    for byte_i, instance_idx in enumerate(unique_idxs):
        byte_map[semseg_map==instance_idx] = byte_i
    return byte_map

def byte2idx(byte_map, unique_idxs):
    semseg_map = np.zeros(byte_map.shape, dtype=np.uint32)
    for byte_i, instance_idx in enumerate(unique_idxs):
        semseg_map[byte_map==byte_i] = instance_idx 
    return semseg_map

def make_dir(path, overwrite=False):
    if not os.path.exists(path) or overwrite:
        os.makedirs(path)
        
def compute_new_dimensions(shape, new_width, flip_for_cv2=True):
    aspect_ratio = shape[0] / shape[1] # ar = height / width
    # height = ar * width --> new_height = ar * new_width
    new_height = int(aspect_ratio * new_width)
    new_dim = (new_height, new_width)
    if flip_for_cv2:
        new_dim = (new_dim[1], new_dim[0])
    return new_dim

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def align_images(mat_im, hab_im, sem_im, spec_radius_bounds):
    unique_idxs = np.unique(sem_im)
    
    # Convert semantic imgs to byte
    sem_byte_im = np.repeat(idx2byte(sem_im, unique_idxs)[..., np.newaxis], 3, axis=-1)
        
    # Compute alignment matrix
    M = get_alignment_matrix(hab_im, mat_im, min_match_count=10)
            
    # Catch when no transform is required (due to no alignment possible)
    if M is None:
        warped_sem_im = np.copy(sem_im)
        return warped_sem_im, -1
    
    # Compute spectral radius of transform matrix, M
    spectral_radius = np.max(np.abs(np.linalg.eigvals(M)))
    
    # Spectral radius too skewed - do not apply transform
    if spec_radius_bounds[0] >= spectral_radius >= spec_radius_bounds[1]:
        warped_sem_im = np.copy(sem_im)
        return warped_sem_im, -1
    
    # Warp images
    warped_sem_byte_im = cv2.warpPerspective(sem_byte_im, M, sem_byte_im.shape[:2][::-1], 
                                             cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

    # Convert sem byte back to sem idx
    warped_sem_im = byte2idx(warped_sem_byte_im[:,:,0], unique_idxs)
    
    return warped_sem_im, spectral_radius

def get_connectivity(root, scene_id):
    """Get the connectivity file."""
    connectivity_file = os.path.join(root, f"connectivity/{scene_id}_connectivity.json")
    connectivity = json.load(open(connectivity_file))
    return connectivity

def get_alignment_matrix(img1, img2, min_match_count=10, verbose=False, debug=False):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # See https://stackoverflow.com/questions/25089393/opencv-flannbasedmatcher
    if min([len(kp1), len(kp2)]) < 2:
        return np.eye(3)
    
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    # Set default transform to identity in case of fail
    M = np.eye(3)
    
    # Check counts
    if len(good) >= min_match_count:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        if verbose:
            print("Not enough matches are found - %d/%d" % (len(good),min_match_count))
    
    return M
    
def quaternion_to_euler(qx, qy, qz, qw):
    qx2 = qx**2
    qy2 = qy**2
    qz2 = qz**2
    attitude = np.arcsin(2 * qx * qy + 2 * qz * qw)
    if qx * qy + qz * qw == 0.5:
        # North pole.
        heading = 2 * np.arctan2(x, w)
        bank = 0
    elif qx * qy + qz * qw == -0.5:
        # South pole.
        heading = -2 * np.arctan2(x, w)
        bank = 0
    else:
        heading = np.arctan2(2 * qy * qw - 2 * qx * qz, 1 - 2 * qy2 - 2 * qz2)
        bank = np.arctan2(2 * qx * qw - 2 * qy * qz, 1 - 2 * qx2 - 2 * qz2)
    return heading, attitude, bank

def quaternion_to_euler(quat):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # Needs to be checked for correctness given openGL coordinate system
    q0, q1, q2, q3 = quat
    phi = np.atan2(2*(q0*q1 + q2*q3), 1-2*(q1**2 + q2**2))
    theta = np.asin(2*(q0*q2 - q3*q1))
    psi = np.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    return phi, theta, psi
    
def euler_to_quat(heading, elevation):
    Q1 = habitat_sim.utils.common.quat_from_angle_axis(elevation, np.array([0.0, 0.0, 1.0]))
    Q2 = habitat_sim.utils.common.quat_from_angle_axis(-heading,  np.array([1.0, 0.0, 0.0]))
    Q3 = Q2 * Q1
    quaternion = list(np.array([float(Q3.z), float(Q3.x), float(Q3.y), float(Q3.w)]))
    return quaternion

def euler_to_quaternion(heading, attitude, bank):
    c1 = np.cos(heading / 2)
    c2 = np.cos(attitude / 2)
    c3 = np.cos(bank / 2)
    s1 = np.sin(heading / 2)
    s2 = np.sin(attitude / 2)
    s3 = np.sin(bank / 2)
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * s2 * c3 + c1 * c2 * s3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    #   print(x, y, z, w)
    return x, y, z, w

def rotation_to_quaternion(rot):
    m00 = rot[0, 0]
    m11 = rot[1, 1]
    m22 = rot[2, 2]
    m12 = rot[1, 2]
    m21 = rot[2, 1]
    m02 = rot[0, 2]
    m20 = rot[2, 0]
    m10 = rot[1, 0]
    m01 = rot[0, 1]
    w = np.sqrt(1.0 + m00 + m11 + m22) / 2.0
    w4 = (4.0 * w)
    x = (m21 - m12) / w4
    y = (m02 - m20) / w4
    z = (m10 - m01) / w4
    return x, y, z, w
