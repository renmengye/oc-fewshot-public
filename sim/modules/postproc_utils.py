import numpy as np
import cv2

def compute_objects_dict(sim_data):
    objects = sim_data["semantic"]#np.string_(json.dumps(sim_data["semantic"]))
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

def apply_attention(img, target_obj_mask, brightness_delta):
    attention_img = np.copy(img[:,:,:3])
    attention_img[~target_obj_mask] = change_brightness(np.copy(attention_img), value=brightness_delta)[~target_obj_mask]
    return attention_img

def get_centroid(binary_map, _MIN_AREA_THRESHOLD, _MAX_AREA_THRESHOLD):
    n_labels, labels, stats, centroids = run_connected_components(binary_map)
    areas = stats[:,-1]
    good_idxs = np.where((areas>_MIN_AREA_THRESHOLD(binary_map)) & (areas<_MAX_AREA_THRESHOLD(binary_map)))
    centroids = centroids[good_idxs]
    if len(centroids) > 1:
        centroids = [centroids[np.argmax(areas)]]
    
    if len(centroids) == 0:
        centroid = None
    else:
        # Grab centroid from list
        centroid = centroids[0]
    return centroid

def run_connected_components(binary_map, connectivity=4):
    # Perform the operation
    # Output: num_labels, label matrix, stat matrix, centroid matrix
    output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    return output