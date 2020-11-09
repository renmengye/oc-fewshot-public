import numpy as np
import cv2
import torch
import modules.utils
import modules.postproc_utils

def zoom_in_on_img(habitat_img, matterport_img, binary_map, semseg_map, object_centroid, unique_idxs, diff_factor=0.5, r=0.75):
    h, w = habitat_img.shape[:2]
    
    # Compute image centroid
    img_centroid = np.array(habitat_img.shape[:2])/2
    
    # Compute difference between img and object centroids
    centroid_delta = (object_centroid[::-1] - img_centroid) * diff_factor
    
    # Compute new image centroid
    new_centroid_y, new_centroid_x = img_centroid + centroid_delta
    
    # Compute new height and width by reduction factor, r
    new_h = h * r
    new_w = w * r
    
    # Delta h, w
    dh, dw = new_h/2, new_w/2
    # Crop
    y1 = max(0, int(new_centroid_y-dh))
    y2 = min(int(new_centroid_y+dh), h)
    x1 = max(0, int(new_centroid_x-dw))
    x2 = min(int(new_centroid_x+dw), w)
    bbox = [y1, y2, x1, x2]
    zoom_rgb = habitat_img[y1:y2, x1:x2]
    zoom_rgb = cv2.resize(zoom_rgb, (w, h))
    
    # Matterport zooming
    # --------------------------------------------------
    zoomed_matterport_img = matterport_img[y1:y2, x1:x2]
    zoomed_matterport_img = cv2.resize(zoomed_matterport_img, (w, h))
    # --------------------------------------------------
    
    # Binary map zooming
    # --------------------------------------------------
    zoomed_binary_map = binary_map[y1:y2, x1:x2]
    zoomed_binary_map = cv2.resize(zoomed_binary_map, (w, h), interpolation=cv2.INTER_NEAREST)
    # --------------------------------------------------
    
    # Semseg map zooming
    # ------------------------------------------------------------------------
    # Remap ids into 0-255 for resizing
    semseg_byte_map = modules.utils.idx2byte(semseg_map, unique_idxs)
    
    # Resize semseg
    semseg_byte_map = semseg_byte_map[y1:y2, x1:x2]
    semseg_byte_map = cv2.resize(semseg_byte_map, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Remap 0-255 into ids
    zoomed_semseg = modules.utils.byte2idx(semseg_byte_map, unique_idxs)
    # ------------------------------------------------------------------------
    return bbox, [zoom_rgb, zoomed_matterport_img, zoomed_binary_map, zoomed_semseg]

def zoom_on_object(valid_idx, habitat_img, matterport_img, binary_map, semseg_map, unique_idxs, n_zooms, 
                   min_area_fxn, max_area_fxn, zoom_out=True,
                   apply_attention_mask=True, apply_attention_mask_recursive=True,
                   brightness_delta=0.9, recursive_brightness_delta=0.9, spec_radius_bounds=(0.95, 1.15)):
    
    # Apply attention 
    if apply_attention_mask:
        obj_mask = semseg_map==valid_idx
        habitat_img = modules.postproc_utils.apply_attention(np.copy(habitat_img), obj_mask, brightness_delta)
        matterport_img = modules.postproc_utils.apply_attention(np.copy(matterport_img), obj_mask, brightness_delta)
        
    # Zoom
    all_zooms = []
    Ms = []
    for n_zoom in range(n_zooms):
        centroid = modules.postproc_utils.get_centroid(binary_map, min_area_fxn, max_area_fxn)
        if centroid is None:
            break
        
        # Warp images
        # ------------------------------------------------
        # Make sem into RGB
        sem_RGB_im = modules.vis_utils.color_semantic_image(semseg_map)
        sem_RGB_im = torch.tensor(sem_RGB_im).permute(2,0,1) 

        # Convert semantic imgs to byte
        semantic_byte_img = np.repeat(modules.utils.idx2byte(semseg_map, unique_idxs)[..., np.newaxis], 3, axis=-1)
        sem_byte_im = torch.tensor(semantic_byte_img).permute(2,0,1)
        mat_im_pt = torch.tensor(matterport_img).permute(2,0,1)
        hab_im_pt = torch.tensor(habitat_img).permute(2,0,1)

        attention_im = np.repeat(binary_map[..., np.newaxis], 3, axis=-1)
        attention_im_pt = torch.tensor(attention_im).permute(2,0,1)
        
        # Warp sem
        warp_results = modules.utils.align_imgs(mat_im_pt, hab_im_pt, sem_byte_im, sem_RGB_im,
                                                min_sr=spec_radius_bounds[0], max_sr=spec_radius_bounds[1],
                                                attention_im=attention_im_pt)

        M, warped_hab_im, warped_sem_byte_im, warped_sem_RGB_im, warped_attention_im = warp_results
        Ms.append(M)
        
        habitat_img = warped_hab_im.permute(1,2,0).numpy()
        warped_sem_byte_im = warped_sem_byte_im.permute(1,2,0).numpy()[:,:,0]
        semseg_map = modules.utils.byte2idx(warped_sem_byte_im, unique_idxs)
        binary_map = warped_attention_im.permute(1,2,0).numpy()[:,:,0]
        # ------------------------------------------------
        
        bbox, zoom_result = zoom_in_on_img(habitat_img, matterport_img, binary_map, semseg_map, centroid, unique_idxs)
        all_zooms.append(zoom_result)
        
        # Update
        habitat_img, matterport_img, binary_map, semseg_map = zoom_result
        
        # Apply attention 
        if apply_attention_mask_recursive:
            obj_mask = semseg_map==valid_idx
            habitat_img = modules.postproc_utils.apply_attention(np.copy(habitat_img), obj_mask, recursive_brightness_delta)
            matterport_img = modules.postproc_utils.apply_attention(np.copy(matterport_img), 
                                                                    obj_mask, 
                                                                    recursive_brightness_delta)

    # Reflect zooms for zoom-out
    if zoom_out:
        n_actual_zooms = len(all_zooms)
        idxs = list(range(n_actual_zooms)) + list(range(n_actual_zooms-1))[::-1]
        all_zooms = [all_zooms[idx] for idx in idxs]
        
    return all_zooms, Ms

def zoom_on_object__bbox(valid_idx, habitat_img, matterport_img, binary_map, semseg_map, unique_idxs, n_zooms, 
                         min_area_fxn, max_area_fxn, zoom_out=True,
                         apply_attention_mask=True, apply_attention_mask_recursive=True,
                         brightness_delta=0.9, recursive_brightness_delta=0.9, spec_radius_bounds=(0.95, 1.15)):
    
    # Apply attention 
    if apply_attention_mask:
        obj_mask = semseg_map==valid_idx
        habitat_img = modules.postproc_utils.apply_attention(np.copy(habitat_img), obj_mask, brightness_delta)
        matterport_img = modules.postproc_utils.apply_attention(np.copy(matterport_img), obj_mask, brightness_delta)
        
    # Zoom
    all_zooms = []
    all_bboxes = []
    Ms = []
    for n_zoom in range(n_zooms):
        centroid = modules.postproc_utils.get_centroid(binary_map, min_area_fxn, max_area_fxn)
        if centroid is None:
            break
        
        # Warp images
        # ------------------------------------------------
        # Make sem into RGB
        sem_RGB_im = modules.vis_utils.color_semantic_image(semseg_map)
        sem_RGB_im = torch.tensor(sem_RGB_im).permute(2,0,1) 

        # Convert semantic imgs to byte
        semantic_byte_img = np.repeat(modules.utils.idx2byte(semseg_map, unique_idxs)[..., np.newaxis], 3, axis=-1)
        sem_byte_im = torch.tensor(semantic_byte_img).permute(2,0,1)
        mat_im_pt = torch.tensor(matterport_img).permute(2,0,1)
        hab_im_pt = torch.tensor(habitat_img).permute(2,0,1)

        attention_im = np.repeat(binary_map[..., np.newaxis], 3, axis=-1)
        attention_im_pt = torch.tensor(attention_im).permute(2,0,1)
        
        # Warp sem
        warp_results = modules.utils.align_imgs(mat_im_pt, hab_im_pt, sem_byte_im, sem_RGB_im,
                                                min_sr=spec_radius_bounds[0], max_sr=spec_radius_bounds[1],
                                                attention_im=attention_im_pt)

        M, warped_hab_im, warped_sem_byte_im, warped_sem_RGB_im, warped_attention_im = warp_results
        Ms.append(M)
        
        habitat_img = warped_hab_im.permute(1,2,0).numpy()
        warped_sem_byte_im = warped_sem_byte_im.permute(1,2,0).numpy()[:,:,0]
        semseg_map = modules.utils.byte2idx(warped_sem_byte_im, unique_idxs)
        binary_map = warped_attention_im.permute(1,2,0).numpy()[:,:,0]
        # ------------------------------------------------
        
        bbox, zoom_result = zoom_in_on_img(habitat_img, matterport_img, binary_map, semseg_map, centroid, unique_idxs)
        all_zooms.append(zoom_result)
        all_bboxes.append(bbox)
        
        # Update
        habitat_img, matterport_img, binary_map, semseg_map = zoom_result
        
        # Apply attention 
        if apply_attention_mask_recursive:
            obj_mask = semseg_map==valid_idx
            habitat_img = modules.postproc_utils.apply_attention(np.copy(habitat_img), obj_mask, recursive_brightness_delta)
            matterport_img = modules.postproc_utils.apply_attention(np.copy(matterport_img), 
                                                                    obj_mask, 
                                                                    recursive_brightness_delta)

    # Reflect zooms for zoom-out
    if zoom_out:
        n_actual_zooms = len(all_zooms)
        idxs = list(range(n_actual_zooms)) + list(range(n_actual_zooms-1))[::-1]
        all_zooms = [all_zooms[idx] for idx in idxs]
        all_bboxes = [all_bboxes[idx] for idx in idxs]
        
    return all_bboxes, all_zooms, Ms