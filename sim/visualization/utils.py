import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

def hex2rgb(value, reverse=True):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    if reverse:
        rgb = rgb[::-1]
    return rgb
    
def compute_max_label_width(it, episode_limit, font_scale, font_thickness):
    print("Generating all instance labels...")
    label_sizes = []
    for file_i, episode_data in zip(range(episode_limit), it):
        sys.stdout.write(f'\rProcessing file {file_i+1}/{episode_limit}...')
        sys.stdout.flush()
        categories = [ele.decode() for ele in episode_data['category'][0].numpy()]
        instance_ids = [ele.decode() for ele in episode_data['instance_id'][0].numpy()]
        
        categories = [ele.replace("_", "") for ele in categories]
        
        get_size = lambda label, i : cv2.getTextSize(label, 
                                                     cv2.FONT_HERSHEY_SIMPLEX, 
                                                     font_scale, 
                                                     font_thickness)[0][i]
        label_sizes += [get_size(f"{category} {instance_id}", 0)
                            for category, instance_id in zip(categories, instance_ids)]
    max_label_width = int(np.max(label_sizes))
    print(f"\nMax width: {max_label_width}")
    return max_label_width

def renorm_img(img):
    renormed_img = img.copy()
    renormed_img = (renormed_img + 1) / 2.0 * 255
    renormed_img = renormed_img.astype(np.uint8)
    renormed_img = renormed_img[...,::-1]
    return renormed_img

def highlight(vis_utils, img, highlight_thickness, bbox_color):
    y1, y2 = 0, img.shape[0]
    x1, x2 = 0, img.shape[1] 
    bbox = [y1, y2, x1, x2]
    img = vis_utils.draw_bbox(img, 
                              bbox=bbox, 
                              bbox_thickness=highlight_thickness, 
                              color=bbox_color)
    return img

def add_instance_bank_img_text(img, label, vis_utils, font_scale=0.6, font_thickness=1):
    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x_offset = 2
    text_y_offset = 16
    color = (255, 255, 255)
   
    img = add_background(img, 22, img.shape[1], p=0.5)
    img = vis_utils.add_text(img, 
                             label, 
                             x=text_x_offset, 
                             y=text_y_offset, 
                             color=color,
                             thickness=font_thickness, 
                             font_scale=font_scale)
    return img

def generate_grid_bank(instance_bank, vis_config, vis_utils, seen_color,
                       instance_id=None, test_grid=False):
    def activate(im, yes):
        if yes:
            im = highlight(vis_utils, im, vis_config._HIGHLIGHT_THICKNESS, seen_color)
        return im
    max_grid_imgs = vis_config._GRID_PARAMS['nrow'] * vis_config._GRID_PARAMS['ncol']
    instance_base_imgs = [activate(val[0], key==instance_id) 
                              for key, val in instance_bank.items()][:max_grid_imgs]
    labels = [key for key in instance_bank.keys()][:max_grid_imgs]
    
    for i, (img, label) in enumerate(zip(instance_base_imgs, labels)):
        instance_base_imgs[i] = add_instance_bank_img_text(img, label, vis_utils)
    
    if len(instance_base_imgs) > 0:
        if test_grid:
            product = vis_config._GRID_PARAMS['nrow']*vis_config._GRID_PARAMS['ncol']
            instance_base_imgs = [instance_base_imgs[0] for _ in range(product)]
        
        prev_img_pt = torch.tensor(instance_base_imgs).permute(0,3,1,2)
        instance_grid = make_grid(prev_img_pt, 
                                  nrow=vis_config._GRID_PARAMS['nrow'], 
                                  padding=vis_config._GRID_PARAMS['padding'],
                                  pad_value=vis_config._GRID_PARAMS['pad_val'])
        instance_grid = instance_grid.permute(1,2,0).numpy()
        return instance_grid
    else:
        return None

def generate_instance_row(instance_array, params, resize_dim):
    # Instance samples
    instance_choices = np.array(instance_array)
    n_instances = len(instance_choices)
    
    # Limit to min available
    n_to_display = min(n_instances, params['n_display'])
    
    # Randomly sample N instances
    sample_idxs = np.random.choice(list(range(n_instances)), n_to_display, replace=False)
    sampled_instances = instance_choices[sample_idxs]
    
    sampled_instances = [cv2.resize(img, resize_dim, interpolation=cv2.INTER_NEAREST) 
                             for img in sampled_instances]
    
    # Convert to tensor
    prev_img_pt = torch.tensor(sampled_instances).permute(0,3,1,2)
    
    # Make grid
    instance_row = make_grid(prev_img_pt, nrow=params['n_display'], 
                             padding=params['padding'], pad_value=params['pad_val'])
    instance_row = instance_row.permute(1,2,0).numpy()
    return instance_row

def add_to_base_img(base_img, add_img, params):
    if add_img is None:
        return base_img
    h, w = add_img.shape[:2]
    x_offset = (base_img.shape[1] - w)//2
    x1, x2 = x_offset, x_offset+w
    y1, y2 = params['y_offset'], params['y_offset']+h
    base_img[y1:y2,x1:x2] = add_img
    return base_img

def add_background(img_src, blackout_h, blackout_w, show_model_info=True, p=0.5):
    if not show_model_info:
        return img_src
    img = img_src.copy()
    if blackout_w > img.shape[1]:
        blackout_w = img.shape[1]
        
    sub_img = img[:blackout_h,:blackout_w]
    black_rect = np.zeros((blackout_h, blackout_w, img.shape[-1]), dtype=img.dtype) * 255
    res = cv2.addWeighted(sub_img, 1-p, black_rect, p, 1.0)
    img[:blackout_h,:blackout_w] = res
    return img

def add_rect(img_src, color, start_point=(0,0), end_point=(0,40), thickness=2):
    img = img_src.copy()
    img = cv2.rectangle(img, start_point, end_point, color, thickness)
    return img

def add_line(img_src, color, start_point=(0,0), end_point=(0,40), thickness=5):
    img = img_src.copy()
    img = cv2.line(img, start_point, end_point, color, thickness)
    return img

def add_legend(vis_utils, img_src, legend_line_colors, show_model_info=True, p=0.8):
    img = img_src.copy()
    blackout_w = img.shape[1]
    blackout_h = 25
    
    sub_img = img[-blackout_h:,:blackout_w]
    black_rect = np.zeros((blackout_h, blackout_w, img.shape[-1]), dtype=img.dtype) * 255
    res = cv2.addWeighted(sub_img, 1-p, black_rect, p, 1.0)
    img[-blackout_h:,:blackout_w] = res
    
    color1, color2 = legend_line_colors
    line_length = 50
    y_line_offset = img.shape[0]-12
    y_text_offset = img.shape[0]-5
    
    fontscale = 0.7
    
    # label 1
    # ----------------------------------------------------------------
    x_offset = 10
    img = add_rect(img, color1, start_point=(x_offset, y_line_offset-5), end_point=(x_offset+line_length, y_line_offset+5))
    img = vis_utils.add_text(img.copy(), 
                             'No Label', 
                             x=x_offset+line_length+10, 
                             y=y_text_offset, 
                             color=color1,
                             thickness=2, 
                             font_scale=fontscale)
    # ----------------------------------------------------------------
    
    # label 2
    # ----------------------------------------------------------------
    x_offset = 230
    img = add_rect(img, color2, start_point=(x_offset, y_line_offset-5), end_point=(x_offset+line_length, y_line_offset+5))
    img = vis_utils.add_text(img.copy(), 
                             'Labeled', 
                             x=x_offset+line_length+10, 
                             y=y_text_offset, 
                             color=color2,
                             thickness=2, 
                             font_scale=fontscale)
    # ----------------------------------------------------------------
    
    if show_model_info:
        # label 3
        # ----------------------------------------------------------------
        x_offset = 460
        img = vis_utils.add_text(img.copy(), 
                                 'Correct Prediction', 
                                 x=x_offset+line_length+10, 
                                 y=y_text_offset, 
                                 color=(0,255,0),
                                 thickness=2, 
                                 font_scale=fontscale)
        # ----------------------------------------------------------------

        # label 4
        # ----------------------------------------------------------------
        x_offset = 720
        img = vis_utils.add_text(img.copy(), 
                                 'Incorrect Prediction', 
                                 x=x_offset+line_length+10, 
                                 y=y_text_offset, 
                                 color=(255,0,255),
                                 thickness=2, 
                                 font_scale=fontscale)
        # ----------------------------------------------------------------
    
    
    return img
    
    
def add_primary_baseline_text(vis_utils, vis_config, img, actual_label, predicted_class_label, correct_prediction,
                              text_x, text_y, font_thickness, font_scale, 
                              blackout_w, blackout_h, unlabeled_key, legend_line_colors, show_model_info=True):
    
    # Add background box
    img = add_background(img, blackout_h, blackout_w, show_model_info=show_model_info)

    # Add legend
    img = add_legend(vis_utils, img, legend_line_colors, show_model_info=show_model_info)
    
    # Add text
    if show_model_info:
        _FIXED_X_OFFSET = 20
        _USED_FIXED_X_OFFSET = True
        textsize = cv2.getTextSize(actual_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x_offset = _FIXED_X_OFFSET if _USED_FIXED_X_OFFSET else (blackout_w - textsize[0]) // 2
        text_y_offset = (blackout_h - textsize[1]) // 6

        img = vis_utils.add_text(img, 
                                 actual_label, 
                                 x=text_x_offset, 
                                 y=text_y_offset+text_y, 
                                 color=(255,255,255),
                                 thickness=font_thickness, 
                                 font_scale=font_scale)

        textsize = cv2.getTextSize(predicted_class_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_y_offset = (blackout_h - textsize[1]) // 6 * 5.6
        text_y_offset = int(text_y_offset)

        # Set text color
        color = (0, 255, 0) if correct_prediction else (255, 0, 255) # (b, g, r)
        img = vis_utils.add_text(img, 
                                 predicted_class_label, 
                                 x=text_x_offset, 
                                 y=text_y_offset+text_y, 
                                 color=color,
                                 thickness=font_thickness, 
                                 font_scale=font_scale)

    return img

def get_text_centering_offset(text, img_w):
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    x_offset = (img_w - textsize[0])//2
    return x_offset

def add_grid_bank_baseline_text(vis_utils, img, num_instances, num_occurences, actual_label, 
                                text_x, text_y1, text_y2, font_thickness, font_scale):
    
    text_upper = f'Instance Bank - Instances Tracked: {num_instances}'
    text_x_offset = get_text_centering_offset(text_upper, img.shape[1])
    img = vis_utils.add_text(img, text_upper, color=(0,0,0), 
                             x=text_x_offset, y=text_y1, 
                             thickness=font_thickness, font_scale=font_scale)
    
    text_lower = f'{actual_label} - Occurrences: {num_occurences}'
    text_x_offset = get_text_centering_offset(text_lower, img.shape[1])
    img = vis_utils.add_text(img, text_lower, color=(0,0,0), 
                             x=text_x_offset, y=text_y2, 
                             thickness=font_thickness, font_scale=font_scale)

    return img