"""Basic utilities for visualization.
Authors: 
 - Mengye Ren (mren@cs.toronto.edu)
 - Michael Iuzzolino (michael.iuzzolino@colorado.edu)
"""
import os
import sys
import cv2
import subprocess
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

try:
    import IPython.display
    from IPython.display import HTML
except:
    print("Could not import IPython")

try:
    from habitat_sim.utils.common import d3_40_colors_rgb
except:
    print("Could not import habitat_sim.utils.common")
    
blend_images = lambda img1, img2, mix_alpha : (img1*(1-mix_alpha) + img2*mix_alpha).byte()

def plot_centroid_boundary(img_src, limit=0.3, color=(0,255,0)):
    def get_valid_limits(img, dim, limit):
        valid_center = img.shape[dim]*(1-limit)
        invalid_border = img.shape[dim]*limit / 2.0
        y1 = invalid_border
        y2 = invalid_border + valid_center
        return y1, y2

    img = img_src.copy()
    y1_lim, y2_lim = get_valid_limits(img, dim=0, limit=limit)
    y1_lim, y2_lim = int(y1_lim), int(y2_lim)
    x1_lim, x2_lim = get_valid_limits(img, dim=1, limit=limit)
    x1_lim, x2_lim = int(x1_lim), int(x2_lim)
    img_i = cv2.rectangle(img, (x1_lim, y1_lim), (x2_lim, y2_lim), color, 1)
    plt.figure(figsize=(12,12))
    plt.imshow(img)
    plt.axis('off')
    
def colorize_img(img):
    img = (np.repeat(img[..., np.newaxis], 3, axis=-1)*255).astype(np.uint8)
    return img

# --------------------------------------------
# Reference: https://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
def drawline(img,pt1,pt2,color,thickness,gap):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    for p in pts:
        cv2.circle(img,p,thickness,color,-1)

def drawpoly(img,pts,color,thickness,gap):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,gap)

def draw_dotted_rect(img, bbox, color, thickness=1, gap=5):
    y1, y2, x1, x2 = bbox
    pt1 = (x1, y2)
    pt2 = (x2, y1)
    dotted_img = img.copy()
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(dotted_img,pts,color,thickness,gap)
    return dotted_img
# --------------------------------------------

def draw_bbox(img_src, bbox=None, bbox_thickness=2, centroid=None, draw_centroid=False, 
              color=(0,255,0), resize=False, resize_dim=(800,600), style='solid', gap=5):
    img = img_src.copy()
    original_size = img.shape[:2]
    if resize:
        img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_NEAREST)
    
    if draw_centroid:
        if centroid is None:
            img = cv2.circle(img, (img.shape[1]//2, img.shape[0]//2), 20, (255, 0, 0) , -1)
        else:
            img = cv2.circle(img, (int(centroid[0]), int(centroid[1])), 20, (255, 0, 0) , -1)
    if bbox is None:
        img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), bbox_thickness)
    else:
        y1, y2, x1, x2 = bbox
        if style == 'solid':
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, bbox_thickness) 
        elif style == 'dotted':
            img = draw_dotted_rect(img, bbox, color, thickness=bbox_thickness, gap=gap)
            
    if resize:
        img = cv2.resize(img, original_size[::-1], interpolation=cv2.INTER_NEAREST)
    return img

def add_text(img_src, text, x, y, thickness=3, font_scale=1, color=(255,255,255), resize=False, resize_dim=(800,600)):
    img = img_src.copy()
    original_size = img.shape[:2]
    if resize:
        img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_NEAREST)
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    img = cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA) 
    if resize:
        img = cv2.resize(img, original_size[::-1], interpolation=cv2.INTER_NEAREST)
    return img

def add_labels_to_img(img, text, fontsize=5, x=0, y=50, text_color=None):
    if text_color is None:
        text_color = [255, 255, 255]
    
    if isinstance(img, np.ndarray):
        img_i = np.copy(img)
    else:
        img_i = img.permute(1,2,0).detach().numpy().copy()
        
    cv2.putText(img_i, text, (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontsize, 
                text_color, 
                thickness=3)
    
    if isinstance(img, torch.Tensor):
        img_i = torch.tensor(img_i).permute(2,0,1)
    return img_i

def annotate_locations_on_img(locations, img, params):
    img = img.copy()
    for idx, loc in enumerate(locations[1:]):
        # Default text color
        text_color = [0,0,255]
        
        # Check for limitations and do not show them on image if condition met
        if 'max_next_loc_rel_heading' in params:
            if np.abs(loc.rel_heading) > params['max_next_loc_rel_heading']:
                text_color = [255,0,0]
        if 'max_next_loc_rel_elevation' in params:
            if np.abs(loc.rel_elevation) > params['max_next_loc_rel_elevation']:
                text_color = [255,0,0]

        # Draw actions on the screen
        fontScale = 3.0 / loc.rel_distance
        x = int(params['width'] / 2 + loc.rel_heading / params['hfov_RAD'] * params['width'])
        y = int(params['height'] / 2 - loc.rel_elevation / params['vfov_RAD'] * params['height'])

        cv2.putText(img, str(idx + 1), (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale, 
                    text_color, 
                    thickness=3)

    return img

def color_semantic_image(semantic_obs, to_numpy=True):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
   
    if to_numpy:
        semantic_img = np.array(semantic_img)[:,:,:3]
    
    return semantic_img

class VideoMaker:
    def __init__(self, outpath, verbose=False):
        self.outpath = outpath
        self.verbose = verbose
    
    def init(self, width, height, fps=25):
        if self.verbose:
            print(f"\nSaving to {outpath}")
        self.height = height
        self.width = width
        fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
        self.video = cv2.VideoWriter(self.outpath, fourcc, fps, (height, width), True)
        
    def write(self, frame, flip_RGB=True):
        if flip_RGB:
            frame = frame[...,::-1]
        self.video.write(frame)
    
    def finish(self):
        print("\nClosing video stream.")
        self.video.release()
            
def vid2imgs(video_path):
    cap = cv2.VideoCapture(video_path)
    imgs = []
    frame_i = 0
    while cap.isOpened():
        print(f"Reading frame {frame_i}...")
        ret, frame = cap.read()
        imgs.append(frame)
        frame_i += 1
    return imgs

def imgs2vid(imgs, outpath, fps=25):
    print(f"\nWriting videos to {outpath}...")
    height, width = imgs[0].shape[0:2]
        
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)
    
    for i, img in enumerate(imgs):
        sys.stdout.write(f"\rWriting frame {i+1}/{len(imgs)}...")
        sys.stdout.flush()
        video.write(img)
    
    video.release()
    print("\nComplete.")
    
def toAnimation(imgs, *args, **kwargs):
    animator = FrameAnimator(imgs, *args, **kwargs)
    
    animation_obj = animator()
    if kwargs.get("savepath", False):
        save_path = kwargs['savepath']
        print(f"Saving to {save_path}...")
        animator.save(save_path, kwargs.get('fps', 15))
    
    print("Fin.")
    return animation_obj
    
class FrameAnimator(object):
    def __init__(self, imgs, interval=100, blit=True, text=None, figsize=(6,6), mode='jsHTML', **kwargs):
        self.mode = mode
        
        # Compute cmap
        # ----------------------------------------------------
        if len(imgs[0].shape) == 2 or imgs[0].shape[-1] == 1:
            cmap = 'gray'
        else:
            cmap = None
        # ----------------------------------------------------
        
        # Setup figure
        fig, ax = plt.subplots(1, figsize=figsize)
        
        # Init image 0
        img_0 = imgs[0]
        if img_0.shape[-1] == 1:
            img_0 = np.repeat(img_0, 3, axis=2)
            
        video_img = ax.imshow(img_0, cmap=cmap)
        ax.axis('off')

        def init():
            img_0 = imgs[0]
            if img_0.shape[-1] == 1:
                img_0 = np.repeat(img_0, 3, axis=2)
            video_img.set_array(np.zeros_like(img_0))
            return (video_img,)

        def animate(i):
            img_i= imgs[i]
            if img_i.shape[-1] == 1:
                img_i = np.repeat(img_i, 3, axis=2)
            
            video_img.set_array(img_i)
            if isinstance(text, str):
                text_str = f' -- {text}' if text is not None else ""
            elif isinstance(text, list):
                text_str = f' -- {text[i]}' if text is not None else ""
            else:
                text_str = ''
            ax.set_title(f"Frame {i+1}/{len(imgs)} {text_str}")
            
            return (video_img,)
        
        plt.tight_layout()
        plt.close() # prevent fig from showing
        
        print("Building animator...")
        self.anim = animation.FuncAnimation(fig, animate, 
                                       init_func=init, 
                                       frames=len(imgs),
                                       interval=interval, # delay between frames in ms (25FPS=25 f/s * 1 s/1000ms = 0.025 f/ms)
                                       blit=blit)
        
    def __call__(self):
        print("Generating animation object...")
        if self.mode == 'HTML':
            anim_obj = HTML(self.anim.to_html5_video())
        elif self.mode == 'jsHTML':
            anim_obj = HTML(self.anim.to_jshtml())
        return anim_obj
    
    def _init_mp4_writer(self, fps):
        Writer = animation.writers['ffmpeg']
        self.writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        
    def save(self, save_path, fps=15): 
        if save_path.endswith('gif'):
            self.anim.save(save_path, writer='imagemagick', fps=fps)
        else:
            self._init_mp4_writer(fps)
            self.anim.save(save_path, writer=self.writer)