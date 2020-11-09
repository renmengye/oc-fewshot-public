import os
import glob
import math
import numpy as np

mode = 'full' # 'test_elevation', 'test_heading', 'full'

# Annotate on matterport sim images
annotate_navigable_locs = False

# Simulator sensor params
depth_on = False
habitat_sim_hfov_deg = 90
matterport_hfov_deg = 100

# Movement
scan_refractory_period = 0
movement_refractory_period = 5
movement_prob = 1.0

# Elevation (up-down scanning)
elevation_delta_prob = 0.95
elevation_delta = math.radians(5)
min_elevation_angle = -math.radians(2.5)
max_elevation_angle = math.radians(2.5)
max_next_loc_rel_elevation = math.radians(10)

# Heading (left-right scanning)
heading_delta_prob = 1 #0.95
heading_delta_mean = math.radians(12.5)
heading_delta_var = math.radians(0.5)
n_scan_steps = int(2*np.pi / heading_delta_mean)
max_next_loc_rel_heading = math.radians(25)

