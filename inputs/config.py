import os
from inputs.env import Environment
import math

################### DATA SET SELECTION ###################
env_name = '30-60-90'
# env_name = '30-60-90+equilateral'
# env_name = '30-60-90+rectangle'
# env_name = "polyiamond-3"
# env_name = "45-45-90+rectangle"
# env_name = "labyrinth"

################### DATA SET CONFIGURATION ###################
# format: "tile set name : (mirror all tiles?, size of the superset, number of training data)"
env_attribute_dict = {
    '30-60-90'             : (True,  9, 10),
    '30-60-90+equilateral' : (True,  9, 12000),
    '30-60-90+rectangle'   : (True,  7, 12000),
    '45-45-90+rectangle'   : (False, 9,7000),
    "labyrinth"            : (False, 9, 5000),
    "polyiamond-3"         : (False, 9, 12000),
}

symmetry_tiles, complete_graph_size, number_of_data = env_attribute_dict[env_name]
env_location = os.path.join('.', 'data', env_name)
environment = Environment(env_location, symmetry_tiles=symmetry_tiles)
# SET YOUR DATASET PATH HERE
dataset_path = os.path.join('./dataset', f"{env_name}-ring{complete_graph_size}-{number_of_data}")

################### CREATING DATA ###################
shape_size_lower_bound=0.4
shape_size_upper_bound=0.6
max_vertices=20
validation_data_proportion=0.2

################### NETWORK PARAMETERS ###################
network_depth = 20
middle_feature_size = 32
with_collision_branch = True

################### TRAINING ###################
new_training = True
rand_seed = 2
batch_size = 1
learning_rate = 1e-3
training_epoch = 10000
save_model_per_epoch = 2
optimizer_path = None

COLLISION_WEIGHT    = 1/math.log(1+1e-1)
ALIGN_LENGTH_WEIGHT = 0.02
AVG_AREA_WEIGHT     = 1
DIVERSITY_WEIGHT = 0.0

EDWNET_PREFIX = 'graph_networks.networks.'
EDWNET_NAME = 'EDWNet'
LOSSES_PREFIX = 'solver.ml_solver.'
LOSSES_NAME = 'losses'

#sampling exp
training_by_sampling = False
epsilon = 0.00
greedy_sampling = False

training_experiment = False


################### DEBUGGING ###################
debug_data_num = 5
search_time_limit = complete_graph_size * 2
debug_base_folder = ".."
debug_id = 2000

#################### EVALUATION ##################
start_angle = 0.0
end_angle = 30
num_of_angle = 2
movement_delta_ratio = [0]
margin_padding_ratios = [0.5]
evaluation_search_time_limit = 200
top_k = 1
max_axis = 160
plot_target = True
plot_sequence = False
output_tree_search_layout = False
save_objs = False
save_tiling_pictures = True
save_layout = True
trial_times = 20
detect_holes = True
silhouette_path = "/home/edwardhui/data/silhouette/selected_v2"
network_path = f"./pre-trained_models/{env_name}.pth"
start_file_name = None
sort_by_coverage = False
sort_keep_num = 10