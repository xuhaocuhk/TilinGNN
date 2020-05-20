import numpy as np
import math
from collections import deque
import random
import torch
import glob
import os

# return the rotation matrix and the translation matrix
# base_segment:[(x1,y1), (x2, y2)] look left is inside of the contour
# tile_segment:[(x1,y1), (x2, y2)] look left is inside of the tile
def align(base_segment_p0, base_segment_p1, tile_segment_p0, tile_segment_p1, mode: int):
    assert mode == 0 or mode == 1

    base_vec  = np.array([base_segment_p1[0] - base_segment_p0[0], base_segment_p1[1] - base_segment_p0[1] , 0])
    align_vec = np.array([tile_segment_p1[0] - tile_segment_p0[0], tile_segment_p1[1] - tile_segment_p0[1] , 0])
    rot_axis = np.cross(base_vec, align_vec)
    cos = np.dot(align_vec, base_vec) / (np.linalg.norm(align_vec) * np.linalg.norm(base_vec))
    cos = np.clip(cos, -1, 1)
    theta = math.acos(cos)
    if rot_axis[2] > 0:
        theta = -theta

    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

    tile_p0_rotate = np.dot(R, tile_segment_p0)
    tile_p1_rotate = np.dot(R, tile_segment_p1)

    if mode == 0:
        T = base_segment_p0 - tile_p0_rotate
    elif mode == 1:
        T = base_segment_p1 - tile_p1_rotate

    return R, T

def create_inputs(state):
    segment_state = state[0]
    length = segment_state.shape[0]
    mask = np.ones(length, dtype=int)
    triangle = np.reshape(np.array(state[1]), -1)
    return (segment_state, length, mask, triangle)

# angle between seg1 and seg2(right of seg1)
def seg_angle(seg1, seg2):
    assert seg1[1][0] == seg2[0][0] and seg1[1][1] == seg2[0][1] # clockwise
    sin_theta = np.cross([seg1[0][0] - seg1[1][0], seg1[0][1] - seg1[1][1]],
                              [seg2[1][0] - seg2[0][0], seg2[1][1] - seg2[0][1]]) / (
                      np.linalg.norm([seg2[1][0] - seg2[0][0], seg2[1][1] - seg2[0][1]]) *
                      np.linalg.norm([seg1[1][0] - seg1[0][0], seg1[1][1] - seg1[0][1]]))
    cos_theta = np.dot([seg1[0][0] - seg1[1][0], seg1[0][1] - seg1[1][1]],
                              [seg2[1][0] - seg2[0][0], seg2[1][1] - seg2[0][1]]) / (
                      np.linalg.norm([seg2[1][0] - seg2[0][0], seg2[1][1] - seg2[0][1]]) *
                      np.linalg.norm([seg1[1][0] - seg1[0][0], seg1[1][1] - seg1[0][1]]))

    cos_theta = np.clip(cos_theta, -1, 1)
    if sin_theta >= 0:
        return math.acos(cos_theta) / (2 * math.pi)
    else:
        return -math.acos(cos_theta) / (2 * math.pi)

def seg_length(segment):
    return math.sqrt( (segment[1][0] - segment[0][0]) ** 2 + (segment[1][1] - segment[0][1]) ** 2 )

def feature_compare(feature):
    angle_left = feature[0]
    length = feature[1]
    angle_right = feature[2]

    return 100*length + 1+angle_left + 0.1*angle_right

def _distance(p0, p1):
    d = np.sqrt(np.square(p0[0] - p1[0]) + np.square(p0[1] - p1[1]))
    return d

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.max_len = 0
        self.rollout_len = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.rollout_len[:]
        self.max_len = 0

    def add_transition(self, state, action, logprobs):
        # add to memory
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprobs)

        self.max_len = max(state[0].size()[0], self.max_len)


class Replay_Buffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def append_new_tuples(self, state, action, reward, next_state, done):
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, dones = list(zip(*experiences))
        actions, rewards, dones = np.array(actions), np.array(rewards), np.array(dones)

        # current state
        current_contour, current_segments_len, current_mask, current_triangles_info = list(zip(*states))
        current_max_len = max(current_segments_len)
        current_contour = np.array([np.concatenate((item, \
             np.zeros((current_max_len - item.shape[0], item.shape[1])))) for item in current_contour])
        current_mask = np.array([np.concatenate((item, \
             np.zeros((current_max_len - item.shape[0])))) for item in current_mask])
        current_segments_len = np.array(current_segments_len)
        current_triangles_info = np.array(current_triangles_info)
        states = (current_contour, current_segments_len, current_mask, current_triangles_info)

        # next state
        next_contour, next_segments_len, next_mask, next_triangles_info = list(zip(*next_states))
        next_max_len = max(next_segments_len)
        next_contour = np.array([np.concatenate((item, \
             np.zeros((next_max_len - item.shape[0], item.shape[1])))) for item in next_contour])
        next_mask = np.array([np.concatenate((item, \
             np.zeros((next_max_len - item.shape[0])))) for item in next_mask])

        # handle of termination
        next_mask[:, 0] = 1
        next_segments_len = np.clip(np.array(next_segments_len), a_min = 1, a_max = None)

        next_triangles_info = np.array(next_triangles_info)
        next_states = (next_contour, next_segments_len, next_mask, next_triangles_info)

        return (states, actions, rewards, next_states, dones)

def contain(super_poly, poly):
    return abs(super_poly.intersection(poly).area - poly.area) < 1e-6

def contain_or_intersect(super_poly, poly):
    return super_poly.intersection(poly).area > 1e-6

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def process_training_time(base_folder):
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    for file_name in subfolders:
        file_type = "*.txt"
        names = [os.path.basename(x) for x in glob.glob(os.path.join(file_name, file_type))]
        filtered_time = [ float(name.split("_")[2][:-4]) for name in names if name.split("_")[0][:4] == "time"]
        print(f"average time for {file_name} is {np.mean(filtered_time)}")

def interp(ratio, vec1, vec2):
    return (vec2 - vec1) * ratio + vec1

def append_text_to_file(save_path, input_string):
    f = open(save_path, 'a+')
    f.write(input_string)

if __name__ == "__main__":
    process_training_time("/home/edwardhui/data/evaluation/running/2020-01-02_13-09-07_evaluation_30-60-90/")