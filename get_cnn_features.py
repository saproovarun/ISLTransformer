import os
import json
import pandas as pd
import sys
from opencv_transforms.transforms import *

import numpy as np
import torch
from tqdm.auto import tqdm

import cv2
import glob
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dim = 1536

FRAME_LENGTH = 256
FRAME_WIDTH = 256
COUNTER = 0

model = models.efficientnet_b3(weights = "DEFAULT")
model.classifier = torch.nn.Identity(1536)
model.eval()
model = model.to(device)
transform = Compose([
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def combine_xy(x, y):
    x, y = np.array(x), np.array(y)
    _, length = x.shape
    x = x.reshape((-1, length, 1))
    y = y.reshape((-1, length, 1))
    return np.concatenate((x, y), -1).astype(np.float64)

def process_video_frames(video_record):
    video_record['pose'] = combine_xy(video_record.pose_x, video_record.pose_y)
    video_record['hand1'] = combine_xy(video_record.hand1_x, video_record.hand1_y)
    video_record['hand2'] = combine_xy(video_record.hand2_x, video_record.hand2_y)
    
    video_record.pose = np.nan_to_num(video_record.pose)
    video_record.hand1 = np.nan_to_num(video_record.hand1)
    video_record.hand2 = np.nan_to_num(video_record.hand2)
    
    video_record.pose[:,:,0] = video_record.pose[:,:,0]*FRAME_WIDTH
    video_record.pose[:,:,1] = video_record.pose[:,:,1]*FRAME_LENGTH
    video_record.hand1[:,:,0] = video_record.hand1[:,:,0]*FRAME_WIDTH
    video_record.hand1[:,:,1] = video_record.hand1[:,:,1]*FRAME_LENGTH
    video_record.hand2[:,:,0] = video_record.hand2[:,:,0]*FRAME_WIDTH
    video_record.hand2[:,:,1] = video_record.hand2[:,:,1]*FRAME_LENGTH
    
    return video_record

def cnn_feat(file_path, save_dir):
    video_record = pd.read_json(file_path, typ="series")

    save_path = os.path.join(save_dir, video_record["id"] + ".npy")
    
    if video_record['n_frames'] == 0:
        print(f"0 Frames : {video_record['id']}")
    # if os.path.exists(save_path) or not video_record['n_frames']:
    #     return

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (11, 12),
        (13, 14),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 5),
        (5, 9),
        (9, 13),
        (13, 17),
        (0, 17),
    ]

    links = [
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 21),
        (15, 17),
        (17, 19),
        (19, 15),
        (22, 16),
        (16, 18),
        (18, 20),
        (16, 20),
    ]
    
    features = np.empty((0, output_dim))
    video_record = process_video_frames(video_record)
    
    for frame_index in range(video_record["n_frames"]):
        image = np.zeros((FRAME_LENGTH, FRAME_WIDTH, 3)).astype(np.uint8)
        for link in links:
            cv2.line(image, tuple(video_record.pose[frame_index][link[0]].astype("int")), \
                tuple(video_record.pose[frame_index][link[1]].astype("int")), (0,255,0),1)
        if video_record.hand1[frame_index, 0, 0] != 0:
            for connection in connections:
                cv2.line(image, tuple(video_record.hand1[frame_index][connection[0]].astype("int")), \
                    tuple(video_record.hand1[frame_index][connection[1]].astype("int")), (0,255,0),1)
        if video_record.hand2[frame_index, 0, 0] != 0:
            for connection in connections:
                cv2.line(image, tuple(video_record.hand2[frame_index][connection[0]].astype("int")), \
                        tuple(video_record.hand2[frame_index][connection[1]].astype("int")), (0,255,0),1)
        with torch.no_grad():
            feat = model(transform(image).unsqueeze(0).to(device))
        features = np.vstack([features, feat.cpu().numpy()])

    
    np.save(save_path, features)

if not os.path.exists('./cnn_features/train_include_features'):
    os.makedirs('./cnn_features/train_include_features')
if not os.path.exists('./cnn_features/val_include_features'):
    os.makedirs('./cnn_features/val_include_features')
if not os.path.exists('./cnn_features/test_include_features'):
    os.makedirs('./cnn_features/test_include_features')

train_paths = glob.glob('./keypts/train_include_keypts/*')
val_paths = glob.glob('./keypts/val_include_keypts/*')
test_paths = glob.glob('./keypts/test_include_keypts/*')

for path in tqdm(val_paths, desc="Validation Features"):
    cnn_feat(path, './cnn_features/val_include_features')
for path in tqdm(train_paths, desc="Train Features"):
    cnn_feat(path, './cnn_features/train_include_features')
for path in tqdm(test_paths, desc="Test Features"):
    cnn_feat(path, './cnn_features/test_include_features')

