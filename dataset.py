from torch.utils.data import Dataset
import glob
import torch 
import numpy as np
import json
import os


with open('./label_maps/label_map.json', 'r') as f:
    label_maps = json.load(f)
MAX_POS_EMBEDDINGS = 150
'''./cnn_features/val_include_features/young_mvi_9346.npy'''
class FeaturesDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = glob.glob(os.path.join(root_dir, "*"))
        self.y = [p.split('/')[-1].split('_')[0] for p in self.paths]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        x = np.load(self.paths[index])[:MAX_POS_EMBEDDINGS]
        n_seq = x.shape[0]
        x = np.pad(x, pad_width=((0, MAX_POS_EMBEDDINGS - n_seq), (0,0)))
        label = self.y[index]
        return {
            "inp": torch.FloatTensor(x), 
            "label": label_maps[label],
            "label_string": label,
            "n_frames": n_seq
            }

train_dataset = FeaturesDataset('./cnn_features/train_include_features')
val_dataset = FeaturesDataset('./cnn_features/val_include_features')
test_dataset = FeaturesDataset('./cnn_features/test_include_features')