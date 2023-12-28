import mediapipe as mp 
import os
import cv2
import json
import numpy as np 
import warnings
import gc
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseOptions = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./mediapipe_models/pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence = 0.5, min_pose_presence_confidence = 0.5, 
    min_tracking_confidence = 0.5
)
HandOptions = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./mediapipe_models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands = 2,
    min_hand_detection_confidence = 0.5, min_hand_presence_confidence = 0.5, 
    min_tracking_confidence = 0.5
)

def get_label(path):
    return ''.join(path.split('/')[3].split(' ')[1:]).lower().replace(')', '').replace('(', '')

def read_paths(dir_path):
    files = []
    with open(dir_path, 'r') as f:
        files = [line.rstrip('\n') for line in f]
    return files

def get_hand_coordinates(hand_landmarker_result):
    hand1_x, hand1_y, hand2_x, hand2_y = [], [], [], []
    if len(hand_landmarker_result.hand_landmarks) > 0:
        for landmarker in hand_landmarker_result.hand_landmarks[0]:
            hand1_x.append(landmarker.x)
            hand1_y.append(landmarker.y)
    if len(hand_landmarker_result.hand_landmarks) > 1:
        for landmarker in hand_landmarker_result.hand_landmarks[1]:
            hand2_x.append(landmarker.x)
            hand2_y.append(landmarker.y)
    
    if len(hand_landmarker_result.handedness) > 0 and hand_landmarker_result.handedness[0][0].category_name == 'Right':
        hand1_x, hand1_y, hand2_x, hand2_y = hand2_x, hand2_y, hand1_x, hand1_y
    return hand1_x, hand1_y, hand2_x, hand2_y

def get_pose_coordinates(pose_landmarker_result):
    pose_x, pose_y = [], []
    if len(pose_landmarker_result.pose_landmarks[0]):
        for landmarker in pose_landmarker_result.pose_landmarks[0][:25]:
            pose_x.append(landmarker.x)
            pose_y.append(landmarker.y)
    return pose_x, pose_y

def get_filename(video_path):
    # label = video_path.split('/')[3].split(' ')[1].lower()
    # temp_label = ""
    # pos1 = video_path.find('(')
    # if pos1 != -1:
    #     pos2 = video_path.find(')')
    #     temp_label = video_path[pos1+1:pos2]
    label = ''.join(video_path.split('/')[3].split(' ')[1:]).lower().replace(')', '').replace('(', '')
    id = video_path.split('/')[-1].split('.')[0].lower()
    return label, id
        
def process_video(file_path, mode):
    label, id = get_filename(file_path)
    save_path = os.path.join(f'./keypts/{mode}_include_keypts', label+"_"+id+".json")
    cap = cv2.VideoCapture(file_path)
    n_frames = 0
    poses_x, poses_y = [], []
    hands1_x, hands1_y, hands2_x, hands2_y = [], [], [], []
    with PoseLandmarker.create_from_options(PoseOptions) as pose_model, HandLandmarker.create_from_options(HandOptions) as hand_model:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_timestamp_ms = int(1000*n_frames/fps)
            pose_landmarker_result = pose_model.detect_for_video(mp_image, frame_timestamp_ms)
            hand_landmarker_result = hand_model.detect_for_video(mp_image, frame_timestamp_ms)
            pose_x, pose_y = get_pose_coordinates(pose_landmarker_result)
            hand1_x, hand1_y, hand2_x, hand2_y = get_hand_coordinates(hand_landmarker_result)
            if not len(pose_x):
                pose_x = [np.nan]*25
                pose_y = [np.nan]*25
            if not len(hand1_x):
                hand1_x = [np.nan]*21
                hand1_y = [np.nan]*21
            if not len(hand2_x):
                hand2_x = [np.nan]*21
                hand2_y = [np.nan]*21
            hands1_x.append(hand1_x)
            hands1_y.append(hand1_y)
            hands2_x.append(hand2_x)
            hands2_y.append(hand2_y)
            
            poses_x.append(pose_x)
            poses_y.append(pose_y)
            n_frames+=1
            
    with open(save_path, 'w') as json_file:
            json.dump({
                'label':label,
                'pose_x' : poses_x,
                'pose_y' : poses_y,
                'hand1_x' : hands1_x,
                'hand1_y' : hands1_y,
                'hand2_x' : hands2_x,
                'hand2_y' : hands2_y,
                'n_frames' : n_frames,
                'id': label+'_'+id,
            }, json_file)
    del poses_x, poses_y, hands1_x, hands1_y, hands2_x, hands2_y
    gc.collect()

train_paths = read_paths('./train_test_paths/final_include_train.txt')
val_paths = read_paths('./train_test_paths/final_include_val.txt')
test_paths = read_paths('./train_test_paths/final_include_test.txt')
if not os.path.exists('./keypts/train_include_keypts'):
    os.mkdir('./keypts/train_include_keypts')
if not os.path.exists('./keypts/val_include_keypts'):
    os.mkdir('./keypts/val_include_keypts')
if not os.path.exists('./keypts/test_include_keypts'):
    os.mkdir('./keypts/test_include_keypts')


Parallel(n_jobs=4, batch_size=2)(
        delayed(process_video)(path, 'train')
        for path in tqdm(train_paths, desc=f"processing Train videos")
    )

Parallel(n_jobs=4, batch_size=2)(
        delayed(process_video)(path, 'val')
        for path in tqdm(val_paths, desc=f"processing Val videos")
    )

Parallel(n_jobs=4, batch_size=2)(
        delayed(process_video)(path, 'test')
        for path in tqdm(test_paths, desc=f"processing Test videos")
    )