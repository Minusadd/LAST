import copy
import os
import json
import heapq
from collections import defaultdict
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from pathlib import Path
from PIL import Image
from enc_visual import FeatureExtractor
#from feat_extractor import FeatureExtractor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#import constants
import numpy as np
import pickle

def transform_vector(vector):
    return [[1, 0] if x == 1 else [0, 1] for x in vector]

def split_string(s):
    first_part, trial_part, last_digit = re.split(r'(_trial_.*?)(_\d)$', s)[:-1]
    return first_part, trial_part[1:]
# Define a dictionary for storing skill frequencies
skill_freq = defaultdict(int)

ACTIONS = {'LookDown', 'RotateLeft', 'MoveAhead', 'LookUp', 'RotateRight', 'PickupObject', 'ToggleObjectOn', 'ToggleObjectOff', 'PutObject', 'OpenObject', 'CloseObject', 'SliceObject'}
#OBJECTS = set(constants.OBJECTS_WSLICED)

# Mapping from action types to indices
action_type_dict = {'LookDown': 0, 'RotateLeft': 1, 'MoveAhead': 2, 'LookUp': 3, 'RotateRight': 4, 'PickupObject': 5,
                    'ToggleObjectOn': 6, 'ToggleObjectOff': 7, 'PutObject': 8, 'OpenObject': 9, 'CloseObject': 10,
                    'SliceObject': 11}


object_type_dict = {'None': 0, 'Toilet': 1, 'AppleSliced': 2, 'DiningTable': 3, 'Spatula': 4, 'TissueBox': 5, 'Safe': 6, 'SoapBar': 7, 'Newspaper': 8, 'ToiletPaperHanger': 9, 'Ottoman': 10, 'GarbageCan': 11, 'KeyChain': 12, 'TennisRacket': 13, 'Bowl': 14, 'PepperShaker': 15, 'Desk': 16, 'ButterKnife': 17, 'Knife': 18, 'TVStand': 19, 'Bathtub': 20, 'RemoteControl': 21, 'Potato': 22, 'Kettle': 23, 'Tomato': 24, 'Fork': 25, 'BathtubBasin': 26, 'CoffeeMachine': 27, 'CoffeeTable': 28, 'Dresser': 29, 'Sink': 30, 'Vase': 31, 'Plunger': 32, 'Candle': 33, 'Cart': 34, 'Plate': 35, 'Cloth': 36, 'BaseballBat': 37, 'Ladle': 38, 'Watch': 39, 'Shelf': 40, 'Bed': 41, 'Pot': 42, 'TomatoSliced': 43, 'SprayBottle': 44, 'Pillow': 45, 'FloorLamp': 46, 'Apple': 47, 'BasketBall': 48, 'CD': 49, 'CounterTop': 50, 'WineBottle': 51, 'Microwave': 52, 'Faucet': 53, 'SinkBasin': 54, 'Glassbottle': 55, 'SideTable': 56, 'Cup': 57, 'DeskLamp': 58, 'Spoon': 59, 'Fridge': 60, 'SoapBottle': 61, 'DishSponge': 62, 'Drawer': 63, 'ArmChair': 64, 'Pencil': 65, 'Cabinet': 66, 'Pan': 67, 'BreadSliced': 68, 'Pen': 69, 'Book': 70, 'LettuceSliced': 71, 'StoveBurner': 72, 'CellPhone': 73, 'HandTowel': 74, 'SaltShaker': 75, 'Mug': 76, 'WateringCan': 77, 'Egg': 78, 'CreditCard': 79, 'Sofa': 80, 'AlarmClock': 81, 'PotatoSliced': 82, 'Bread': 83, 'Statue': 84, 'Laptop': 85, 'ToiletPaper': 86, 'Box': 87, 'Lettuce': 88}

# Define the directory containing the JSON files
json_dir = './data_gpt4/train'

# Define the directory for the output JSON files
output_dir = './data_gpt4_processed/train'  # Change this to your output directory
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Transformation rules
transformation_rules = {
    "MoveAhead": 25,
    "LookDown": 15,
    "LookUp": 15,
    "RotateRight": 90,
    "RotateLeft": 90
}

# Get a list of all JSON files in the directory
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for json_file in json_files:
    # Construct the full path to the JSON file
    full_path = os.path.join(json_dir, json_file)

    # Check if it's a file
    if os.path.isfile(full_path):
        # Open the file and load its JSON content
        with open(full_path, 'r') as f:
            data = json.load(f)

        # Iterate over the skills in the current file
        for skill in data['summary_actions']:
            # Increment the count for this skill
            skill_freq[skill] += 1

        # Iterate over the actions
        for i, action in enumerate(data['action_sequence']):
            # Check if this action has a numeric value
            action = action.replace("\'", "")
            if '_' in action:
                # Get the action type and value
                action_type = action.split('_')[0]
                action_value = int(action.split('_')[1])

                # Check if this action type has a transformation rule
                if action_type in transformation_rules:
                    # Calculate the original number of actions
                    num_original_actions = action_value // transformation_rules[action_type]

                    # Replace this action with the original actions
                    data['action_sequence'][i:i+1] = [f"{action_type}_{transformation_rules[action_type]}" for _ in range(num_original_actions)]

        # Iterate over the action alignment sequences
        for action_alignment in data['action_alignment']:
            robot_actions_str = action_alignment['robot_actions_str'].split(', ')
            for i, action in enumerate(robot_actions_str):
                # Check if this action has a numeric value
                action = action.replace("\'", "")
                if '_' in action:
                    # Get the action type and value
                    action_type = action.split('_')[0].replace(",", "")
                    action_value = int(action.split('_')[1].replace(",", ""))

                    # Check if this action type has a transformation rule
                    if action_type in transformation_rules:
                        # Calculate the original number of actions
                        num_original_actions = action_value // transformation_rules[action_type]

                        # Replace this action with the original actions
                        robot_actions_str[i:i+1] = [f"{action_type}_{transformation_rules[action_type]}" for _ in range(num_original_actions)]

            # Rejoin the transformed actions into a string
            action_alignment['robot_actions_str'] = ', '.join(robot_actions_str)

        # Save the modified JSON data to a new file
        with open(os.path.join(output_dir, json_file), 'w') as f:
            json.dump(data, f, indent=4)
            print("save_success")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5EncoderModel.from_pretrained('t5-base').to(device)
model.eval()
output_feat_dir = "./lang_feats"
output_switch_dir = "./switching_points_feats"
output_goal_dir = "./goal_feats"
os.makedirs(output_feat_dir, exist_ok=True)
os.makedirs(output_switch_dir, exist_ok=True)
os.makedirs(output_goal_dir, exist_ok=True)
# Define the special token
special_token = tokenizer.eos_token  # or define your own special token

# Define the directory containing the JSON files
json_dir = './data_gpt4_processed/train'  # Change this to your directory

# Initialize an empty dictionary to hold the embeddings
start_idx = 20742
task_skill_embeddings_buffer = []
skill_embeddings_buffer = []
task_embeddings_buffer = []
switching_points_buffer = []
iter = copy.deepcopy(start_idx)
#Iterate over all JSON files in the directory
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        # Load the JSON data
        with open(os.path.join(json_dir, json_file), 'r') as f:
            json_data = json.load(f)

        # Extract the task description
        task_desc = json_data['task_desc']
        task_desc = task_desc.replace('Output', '').replace(':', '').strip()

        task_skill_embeddings = []
        skill_embeddings = []
        task_switching_points = []
        # Iterate over all the skills in summary actions
        for i, skill in enumerate(json_data['summary_actions']):

            action_sequence = json_data['action_alignment'][i]['robot_actions_str'].split(', ')
            switching_points = [0] * len(action_sequence)
            switching_points[0] = 1  # Mark the start of the new skill
            task_switching_points.extend(switching_points)

        task_switching_points = torch.tensor(transform_vector(task_switching_points))

        skills = json_data['summary_actions']
        #
        # # Concatenate task description and each skill
        sentences = [task_desc + special_token + skill for skill in skills]
        #
        skill_goal_token = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        skill_goal_token = {name: tensor.to(device) for name, tensor in skill_goal_token.items()}
        # Get the output from the T5 model
        output_skill_goal = model(**skill_goal_token)

        # Get the embeddings
        embeddings_skill_goal = output_skill_goal.last_hidden_state

        # Store the embeddings in the dictionary
        feat = embeddings_skill_goal.mean(dim=1).detach().cpu()

        switch_points = task_switching_points.view(-1, 2)

        # Initialize empty tensor for new embeddings
        new_embeddings = torch.empty(switch_points.shape[0], feat.shape[1])

        # Initialize running index
        running_idx = 0

        for i in range(switch_points.shape[0]):
            if torch.all(switch_points[i] == torch.tensor([1, 0])):
                running_idx += 1
            new_embeddings[i] = feat[running_idx - 1]

        torch.save(new_embeddings, output_feat_dir + '/' + f'goal_skill_tensor_{iter}.pt')





        inputs_goal = tokenizer(task_desc, return_tensors='pt', padding=True, truncation=True)
        inputs_goal = {name: tensor.to(device) for name, tensor in inputs_goal.items()}
        # Get the output from the T5 model
        output_goal = model(**inputs_goal)

        # Get the embeddings
        embeddings_goal = output_goal.last_hidden_state

        # Store the embeddings in the dictionary
        feat = embeddings_goal.mean(dim=1)
        print(iter)
        # #
        #
        torch.save(feat.detach().cpu().repeat(task_switching_points.shape[0],1), output_goal_dir + '/' + f'goal_tensor_{iter}.pt')
        torch.save(torch.tensor(task_switching_points), output_switch_dir + '/' + f'switch_point_{iter}.pt')
        iter+=1



img_dir = "../../../data/ET/data/generated_2.1.0/train/"

def read_traj_images(json_path, image_folder):
    root_path = json_path.parents[0]
    with open(json_path) as json_file:
        json_dict = json.load(json_file)
    image_names = [None] * len(json_dict['plan']['low_actions'])
    for im_idx, im_dict in enumerate(json_dict['images']):

        if image_names[im_dict['low_idx']] is None:
            image_names[im_dict['low_idx']] = im_dict['image_name']
    before_last_image = json_dict['images'][-1]['image_name']
    last_image = '{:09d}.png'.format(int(before_last_image.split('.')[0]) + 1)
    image_names.append(last_image)
    fimages = [image_folder / im for im in image_names]

    if not any([os.path.exists(path) for path in fimages]):
        # maybe images were compressed to .jpg instead of .png
        fimages = [Path(str(path).replace('.png', '.jpg')) for path in fimages]
    if not all([os.path.exists(path) for path in fimages]):
        return None
    assert len(fimages) > 0
    # this reads on images (works with render_trajs.py)
    # fimages = sorted(glob.glob(os.path.join(root_path, image_folder, '*.png')))
    try:
        images = read_images(fimages)
    except:
        return None
    return images

def read_images(image_path_list):
    images = []
    for image_path in image_path_list:
        image_orig = Image.open(image_path)
        images.append(image_orig.copy())
        image_orig.close()
    return images

def extract_features(images, extractor):
    if images is None:
        return None
    feat = extractor.featurize(images, batch=8)
    return feat.cpu()

output_img_dir = "./images_feats"
os.makedirs(output_img_dir, exist_ok=True)
extractor = FeatureExtractor(
        'fasterrcnn', 'cuda', "./pretrained/fasterrcnn_model.pth",
        share_memory=True, compress_type='4x', load_heads=False)
obj_predictor = FeatureExtractor(
        archi='maskrcnn', device='cuda',
        checkpoint="./pretrained/maskrcnn_model.pth", load_heads=True)
for i, json_file in enumerate(os.listdir(json_dir)):
    print(i)
    if json_file.endswith('.json'):
        if os.path.exists(output_img_dir + '/' + f'image_tensor_{i+start_idx}.pt'):
            continue
        else:
            task_name, trial_name = split_string(json_file.split('.')[0])
            temp_p = img_dir + task_name + "/" + trial_name
            json_orig = Path(temp_p + "/" + "traj_data.json")
            temp_img = Path(temp_p + "/" + "raw_images")
            images = read_traj_images(json_orig, temp_img)
            feat = extract_features(images, extractor)
            torch.save(feat, output_img_dir + '/' + f'image_tensor_{i+start_idx}.pt')
    print("image done")


output_mask_dir = "./masks_feats"
os.makedirs(output_mask_dir, exist_ok=True)


for i, json_file in enumerate(os.listdir(json_dir)):
    print(i)
    if json_file.endswith('.json'):
        if os.path.exists(output_mask_dir + '/' + f'masked_actions_{i+start_idx}.pt'):
            continue
        else:
            task_name, trial_name = split_string(json_file.split('.')[0])
            temp_p = img_dir + task_name + "/" + trial_name
            json_orig = Path(temp_p + "/" + "traj_data.json")
            temp_img = Path(temp_p + "/" + "raw_images")
            images = read_traj_images(json_orig, temp_img)
            if images is None:
                print('None')
                torch.save(None, output_mask_dir + '/' + f'masked_actions_{i+start_idx}.pt')
                continue
            obj_list = []

            with open(os.path.join(json_dir, json_file)) as f:
                data = json.load(f)

            action_sequence = data['action_sequence']

            action_sequence_one_hot = []

            for idx in range(len(action_sequence)):

                rcnn_pred = obj_predictor.predict_objects(images[idx])
                labels = list(set([pred.label for pred in rcnn_pred]))
                object_type_one_hot = torch.zeros(len(object_type_dict))
                for obj_type in labels:
                    object_type_idx = object_type_dict[obj_type]
                    object_type_one_hot[object_type_idx] = 1
                action = action_sequence[idx]
                action_split = action.split(' ') if ' ' in action else action.split('_')
                action_type = action_split[0]
                action_type_idx = action_type_dict[action_type]
                action_type_one_hot = np.zeros(len(action_type_dict))
                action_type_one_hot[action_type_idx] = 1



                # If action has an object type, update the one-hot vector
                if action_type in {'PickupObject', 'ToggleObjectOn', 'ToggleObjectOff', 'PutObject', 'OpenObject',
                                   'CloseObject', 'SliceObject'}:
                    print("***************")
                    object_type = action_split[1]
                    object_type_idx = object_type_dict[object_type]
                    object_type_one_hot[object_type_idx] = 1

                obj_list.append(object_type_one_hot)


            torch.save(torch.stack(obj_list), output_mask_dir + '/' + f'masked_actions_{i+start_idx}.pt')
    print("mask done")


#
# Mapping from object types to indices


action_sequence_list = []  # list to save all the action sequences
iter = 0
# Loop over json files
for json_file in os.listdir(json_dir):

    with open(os.path.join(json_dir, json_file)) as f:
        data = json.load(f)

    action_sequence = data['action_sequence']

    action_sequence_one_hot = []

    for action in action_sequence:
        action_split = action.split(' ') if ' ' in action else action.split('_')
        action_type = action_split[0]
        action_type_idx = action_type_dict[action_type]
        action_type_one_hot = np.zeros(len(action_type_dict))
        action_type_one_hot[action_type_idx] = 1

        object_type_one_hot = np.zeros(len(object_type_dict))  # start with all zeros

        # If action has an object type, update the one-hot vector
        if action_type in {'PickupObject', 'ToggleObjectOn', 'ToggleObjectOff', 'PutObject', 'OpenObject',
                           'CloseObject', 'SliceObject'}:
            object_type = action_split[1]
            object_type_idx = object_type_dict[object_type]
            object_type_one_hot[object_type_idx] = 1
        else:
            object_type_one_hot[0] = 1
        # Concatenate action_type_one_hot and object_type_one_hot
        action_one_hot = np.concatenate([action_type_one_hot, object_type_one_hot])

        action_sequence_one_hot.append(action_one_hot)

    action_sequence_list.append(action_sequence_one_hot)
    print(iter)
    iter+=1
with open('action_sequences.pkl', 'wb') as f:
    pickle.dump(action_sequence_list, f)



