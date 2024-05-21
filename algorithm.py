import copy
import pickle
import torch
import numpy as np
from networks import VariationalNetwork, LowLevelPolicyNetwork, LowLevelMLPPolicyNetwork, DiscretePolicyNetwork, \
    Obsencoder, Actionencoder, Lanencoder_skill, Lanencoder_goal
import TFLogger
from headers import *
from data_management.simple_replay_buffer import SimpleReplayBuffer
from net_utils import FlattenMlp
import pytorch_util as ptu
import time
import json
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import wandb

action_type_dict = {'LookDown': 0, 'RotateLeft': 1, 'MoveAhead': 2, 'LookUp': 3, 'RotateRight': 4, 'PickupObject': 5,
                    'ToggleObjectOn': 6, 'ToggleObjectOff': 7, 'PutObject': 8, 'OpenObject': 9, 'CloseObject': 10,
                    'SliceObject': 11}
id_to_action = {v: k for k, v in action_type_dict.items()}
object_type_dict = {'None': 0, 'Toilet': 1, 'AppleSliced': 2, 'DiningTable': 3, 'Spatula': 4, 'TissueBox': 5, 'Safe': 6,
                    'SoapBar': 7, 'Newspaper': 8, 'ToiletPaperHanger': 9, 'Ottoman': 10, 'GarbageCan': 11,
                    'KeyChain': 12, 'TennisRacket': 13, 'Bowl': 14, 'PepperShaker': 15, 'Desk': 16, 'ButterKnife': 17,
                    'Knife': 18, 'TVStand': 19, 'Bathtub': 20, 'RemoteControl': 21, 'Potato': 22, 'Kettle': 23,
                    'Tomato': 24, 'Fork': 25, 'BathtubBasin': 26, 'CoffeeMachine': 27, 'CoffeeTable': 28, 'Dresser': 29,
                    'Sink': 30, 'Vase': 31, 'Plunger': 32, 'Candle': 33, 'Cart': 34, 'Plate': 35, 'Cloth': 36,
                    'BaseballBat': 37, 'Ladle': 38, 'Watch': 39, 'Shelf': 40, 'Bed': 41, 'Pot': 42, 'TomatoSliced': 43,
                    'SprayBottle': 44, 'Pillow': 45, 'FloorLamp': 46, 'Apple': 47, 'BasketBall': 48, 'CD': 49,
                    'CounterTop': 50, 'WineBottle': 51, 'Microwave': 52, 'Faucet': 53, 'SinkBasin': 54,
                    'Glassbottle': 55, 'SideTable': 56, 'Cup': 57, 'DeskLamp': 58, 'Spoon': 59, 'Fridge': 60,
                    'SoapBottle': 61, 'DishSponge': 62, 'Drawer': 63, 'ArmChair': 64, 'Pencil': 65, 'Cabinet': 66,
                    'Pan': 67, 'BreadSliced': 68, 'Pen': 69, 'Book': 70, 'LettuceSliced': 71, 'StoveBurner': 72,
                    'CellPhone': 73, 'HandTowel': 74, 'SaltShaker': 75, 'Mug': 76, 'WateringCan': 77, 'Egg': 78,
                    'CreditCard': 79, 'Sofa': 80, 'AlarmClock': 81, 'PotatoSliced': 82, 'Bread': 83, 'Statue': 84,
                    'Laptop': 85, 'ToiletPaper': 86, 'Box': 87, 'Lettuce': 88}
id_to_object = {v: k for k, v in object_type_dict.items()}


def custom_collate_fn(batch):
    image_data, action_data, goal_data, goal_skill_data, switching_point_data, masked_action_data = zip(*batch)
    image_data = pad_sequence(image_data, batch_first=True, padding_value=-1)

    action_data = pad_sequence(action_data, batch_first=True, padding_value=-1)
    goal_data = pad_sequence(goal_data, batch_first=True, padding_value=-1)
    goal_skill_data = pad_sequence(goal_skill_data, batch_first=True, padding_value=-1)
    switching_point_data = pad_sequence(switching_point_data, batch_first=True, padding_value=2)
    masked_action_data = pad_sequence(masked_action_data, batch_first=True, padding_value=-1)
    return image_data, action_data, goal_data, goal_skill_data, switching_point_data, masked_action_data


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Compute cross entropy
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # Get the index of the true class
        pt = torch.exp(-ce_loss)

        # Compute the focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Learning Skills from Demonstrations')

    # Setup training.

    parser.add_argument('--datadir', dest='datadir', type=str, default='./Data/')
    parser.add_argument('--a6000', dest='a6000', type=int, default=0)
    parser.add_argument('--more_skills', dest='more_skills', type=int, default=1)
    parser.add_argument('--include_goal', dest='include_goal', type=int, default=0)
    parser.add_argument('--gpuid', dest='gpuid', type=int, default=0)
    parser.add_argument('--pretrain', dest='pretrain', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--large_dataset', dest='large_dataset', type=int, default=0)
    parser.add_argument('--split', dest='split', type=float, default=0.99)
    parser.add_argument('--online_skill_dim', dest='online_skill_dim', type=int, default=100)
    parser.add_argument('--history_cond', dest='history_cond', type=int, default=0)
    parser.add_argument('--debug', dest='debug', type=int, default=0)
    parser.add_argument('--name', dest='name', type=str, default=None)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--training_phase_size', dest='training_phase_size', type=int, default=40000)  # 500000  #30000
    parser.add_argument('--initial_counter_value', dest='initial_counter_value', type=int, default=0)
    parser.add_argument('--data', dest='data', type=str, default='testcase')
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--logdir', dest='logdir', type=str, default='results/')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=80)  # Number of epochs to train for. Reduce for Mocap.
    parser.add_argument('--online_training', dest='online_training', type=int, default=0)
    parser.add_argument('--use_our_skill', dest='use_our_skill', type=int, default=1)
    parser.add_argument('--env_name', dest='env_name', type=str, default='pick_clean_then_place_in_recep-SoapBar-None-BathtubBasin-413')
    #env names: 'look_at_obj_in_light-KeyChain-None-DeskLamp-327' 'pick_clean_then_place_in_recep-SoapBar-None-BathtubBasin-413' pick_heat_then_place_in_recep-Potato-None-Fridge-11
    #pick_two_obj_and_place-AlarmClock-None-Dresser-305, pick_cool_then_place_in_recep-Tomato-None-Microwave-13, pick_and_place_simple-Book-None-SideTable-329
    # Training setting.
    parser.add_argument('--constrain_m', dest='constrain_m', type=int, default=1)
    parser.add_argument('--causal_mask', dest='causal_mask', type=int, default=1)
    parser.add_argument('--fix_subpolicyfirst', dest='fix_subpolicyfirst', type=int, default=0)
    parser.add_argument('--highlevel_only', dest='highlevel_only', type=int, default=0)
    parser.add_argument('--forward_weight', dest='forward_weight', type=float, default=0.5)
    parser.add_argument('--m_known', dest='m_known', type=int, default=0)
    parser.add_argument('--joint_primitive_net', dest='joint_primitive_net', type=int, default=1)
    parser.add_argument('--use_rnn_primitive', dest='use_rnn_primitive', type=int, default=1)
    parser.add_argument('--num_m_loss_weight', dest='num_m_loss_weight', type=float, default=1.)
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    # parser.add_argument('--transformer',dest='transformer',type=int,default=0)
    parser.add_argument('--z_dimensions', dest='z_dimensions', type=int, default=2)  # 64
    parser.add_argument('--number_layers', dest='number_layers', type=int, default=2)
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=32)
    parser.add_argument('--mask_var', dest='mask_var', type=int, default=0)

    parser.add_argument('--replay_buffer_size', dest='replay_buffer_size', type=int, default=100000)

    parser.add_argument('--number_policies', dest='number_policies', type=int, default=100)
    parser.add_argument('--fix_subpolicy', dest='fix_subpolicy', type=int, default=0)
    parser.add_argument('--train_only_policy', dest='train_only_policy', type=int,
                        default=0)  # Train only the policy network and use a pretrained encoder. This is weird but whatever.
    parser.add_argument('--load_latent', dest='load_latent', type=int,
                        default=1)  # Whether to load latent policy from model or not.
    parser.add_argument('--subpolicy_model', dest='subpolicy_model', type=str)
    parser.add_argument('--traj_length', dest='traj_length', type=int, default=-1)
    parser.add_argument('--skill_length', dest='skill_length', type=int, default=5)
    parser.add_argument('--var_skill_length', dest='var_skill_length', type=int, default=1)
    parser.add_argument('--display_freq', dest='display_freq', type=int, default=10000)
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=1)
    parser.add_argument('--eval_freq', dest='eval_freq', type=int, default=1)
    parser.add_argument('--updates_per_epoch', dest='updates_per_epoch', type=int, default=1500)
    parser.add_argument('--entropy', dest='entropy', type=int, default=0)
    parser.add_argument('--var_entropy', dest='var_entropy', type=int, default=0)
    parser.add_argument('--ent_weight', dest='ent_weight', type=float, default=0.)
    parser.add_argument('--var_ent_weight', dest='var_ent_weight', type=float, default=2.)
    parser.add_argument('--alpha', dest='alpha', type=float, default=1)
    parser.add_argument('--lat_z_wt', dest='lat_z_wt', type=float, default=0.1)
    parser.add_argument('--act_scale', dest='act_scale', type=float, default=27.)
    parser.add_argument('--lat_m_wt', dest='lat_m_wt', type=float, default=1.)
    parser.add_argument('--lat_k_wt', dest='lat_k_wt', type=float, default=1.)

    parser.add_argument('--latent_loss_weight', dest='latent_loss_weight', type=float, default=0.01)
    parser.add_argument('--kl_weight', dest='kl_weight', type=float, default=0.01)
    parser.add_argument('--var_loss_weight', dest='var_loss_weight', type=float, default=1.)
    parser.add_argument('--temperature', dest='temperature', type=float, default=1.)

    # Exploration and learning rate parameters.
    parser.add_argument('--epsilon_from', dest='epsilon_from', type=float, default=0.3)
    parser.add_argument('--epsilon_to', dest='epsilon_to', type=float, default=0.05)
    parser.add_argument('--epsilon_over', dest='epsilon_over', type=int, default=5)
    parser.add_argument('--temperature_from', dest='temperature_from', type=float, default=1.)
    parser.add_argument('--temperature_to', dest='temperature_to', type=float, default=1.)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=3e-4)

    # Baseline parameters.


    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, image_directory, goal_skill_feat_dir, switching_points_dir, goal_feat_dir, action_dir,
                 masked_action_dir, large_dataset=False, more_skills=False, image_directory2=None, goal_skill_feat_dir2=None,
                 switching_points_dir2=None, goal_feat_dir2=None, action_dir2=None, masked_action_dir2=None):
        self.image_directory = image_directory
        self.goal_skill_feat_dir = goal_skill_feat_dir
        self.switching_point_dir = switching_points_dir
        self.goal_feat_dir = goal_feat_dir
        self.masked_action_dir = masked_action_dir
        self.image_file_list = os.listdir(self.image_directory)
        self.goal_skill_feat_list = os.listdir(self.goal_skill_feat_dir)
        self.goal_list = os.listdir(goal_feat_dir)
        self.switching_point_list = os.listdir(switching_points_dir)
        self.masked_action_list = os.listdir(masked_action_dir)
        self.large_dataset = large_dataset

        assert len(self.image_file_list) == len(self.goal_skill_feat_list), \
            "The number of files should be the same."
        with open(action_dir, 'rb') as f:
            self.action_lists = pickle.load(f)
        self.more_skills = more_skills
        self.rm_idx = []
        for idxx in range(len(self.image_file_list)):

            idx = idxx + 20742

            image_file_name = f'image_tensor_{idx}.pt'
            goal_skill_file_name = f'goal_skill_tensor_{idx}.pt'
            if torch.load(os.path.join(self.image_directory, image_file_name)) is None:
                self.rm_idx.append(idxx)
                print("IDX:", idxx)
            else:
                image_data = torch.load(os.path.join(self.image_directory, image_file_name)).to("cpu")
                goal_skill_data = torch.load(os.path.join(self.goal_skill_feat_dir, goal_skill_file_name)).to("cpu")
                if goal_skill_data.shape[0] != (image_data.shape[0] - 1):
                    self.rm_idx.append(idxx)
                    print("IDX:", idxx)


    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):


        while idx in self.rm_idx:
            idx -= 1
        if idx < 20078: #20742
            idxx = idx + 20742
        else:
            idxx = idx
        image_file_name = f'image_tensor_{idxx}.pt'
        goal_skill_file_name = f'goal_skill_tensor_{idxx}.pt'
        goal_file_name = f'goal_tensor_{idxx}.pt'
        switching_point_file_name = f'switch_point_{idxx}.pt'
        masked_action_file_name = f'masked_actions_{idxx}.pt'

        image_data = torch.load(os.path.join(self.image_directory, image_file_name)).to("cpu")
        goal_skill_data = torch.load(os.path.join(self.goal_skill_feat_dir, goal_skill_file_name)).to("cpu")
        goal_data = torch.load(os.path.join(self.goal_feat_dir, goal_file_name)).to("cpu")
        switching_point_data = torch.load(os.path.join(self.switching_point_dir, switching_point_file_name)).to(
            "cpu")
        masked_action_data = torch.load(os.path.join(self.masked_action_dir, masked_action_file_name)).to("cpu")

        action_data = torch.FloatTensor(self.action_lists[idxx-20742]).to("cpu")

        return image_data, action_data, goal_data, goal_skill_data, switching_point_data, masked_action_data


class PolicyManager_Joint():


    def __init__(self, train_dataloader=None, test_dataloader=None, args=None, train_size=100, test_size=10,
                 device="cpu"):

        super(PolicyManager_Joint, self).__init__()

        self.args = args
        self.seed = args.seed
        self.device = device
        self.num_policies = args.number_policies
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.hidden_size = args.hidden_size
        self.train_size = train_size
        self.test_size = test_size

        self.state_size = 512
        self.action_dim = 12 + 89
        self.input_size = 1 + 1
        self.img_emb_size = 256
        self.lang_emb_size = 256
        self.pretrain = self.args.pretrain
        self.history_cond = self.args.history_cond

        self.output_size = 1
        self.number_layers = args.number_layers
        self.traj_length = args.traj_length
        self.conditional_info_size = 0
        self.updates_per_epoch = args.updates_per_epoch
        self.training_phase_size = args.training_phase_size
        self.number_epochs = args.epochs
        self.baseline_value = 0.
        self.beta_decay = 0.9
        self.soft_target_tau = 0.005
        self.act_scale = args.act_scale
        self.learning_rate = args.learning_rate
        self.forward_weight = args.forward_weight
        self.eval=False
        self.latent_m_loss_weight = args.lat_m_wt
        self.latent_z_loss_weight = args.lat_z_wt
        self.latent_k_loss_weight = args.lat_k_wt
        self.initial_epsilon = self.args.epsilon_from
        self.final_epsilon = self.args.epsilon_to
        self.initial_temperature = self.args.temperature_from
        self.final_temperature = self.args.temperature_to
        self.decay_epochs = self.args.epsilon_over
        self.decay_counter = self.decay_epochs * self.train_size
        self.replay_buffer = SimpleReplayBuffer(
            args.replay_buffer_size,
            [512,7,7],#FIXME obs shape
            89 + 12, #action dim
            100 #FIXME
        )
        self.alpha = self.args.alpha
        self.decay_rate = (self.initial_epsilon - self.final_epsilon) / (self.decay_counter)
        self.decay_rate_temperature = (self.args.temperature_from - self.args.temperature_to) / (self.decay_counter)

    def create_networks(self):
        self.obs_encoder = Obsencoder(output_size=self.img_emb_size).to(self.device)
        self.act_encoder = Actionencoder(output_size=64).to(self.device)
        self.goal_encoder = Lanencoder_goal(output_size=self.lang_emb_size).to(self.device)
        self.lang_encoder = Lanencoder_skill(output_size=self.lang_emb_size).to(self.device)
        self.variational_policy = VariationalNetwork(self.args, self.obs_encoder, self.lang_encoder, self.act_encoder,
                                                     img_emb_size=self.img_emb_size, g_dimensions=self.lang_emb_size,
                                                     action_size=89 + 12, hidden_size=256,
                                                     num_subpolicies=self.num_policies, num_objects=89).to(self.device)
        if self.args.use_rnn_primitive:
            self.primitive_policy = LowLevelPolicyNetwork(self.args, self.obs_encoder, self.goal_encoder,
                                                          img_emb_size=self.img_emb_size, action_size=12,
                                                          hidden_size=256, num_subpolicies=self.num_policies,
                                                          num_objects=89, g_dimensions=self.lang_emb_size,
                                                          causal_mask=self.args.causal_mask,
                                                          include_goal=self.args.include_goal).to(self.device)
        else:
            self.primitive_policy = LowLevelMLPPolicyNetwork(self.args, self.obs_encoder, self.goal_encoder,
                                                             img_emb_size=self.img_emb_size, action_size=12,
                                                             hidden_size=256, num_subpolicies=self.num_policies,
                                                             num_objects=89, g_dimensions=self.lang_emb_size,
                                                             causal_mask=self.args.causal_mask).to(self.device)
        if self.args.highlevel_only:
            for params in (list(self.obs_encoder.parameters()) + list(self.goal_encoder.parameters()) + list(
                    self.lang_encoder.parameters()) + list(self.act_encoder.parameters()) + list(
                    self.variational_policy.parameters()) + list(self.primitive_policy.parameters())):
                params.requires_grad = False
        self.discrete_policy = DiscretePolicyNetwork(self.args, self.obs_encoder, self.lang_encoder, self.act_encoder,
                                                     img_emb_size=self.img_emb_size, g_dimensions=self.lang_emb_size,
                                                     action_size=89 + 12, hidden_size=256,
                                                     num_subpolicies=self.num_policies, num_objects=89).to(self.device)
        self.qf1 = FlattenMlp(
            hidden_sizes=[256, 256],
            input_size=256, #+ self.args.online_skill_dim,
            output_size=self.args.online_skill_dim,
        ).to(self.device)  # qnetwork1
        self.qf2 = FlattenMlp(
            hidden_sizes=[256, 256],
            input_size=256, #+ self.args.online_skill_dim,
            output_size=self.args.online_skill_dim,
        ).to(self.device) # qnetwork2
        self.vf = FlattenMlp(
            hidden_sizes=[256, 256],
            input_size=256,
            output_size=1,
        ).to(self.device)
        self.target_vf = self.vf.copy().to(self.device)
        self.target_qf1 = self.qf1.copy().to(self.device)
        self.target_qf2 = self.qf2.copy().to(self.device)

    def create_training_ops(self, fix_subpolicy=False, highlevel_only=False):
        self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
        #DECREASE THE WEIGHT FOR MOVE_FORWARD (IT'S USED TOO FREQUENTLY!)
        class_weights = torch.tensor([1.0, 1.0, self.forward_weight, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(
            self.device)
        self.mse_loss = torch.nn.MSELoss()
        self.cross_entropy_loss1 = torch.nn.CrossEntropyLoss(class_weights)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=2)
        if fix_subpolicy:
            parameter_list = list(self.goal_encoder.parameters()) + list(self.act_encoder.parameters()) + list(
                self.variational_policy.parameters()) + list(
                self.discrete_policy.parameters())
        elif highlevel_only:
            parameter_list = list(
                self.discrete_policy.parameters())
            self.primitive_policy.eval()
            self.variational_policy.eval()
            self.obs_encoder.eval()
            self.act_encoder.eval()
            self.goal_encoder.eval()
            self.lang_encoder.eval()

        else:
            parameter_list = list(self.obs_encoder.parameters()) + list(self.goal_encoder.parameters()) + list(
                self.lang_encoder.parameters()) + list(self.act_encoder.parameters()) + list(
                self.variational_policy.parameters()) + list(self.primitive_policy.parameters()) + list(
                self.discrete_policy.parameters())

        self.optimizer = torch.optim.Adam(parameter_list, lr=self.learning_rate)
        self.qf1_optimizer = torch.optim.Adam(
            self.qf1.parameters(),
            lr= 3e-4,
        )
        self.qf2_optimizer = torch.optim.Adam(
            self.qf2.parameters(),
            lr= 3e-4,
        )
        self.vf_optimizer = torch.optim.Adam(
            self.vf.parameters(),
            lr= 3e-4,
        )

    def save_all_models(self, suffix):

        logdir = os.path.join(self.args.logdir, self.args.name)
        savedir = os.path.join(logdir, "saved_models")
        if not (os.path.isdir(savedir)):
            os.mkdir(savedir)
        save_object = {}
        save_object['Obs_Encoder'] = self.obs_encoder.state_dict()
        save_object['Action_Encoder'] = self.act_encoder.state_dict()
        save_object['Goal_Encoder'] = self.goal_encoder.state_dict()
        save_object['Skill_Encoder'] = self.lang_encoder.state_dict()
        save_object['Variational_Policy'] = self.variational_policy.state_dict()
        save_object['Discrete_Policy'] = self.discrete_policy.state_dict()
        save_object['Primitive_Policy'] = self.primitive_policy.state_dict()
        if self.args.online_training:
            save_object['Q_network1'] = self.qf1
            save_object['Q_network2'] = self.qf2
            save_object['V_network'] = self.vf
        torch.save(save_object, os.path.join(savedir, "Model_" + suffix))

    def load_all_models(self, path, just_subpolicy=False):
        load_object = torch.load(path, map_location=self.device)
        if not just_subpolicy:
            self.discrete_policy.load_state_dict(load_object['Discrete_Policy'])
            self.variational_policy.load_state_dict(load_object['Variational_Policy'])
            self.act_encoder.load_state_dict(load_object['Action_Encoder'])
            self.goal_encoder.load_state_dict(load_object['Goal_Encoder'])
        self.primitive_policy.load_state_dict(load_object['Primitive_Policy'])
        self.obs_encoder.load_state_dict(load_object['Obs_Encoder'])
        self.lang_encoder.load_state_dict(load_object['Skill_Encoder'])

    def set_epoch(self, counter):
        # FIXME
        if self.args.train:
            if counter < self.decay_counter:
                self.epsilon = self.initial_epsilon - self.decay_rate * counter
                self.temperature = self.initial_temperature - self.decay_rate_temperature * counter
            else:
                self.epsilon = self.final_epsilon
                self.temperature = self.final_temperature

            if counter < self.training_phase_size:
                self.training_phase = 1
            else:
                self.training_phase = 2

        else:
            self.epsilon = 0.
            self.training_phase = 1
            self.temperature = 1.


    def assemble_inputs(self, image_data, action_data, goal_data, latent_k, latent_m, masked_action_data,
                        conditional_information=None, eval=False):

        return image_data, action_data, goal_data, latent_k, latent_m, masked_action_data

    def concat_state_action(self, sample_traj, sample_action_seq):
        # Add blank to start of action sequence and then concatenate.
        sample_action_seq = np.concatenate([np.zeros((1, self.output_size)), sample_action_seq], axis=0)

        return np.concatenate([sample_traj, sample_action_seq], axis=-1)

    def old_concat_state_action(self, sample_traj, sample_action_seq):
        # Add blank to the END of action sequence and then concatenate.
        sample_action_seq = np.concatenate([sample_action_seq, np.zeros((1, self.output_size))], axis=0)
        return np.concatenate([sample_traj, sample_action_seq], axis=-1)


    def evaluate_loglikelihoods(self, image_data, action_data, goal_data, latent_k, latent_m, presampled_m,
                                masked_action_data, goal_skill_data=None, prem_latent_k=None):


        #GET THE PREDICTIONS FOR THE PRIMITIVE ACTIONS
        if self.pretrain:
            if self.history_cond:
                selected_a = self.primitive_policy.forward(image_data, goal_data, prem_latent_k,
                                                           mask_input=masked_action_data,
                                                           temperature=1.0)
            else:
                selected_a = self.primitive_policy.forward(image_data, goal_data, lang_input=goal_skill_data,
                                                           mask_input=masked_action_data,
                                                           temperature=1.0)
        else:
            selected_a = self.primitive_policy.forward(image_data, goal_data, latent_k, mask_input=masked_action_data,
                                                       temperature=1.0)


        action_data_input = torch.cat(
            (((torch.ones((action_data.shape[0], 1, self.action_dim))) * (-1)).to(self.device), action_data[:, :-1, :]),
            dim=1)
        latent_k_input = torch.concat(
            [((torch.ones((latent_k.shape[0], 1, self.num_policies))) * (-1)).to(self.device), latent_k[:, :-1, :]],
            dim=1)
        latent_m_input = torch.concat(
            [((torch.ones((latent_m.shape[0], 1, 2))) * (-1)).to(self.device), latent_m[:, :-1, :]], dim=1)

        #GET THE PREDICTIONS FOR THE SKILLS FROM THE CAUSAL NETWORKS
        latent_k1, latent_m1, latent_k_preprobabilities, latent_m_preprobabilities = self.discrete_policy.forward(
            image_data, action_data_input, goal_data, latent_k_input.detach(), latent_m_input.detach(), presampled_m,
            epsilon=0., temperature=self.temperature, deterministic=False)

        return selected_a, latent_k1, latent_m1, latent_k_preprobabilities, latent_m_preprobabilities


    def new_update_policies(self, i, variational_m, variational_k, latent_m, latent_k_indices,
                            variational_k_probabilities,
                            variational_m_probabilities, latent_k_probabilities,
                            latent_m_probabilities, sampled_actions, output_actions, presampled_m,
                            variational_k_codelen, variational_k_entropys, mask_var=False, mask_lat=False):

        # Set optimizer gradients to zero.
        self.optimizer.zero_grad()


        if mask_var or self.training_phase == 1:
            variational_k_logprobabilities = torch.log(variational_k_probabilities.clone().detach() + 1e-30)
            variational_m_logprobabilities = torch.log(variational_m_probabilities.clone().detach() + 1e-30)
        else:
            variational_k_logprobabilities = torch.log(variational_k_probabilities + 1e-30)
            variational_m_logprobabilities = torch.log(variational_m_probabilities + 1e-30)

        if mask_lat:
            latent_k_logprobabilities = torch.log(latent_k_probabilities.clone().detach() + 1e-30)
            latent_m_logprobabilities = torch.log(latent_m_probabilities.clone().detach() + 1e-30)
        else:
            latent_k_logprobabilities = torch.log(latent_k_probabilities + 1e-30)
            latent_m_logprobabilities = torch.log(latent_m_probabilities + 1e-30)

        indexes = (presampled_m == torch.tensor([1, 0]).to(self.device)).all(dim=-1)

        mask_pad = (presampled_m != 2).any(dim=-1)

        self.latent_m_loss = torch.nn.functional.kl_div(latent_m_logprobabilities[mask_pad], variational_m_logprobabilities[mask_pad], reduction='batchmean', log_target=True)


        #FIXME
        self.latent_k_loss = torch.nn.functional.kl_div(latent_k_logprobabilities[mask_pad], variational_k_logprobabilities[mask_pad], reduction='batchmean', log_target=True)
        # self.latent_k_loss = torch.nn.functional.kl_div(latent_k_logprobabilities,
        #                                                 variational_k_logprobabilities, reduction='batchmean',
        #                                                 log_target=True)


        #FIXME
        self.mdl_loss2 = (variational_k_entropys.squeeze(2)[mask_pad]).mean()
        #self.mdl_loss2 = variational_k_entropys.mean()
        self.total_latent_loss = self.args.kl_weight * (self.latent_k_loss + self.latent_m_loss)

        # Get subpolicy losses SEPARATELY FOR THE ACTION TYPES AND OBJECT TYPES
        target_actions = torch.argmax(sampled_actions[..., :12], dim=-1)
        target_objects = torch.argmax(sampled_actions[..., 12:], dim=-1)

        self.subpolicy_loss1 = self.cross_entropy_loss1(output_actions[mask_pad][..., :12].view(-1, 12),
                                                        target_actions[mask_pad].view(-1)).sum()

        target_masks = (target_actions > 4).view(-1)
        target_masks2 = (target_actions > 4)  # .view(-1)
        self.subpolicy_loss2 = self.cross_entropy_loss(output_actions[..., 12:].view(-1, 89)[target_masks],
                                                       target_objects.view(-1)[target_masks]).sum()

        if self.training_phase == 1:
            if self.args.highlevel_only:
                self.total_loss = self.total_latent_loss
            else:
                self.total_loss = self.subpolicy_loss1 + 0.1 * self.subpolicy_loss2 + self.total_latent_loss
        # IF DONE WITH PHASE ONE:
        elif self.training_phase == 2 or self.training_phase == 3:
            self.total_loss = self.subpolicy_loss1 + 0.1 * self.subpolicy_loss2 + self.total_latent_loss + self.args.ent_weight * self.mdl_loss2

        self.total_loss.sum().backward()

        self.optimizer.step()



    # compute next_state = state + action
    def take_rollout_step(self, state_input, goal_input, t, discrete_skill, use_env=False):

        actions = self.primitive_policy.get_actions(state_input, goal_input, discrete_skill, greedy=True)
        action_to_execute = actions[-1]

        return action_to_execute


    def rollout_variational_network(self, image_data, action_data, goal_data, goal_skill_data, switching_point_data,
                                    masked_action_data):

        variational_k, variational_m, variational_k_probabilities, variational_m_probabilities, prem_variational_k, variational_k_logprobabilities, variational_k_entropys = self.variational_policy.forward(
            image_data, action_data, goal_skill_data, switching_point_data, deterministic=True, epsilon=0.,
            temperature=self.temperature, constrain_m=self.args.constrain_m)
        if self.pretrain:
            if self.history_cond:
                actions, skill_input = self.primitive_policy.get_actions(image_data, goal_data, prem_variational_k,
                                                                         mask_input=masked_action_data,
                                                                         temperature=self.temperature, greedy=True)
            else:
                actions, skill_input = self.primitive_policy.get_actions(image_data, goal_data, None,
                                                                         masked_action_data, lang_input=goal_skill_data,
                                                                         temperature=self.temperature, greedy=True)
        else:
            actions, skill_input = self.primitive_policy.get_actions(image_data, goal_data, variational_k,
                                                                     masked_action_data, temperature=self.temperature,
                                                                     greedy=True)


        return actions, variational_k, variational_m, skill_input




    def rollout_latent_policy(self, image_data, action_data, goal_data, goal_skill_data, switching_point_data,
                              masked_action_data, variational_k, variational_m):

        self.rollout_timesteps = image_data.shape[1]
        action_data_input = torch.cat(
            (((torch.ones((action_data.shape[0], 1, self.action_dim))) * (-1)).to(self.device), action_data),
            dim=1)
        latent_k_input = torch.cat(
            [((torch.ones((variational_k.shape[0], 1, self.num_policies))) * (-1)).to(self.device),
             variational_k],
            dim=1)
        latent_m_input = torch.cat(
            [((torch.ones((variational_m.shape[0], 1, 2))) * (-1)).to(self.device), variational_m], dim=1)

        for t in range(self.rollout_timesteps):

            new_selected_k, selected_m = self.discrete_policy.get_actions(
                image_data[:, :(t + 1), :], action_data_input[:, :(t + 1), :], goal_data[:, :(t + 1), :],
                latent_k_input[:, :(t + 1), :], latent_m_input[:, :(t + 1), :], switching_point_data[:, :(t + 1), :],
                temperature=self.temperature, greedy=True)

            latent_k_input[:, t + 1] = new_selected_k[:, -1]
            latent_m_input[:, t + 1] = selected_m[:, -1]



            if self.pretrain:
                actions, skill_input = self.primitive_policy.get_actions(image_data[:, :(t + 1), :],
                                                                         goal_data[:, :(t + 1), :], None,
                                                                         masked_action_data[:, :(t + 1), :],
                                                                         lang_input=goal_skill_data[:, :(t + 1), :],
                                                                         temperature=self.temperature, greedy=True)
            else:
                actions, skill_input = self.primitive_policy.get_actions(image_data[:, :(t + 1), :],
                                                                         goal_data[:, :(t + 1), :],
                                                                         latent_k_input[:, 1:(t + 2), :],
                                                                         masked_action_data[:, :(t + 1), :],
                                                                         temperature=self.temperature, greedy=True)

            action_data_input[:, t + 1] = actions[-1]
            del actions, skill_input, new_selected_k, selected_m

        self.latent_trajectory_rollout = copy.deepcopy(action_data_input[:, 1:].detach())
        self.latent_k_rollout = copy.deepcopy(latent_k_input[:, 1:].detach())
        self.latent_m_rollout = copy.deepcopy(latent_m_input[:, 1:].detach())
        del latent_k_input, latent_m_input, action_data_input

        return 0

    def rollout_visuals(self, counter, i, get_image=True):

        pass

    def train(self, model=None):

        if model:
            print("Loading model in training.")
            self.load_all_models(model)
        counter = self.args.initial_counter_value

        # For number of training epochs.
        for e in range(self.number_epochs):

            self.current_epoch_running = e
            print("Starting Epoch: ", e)

            if e % self.args.save_freq == 0:
                self.save_all_models("epoch{0}".format(e))

            np.random.shuffle(self.index_list)

            # For every item in the epoch:
            extent = self.updates_per_epoch  # len(self.index_list)
            # FIXME

            i = 0
            for batch in self.train_dataloader:
                self.run_iteration(counter, batch)
                i += 1
                counter = counter + 1
                if i > extent:
                    break

            if e % self.args.eval_freq == 0:
                self.automatic_evaluation(e)

        self.write_and_close()

    def run_iteration(self, counter, batch):



        self.set_epoch(counter)
        self.iter = counter


        image_data, action_data, goal_data, goal_skill_data, switching_point_data, masked_action_data = self.process_data(
            batch)
        if image_data is not None:

            if self.args.highlevel_only:
                variational_k, variational_m, variational_k_probabilities, variational_m_probabilities, prem_variational_k, variational_k_codelen, variational_k_entropys = self.variational_policy.forward(
                    image_data, action_data, goal_skill_data, switching_point_data, self.epsilon, deterministic=True,
                    # FIXME
                    temperature=self.temperature, constrain_m=self.args.constrain_m)
            else:
                variational_k, variational_m, variational_k_probabilities, variational_m_probabilities, prem_variational_k, variational_k_codelen, variational_k_entropys = self.variational_policy.forward(
                    image_data, action_data, goal_skill_data, switching_point_data, self.epsilon,
                    temperature=self.temperature, constrain_m=self.args.constrain_m)


            selected_a, latent_k, latent_m, latent_k_probabilities, latent_m_probabilities = self.evaluate_loglikelihoods(
                image_data, action_data, goal_data, variational_k, variational_m, switching_point_data,
                masked_action_data, goal_skill_data, prem_variational_k)
            # FIXME

            if self.args.train:
                self.new_update_policies(counter, variational_m, variational_k, latent_m, latent_k,
                                         variational_k_probabilities, variational_m_probabilities,
                                         latent_k_probabilities, latent_m_probabilities, action_data,
                                         selected_a, switching_point_data.to(torch.float32), variational_k_codelen,
                                         variational_k_entropys, mask_var=self.args.mask_var)

                self.update_plots(counter)

    def evaluate_metrics(self, epoch=0):
        if not self.args.train:
            test_set_size = 500
        else:
            test_set_size = 10
        self.distances = -np.ones((test_set_size))
        self.distancesaction = -np.ones((test_set_size))
        self.distancesobject = -np.ones((test_set_size))
        self.distances_latent = -np.ones((test_set_size))
        self.objacc = -np.ones((test_set_size))
        self.actacc = -np.ones((test_set_size))
        self.distancesaction_latent = -np.ones((test_set_size))
        self.distancesobject_latent = -np.ones((test_set_size))
        self.objacc_latent = -np.ones((test_set_size))
        self.actacc_latent = -np.ones((test_set_size))
        self.m_all = -np.ones((test_set_size))
        self.k_count = np.zeros([self.num_policies])
        self.k_count_latent = np.zeros([self.num_policies])
        self.all_correct = np.zeros((test_set_size))
        self.a_all_correct = np.zeros((test_set_size))
        self.obj_all_correct = np.zeros((test_set_size))
        # Get test set elements as last (self.test_set_size) number of elements of dataset.
        iter = 0
        self.error_counts = np.zeros([12])
        self.error_object_counts = np.zeros([89])
        self.skill_content = []
        self.skill_content_latent = []
        self.skill_seq = []
        for ii in range(self.num_policies):
            self.skill_content.append([])
            self.skill_content_latent.append([])
        for batch in self.test_dataloader:
            # Collect inputs.

            image_data, action_data, goal_data, goal_skill_data, switching_point_data, masked_action_data = self.process_data(
                batch)

            actions, variational_k, variational_m, skill_input = self.rollout_variational_network(image_data,
                                                                                                  action_data,
                                                                                                  goal_data,
                                                                                                  goal_skill_data,
                                                                                                  switching_point_data,
                                                                                                  masked_action_data)  # orig_assembled_inputs, discrete_inputs, continuous_inputs, orig_subpolicy_inputs, latent_m

            _ = self.rollout_latent_policy(image_data, action_data, goal_data, goal_skill_data, switching_point_data,
                                           masked_action_data, variational_k, variational_m)

            actions_latent = self.latent_trajectory_rollout

            if self.args.highlevel_only:
                # pass
                actions = actions_latent.squeeze(0)
            else:
                actions_action_latent = actions_latent[..., :12].squeeze(0)
                actions_object_latent = actions_latent[..., 12:].squeeze(0)
                actiondata_action = action_data[..., :12].squeeze(0)
                actiondata_object = action_data[..., 12:].squeeze(0)
                _, action_pred_indices_latent = torch.max(actions_action_latent, dim=1)
                _, object_pred_indices_latent = torch.max(actions_object_latent, dim=1)
                _, action_indices = torch.max(actiondata_action, dim=1)
                _, object_indices = torch.max(actiondata_object, dim=1)
                # correct_predictions = (action_pred_indices == action_indices).sum().item()
                indices_latent = []
                indices_a_latent = []
                for id in range(object_indices.shape[0]):
                    if action_indices[id] >= 5:
                        indices_latent.append(id)
                    if action_indices[id] != 2:
                        indices_a_latent.append(id)
                correctaction_predictions = (action_pred_indices_latent == action_indices).sum().item()
                correctobject_predictions = (object_pred_indices_latent == object_indices).sum().item()
                correctobjectacc_predictions = (
                            object_pred_indices_latent[indices_latent] == object_indices[indices_latent]).sum().item()
                correctactionacc_predictions = (
                        action_pred_indices_latent[indices_a_latent] == action_indices[indices_a_latent]).sum().item()
                action_accuracy = correctaction_predictions / actiondata_action.size(0)
                object_accuracy = correctobject_predictions / actiondata_object.size(0)
                objectacc_accuracy = correctobjectacc_predictions / len(indices_latent)
                actionacc_accuracy = correctactionacc_predictions / len(indices_a_latent)
                # self.distances[iter] = accuracy
                self.distancesaction_latent[iter] = action_accuracy
                self.distancesobject_latent[iter] = object_accuracy
                self.objacc_latent[iter] = objectacc_accuracy
                self.actacc_latent[iter] = actionacc_accuracy
                if action_accuracy == 1:
                    self.a_all_correct[iter] = 1
                if object_accuracy == 1:
                    self.obj_all_correct[iter] = 1
                if action_accuracy == 1 and object_accuracy == 1:
                    self.all_correct[iter] = 1
            actions_action = actions[..., :12]
            actions_object = actions[..., 12:]
            actiondata_action = action_data[..., :12].squeeze(0)
            actiondata_object = action_data[..., 12:].squeeze(0)
            _, action_pred_indices = torch.max(actions_action, dim=1)
            _, object_pred_indices = torch.max(actions_object, dim=1)
            _, action_indices = torch.max(actiondata_action, dim=1)
            _, object_indices = torch.max(actiondata_object, dim=1)
            # correct_predictions = (action_pred_indices == action_indices).sum().item()
            indices = []
            indices_a = []
            for id in range(object_indices.shape[0]):
                if action_indices[id] >= 5:
                    indices.append(id)
                if action_indices[id] != 2:
                    indices_a.append(id)
            correctaction_predictions = (action_pred_indices == action_indices).sum().item()
            correctobject_predictions = (object_pred_indices == object_indices).sum().item()
            correctobjectacc_predictions = (object_pred_indices[indices] == object_indices[indices]).sum().item()
            correctactionacc_predictions = (action_pred_indices[indices_a] == action_indices[indices_a]).sum().item()
            for i in action_indices[(action_pred_indices != action_indices)]:
                self.error_counts[i] += 1
            for i in object_indices[(object_pred_indices != object_indices)]:
                self.error_object_counts[i] += 1
            # Calculate accuracy
            action_accuracy = correctaction_predictions / actiondata_action.size(0)
            object_accuracy = correctobject_predictions / actiondata_object.size(0)
            objectacc_accuracy = correctobjectacc_predictions / len(indices)
            actionacc_accuracy = correctactionacc_predictions / len(indices_a)
            # self.distances[iter] = accuracy
            self.distancesaction[iter] = action_accuracy
            self.distancesobject[iter] = object_accuracy
            self.objacc[iter] = objectacc_accuracy
            self.actacc[iter] = actionacc_accuracy

            self.m_all[iter] = variational_m.squeeze(0).sum(dim=0)[0]

            iter += 1
            action_seq = action_data.squeeze(0)[:, :12].argmax(dim=1)
            obj_seq = action_data.squeeze(0)[:, 12:].argmax(dim=1)
            action_sequence = [id_to_action[idd.item()] for idd in action_seq]
            obj_sequence = [id_to_object[idd.item()] for idd in obj_seq]
            last_id1 = None
            last_id2 = None
            if not self.args.train:
                taskskill_seq = []
                for idx in range(len(action_sequence)):
                    action = action_sequence[idx]
                    obj = obj_sequence[idx]
                    skillid_latent = self.latent_k_rollout.squeeze(0).argmax(dim=1)[idx]
                    if last_id1 != skillid_latent:
                        taskskill_seq.append(skillid_latent)
                        self.k_count_latent[skillid_latent] += 1
                        self.skill_content_latent[skillid_latent].append([])
                    skillid = variational_k.squeeze(0).argmax(dim=1)[idx]
                    if last_id2 != skillid:
                        self.k_count[skillid] += 1
                        self.skill_content[skillid].append([])
                        # Append the action to the last list in list_of_lists[id]
                    if obj != 'None':
                        action = action + obj
                    self.skill_content[skillid][-1].append(action)
                    self.skill_content_latent[skillid_latent][-1].append(action)
                    # Update the last_id variable
                    last_id1 = skillid_latent
                    last_id2 = skillid
                # breakpoint()
                self.skill_seq.append(taskskill_seq)
            if iter >= test_set_size:
                break

        if self.args.highlevel_only:
            self.mean_action_accuracy = self.distancesaction.mean()
            self.mean_object_accuracy = self.distancesobject.mean()
            self.mean_object_accuracy_acc = self.objacc.mean()
            self.mean_action_accuracy_acc = self.actacc.mean()
        else:
            self.mean_action_accuracy = self.distancesaction.mean()
            self.mean_object_accuracy = self.distancesobject.mean()
            self.mean_object_accuracy_acc = self.objacc.mean()
            self.mean_action_accuracy_acc = self.actacc.mean()
            self.mean_action_accuracy_latent = self.distancesaction_latent.mean()
            self.mean_object_accuracy_latent = self.distancesobject_latent.mean()
            self.mean_object_accuracy_acc_latent = self.objacc_latent.mean()
            self.mean_action_accuracy_acc_latent = self.actacc_latent.mean()
        # self.mean_distance_latent = self.distances_latent[self.distances_latent > 0].mean()
        if not self.args.train:
            self.m_mean = self.m_all.mean()
            indices_skill = np.arange(100)[self.k_count!=0]
            count_res1 = []
            count_res2 = []
            for index_s in indices_skill:
                all_words = [word for sublist in self.skill_content[index_s] for word in sublist]
                one_count = Counter(all_words)
                most_freq_one = one_count.most_common(10)
                bigrams = zip(all_words, all_words[1:])
                bi_count = Counter(bigrams)
                most_freq_two = bi_count.most_common(10)
                count_res1.append(most_freq_one)
                count_res2.append(most_freq_two)
            unique_numbers = set()
            for tensor_list in self.skill_seq:
                for number in tensor_list:
                    unique_numbers.add(number.item())
            unique_numbers = sorted(list(unique_numbers))

            # Map each unique number to an index
            number_to_index = {number: index for index, number in enumerate(unique_numbers)}

            # Initialize a 5x5 transition matrix (as there are 5 unique numbers)
            vocab_size = len(unique_numbers)
            transition_matrix = np.zeros((vocab_size, vocab_size))

            # Count transitions
            for tensor_list in self.skill_seq:
                for i in range(len(tensor_list) - 1):
                    current_num = tensor_list[i].item()
                    next_num = tensor_list[i + 1].item()
                    transition_matrix[number_to_index[current_num], number_to_index[next_num]] += 1

            # Convert counts to probabilities
            transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)

            # Replace NaN values with 0 (in case of division by zero)
            transition_matrix = np.nan_to_num(transition_matrix)

            # Plot the heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(transition_matrix, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('ALFRED Skill Transition Probabilities', fontsize=14, fontweight='bold')
            plt.xlabel('Skill Label', fontsize=14, fontweight='bold')
            plt.ylabel('Skill Label', fontsize=14, fontweight='bold')
            plt.xticks(np.arange(vocab_size), fontsize=14, fontweight='bold')
            plt.yticks(np.arange(vocab_size), fontsize=14, fontweight='bold')
            plt.grid(False)
            plt.tight_layout()
            plt.show()
            print("m_mean", self.m_mean)
            print("k_count:", self.k_count)
            print("mean_action_accuracy:", self.actacc_latent.mean())
            print("mean_object_accuracy:", self.objacc_latent.mean())
            print("action_error_counts:", self.error_counts)
            print("object_error_counts:", self.error_object_counts)
            # breakpoint()
        # Create save directory:
        upper_dir_name = os.path.join(self.args.logdir, self.args.name, "MEval")
        if not (os.path.isdir(upper_dir_name)):
            os.mkdir(upper_dir_name)

        # model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
        self.dir_name = os.path.join(self.args.logdir, self.args.name, "MEval", "m{}".format(epoch))
        if not (os.path.isdir(self.dir_name)):
            os.mkdir(self.dir_name)

        # self.tf_logger.scalar_summary('Latent Distance', self.mean_distance_latent, epoch)
        # self.tf_logger.scalar_summary('Test accuracy', self.mean_accuracy, epoch)
        self.tf_logger.scalar_summary('Test action accuracy', self.mean_action_accuracy, epoch)
        self.tf_logger.scalar_summary('Test object accuracy', self.mean_object_accuracy, epoch)
        self.tf_logger.scalar_summary('Test object accuracy (after12)', self.mean_object_accuracy_acc, epoch)
        self.tf_logger.scalar_summary('Test action accuracy (except2)', self.mean_action_accuracy_acc, epoch)
        if not self.args.highlevel_only:
            self.tf_logger.scalar_summary('Test action accuracy latent', self.mean_action_accuracy_latent, epoch)
            self.tf_logger.scalar_summary('Test object accuracy latent', self.mean_object_accuracy_latent, epoch)
            self.tf_logger.scalar_summary('Test object accuracy (after12) latent', self.mean_object_accuracy_acc_latent,
                                          epoch)
            self.tf_logger.scalar_summary('Test action accuracy (except2) latent', self.mean_action_accuracy_acc_latent,
                                          epoch)

    def evaluate(self, model):

        if not self.args.train:
            self.set_epoch(0)
        if model:
            self.load_all_models(model)

        # np.set_printoptions(suppress=True, precision=2)

        print("Running Evaluation of State Distances on small test set.")
        self.evaluate_metrics()




    def step_online(self, image_list, vocab, timestep, prev_action=None, mask_list=None, task_goal=None, max_steps=200, presampled_m=None):

        self.max_rollout_timesteps = self.args.max_steps
        if timestep == 0:
            self.online_action_input = ((torch.ones((1, self.max_rollout_timesteps + 1, self.action_dim))) * (-1)).to(
                self.device)
            self.online_k_input = ((torch.ones((1, self.max_rollout_timesteps + 1, self.num_policies))) * (-1)).to(
                self.device)
            self.online_m_input = ((torch.ones((1, self.max_rollout_timesteps + 1, 2))) * (-1)).to(self.device)
            self.mask_input = torch.zeros([1, self.max_rollout_timesteps + 1, self.num_policies]).to(self.device)
            if self.args.use_our_skill:
                for i in [3, 9, 10, 72, 88]:
                    self.mask_input[:, :, i] = 1
            else:
                for i in range(self.num_policies):
                    self.mask_input[:, :, i] = 1

        # For number of rollout timesteps:
        online_image = torch.stack(image_list).view(1, timestep + 1, image_list[0].shape[1], image_list[0].shape[2],
                                                    image_list[0].shape[3])
        online_mask = torch.stack(mask_list).view(1, timestep + 1, -1)
        goal_data = task_goal.view(1, 1, -1).repeat(1, timestep + 1, 1)


        new_selected_k, selected_m = self.discrete_policy.get_actions(
            online_image[:, :(timestep + 1)], self.online_action_input[:, :(timestep + 1), :],
            goal_data[:, :(timestep + 1), :],
            self.online_k_input[:, :(timestep + 1), :], self.online_m_input[:, :(timestep + 1), :], self.online_m_input[:, :(timestep + 1), :], mask_input=self.mask_input[:, :(timestep + 1), :],#TODO
            greedy=self.eval, temperature=self.temperature)

        # continuous policy input
        self.online_k_input[:, timestep + 1] = new_selected_k[:, -1]
        self.online_m_input[:, timestep + 1] = selected_m[:, -1]

        if self.pretrain:
            actions, skill_input = self.primitive_policy.get_actions(online_image[:, :(timestep + 1)],
                                                                     goal_data[:, :(timestep + 1), :], None,
                                                                     online_mask[:, :(timestep + 1), :],
                                                                     lang_input=goal_data[:, :(timestep + 1), :],
                                                                     greedy=True)
        else:
            actions, skill_input = self.primitive_policy.get_actions(online_image[:, :(timestep + 1)],
                                                                     goal_data[:, :(timestep + 1), :],
                                                                     self.online_k_input[:, 1:(timestep + 2), :],
                                                                     online_mask[:, :(timestep + 1), :],
                                                                     greedy=True) #(self.temperature<2)
            # Todo
        if (torch.argmax(new_selected_k[0, -1]) != torch.argmax(self.online_k_input[0, timestep])) and (torch.argmax(new_selected_k[0, -1])==88):
            new_skill=True
        elif (torch.argmax(new_selected_k[0, -1]) != torch.argmax(self.online_k_input[0, timestep])) and (torch.argmax(self.online_k_input[0, timestep])==88):
            new_skill=True
        else:
            new_skill=False

        self.online_action_input[:, timestep + 1] = actions[-1]
        del online_image, online_mask, goal_data

        return actions[-1], new_selected_k[:, -1].squeeze(0).argmax(dim=-1), selected_m, new_selected_k[:, -1].squeeze(0), new_skill

    def update_plots(self, counter):

        self.tf_logger.scalar_summary('Latent Loss', torch.mean(self.total_latent_loss).cpu().detach().numpy(), counter)

        self.tf_logger.scalar_summary('Latent m Loss', torch.mean(self.latent_m_loss).cpu().detach().numpy(), counter)
        self.tf_logger.scalar_summary('Latent k Loss', torch.mean(self.latent_k_loss).cpu().detach().numpy(), counter)

        self.tf_logger.scalar_summary('Policy Loss1',
                                      torch.mean(self.subpolicy_loss1).cpu().detach().numpy(), counter)
        self.tf_logger.scalar_summary('Policy Loss2',
                                      torch.mean(self.subpolicy_loss2).cpu().detach().numpy(), counter)
        self.tf_logger.scalar_summary('Total Loss',
                                      torch.mean(self.total_loss).cpu().detach().numpy(), counter)
        self.tf_logger.scalar_summary('MDL Loss', torch.mean(self.mdl_loss2).cpu().detach().numpy(), counter)
        self.tf_logger.scalar_summary('Epsilon', self.epsilon, counter)

    def process_data(self, batch):

        image_data, action_data, goal_data, goal_skill_data, switching_point_data, masked_action_data = batch

        mask = torch.argmax(action_data, dim=-1) < 5

        # Apply the mask to set the first element to 1 and the rest to 0 in masked_action_data
        masked_action_data[mask] = torch.tensor([1] + [0] * (masked_action_data.size(-1) - 1), dtype=torch.float32).to(
            masked_action_data.device)
        return image_data[:, :-1, ...].to(self.device), action_data.to(self.device), goal_data.to(
            self.device), goal_skill_data.to(self.device), switching_point_data.to(self.device), masked_action_data.to(
            self.device)

    def automatic_evaluation(self, epoch):

        # np.set_printoptions(suppress=True, precision=2)

        print("Running Evaluation of State Distances on small test set.")
        self.evaluate_metrics(epoch=epoch)


    def setup(self):

        # Fixing seeds.
        np.random.seed(seed=self.seed)
        torch.manual_seed(self.seed)
        np.set_printoptions(suppress=True, precision=4)

        self.create_networks()
        self.create_training_ops(fix_subpolicy=self.args.fix_subpolicyfirst, highlevel_only=self.args.highlevel_only)


        extent = self.train_size

        self.index_list = np.arange(0, extent)
        self.initialize_plots()

    def initialize_plots(self):
        if self.args.name is not None:
            logdir = os.path.join(self.args.logdir, self.args.name)
            if not (os.path.isdir(logdir)):
                os.mkdir(logdir)
            logdir = os.path.join(logdir, "logs")
            if not (os.path.isdir(logdir)):
                os.mkdir(logdir)
            # Create TF Logger.
            self.tf_logger = TFLogger.Logger(logdir)
        else:
            self.tf_logger = TFLogger.Logger()


        self.rollout_gif_list = []
        self.gt_gif_list = []

        self.dir_name = os.path.join(self.args.logdir, self.args.name, "MEval")
        if not (os.path.isdir(self.dir_name)):
            os.mkdir(self.dir_name)

    def train_mode(self, highlevel_only=True, critic_only=False):
        self.discrete_policy.train()
        self.variational_policy.eval()
        self.act_encoder.eval()
        self.goal_encoder.eval()

        self.primitive_policy.eval()
        self.obs_encoder.eval()
        # self.goal_encoder.load_state_dict(load_object['Goal_Encoder'])
        self.lang_encoder.eval()
        if critic_only:
            for params in (list(self.discrete_policy.parameters())):
                params.requires_grad = False
        else:
            for params in (list(self.discrete_policy.parameters())):
                params.requires_grad = True
        if highlevel_only and self.args.include_goal:
            for params in (list(self.obs_encoder.parameters()) + list(self.goal_encoder.parameters()) + list(
                    self.lang_encoder.parameters()) + list(self.act_encoder.parameters()) + list(
                self.variational_policy.parameters()) + list(self.primitive_policy.parameters())):
                params.requires_grad = False
        elif highlevel_only and not self.args.include_goal:
            for params in (list(self.obs_encoder.parameters()) + list(
                    self.lang_encoder.parameters()) + list(self.act_encoder.parameters()) + list(
                self.variational_policy.parameters()) + list(self.primitive_policy.parameters())):
                params.requires_grad = False
        else:
            for params in (list(self.obs_encoder.parameters()) + list(self.goal_encoder.parameters()) + list(
                    self.lang_encoder.parameters()) + list(self.act_encoder.parameters()) + list(
                self.variational_policy.parameters()) + list(self.primitive_policy.parameters())):
                params.requires_grad = True
        self.eval=False

    def eval_mode(self):
        self.discrete_policy.eval()
        self.variational_policy.eval()
        self.act_encoder.eval()
        self.goal_encoder.eval()

        self.primitive_policy.eval()
        self.obs_encoder.eval()
        # self.goal_encoder.load_state_dict(load_object['Goal_Encoder'])
        self.lang_encoder.eval()
        self.eval=True

    def sample_data(self):
        ''' sample data from replay buffers to construct a training meta-batch '''
        # collect data from multiple tasks for the meta-batch

        obs, actions, skills, ms, rewards, terms = self.replay_buffer.random_paths(batch_size=self.args.batch_size)

        obs = pad_sequence(obs, batch_first=True, padding_value=-1)
        actions = pad_sequence(actions, batch_first=True, padding_value=-1)
        skills = pad_sequence(skills, batch_first=True, padding_value=-1)
        ms = pad_sequence(ms, batch_first=True, padding_value=-1)
        rewards = pad_sequence(rewards, batch_first=True, padding_value=-1)
        #next_obs = pad_sequence(next_obs, batch_first=True, padding_value=-1)
        terms = pad_sequence(terms, batch_first=True, padding_value=-1)
        return obs.to(self.device), actions.to(self.device), skills.to(self.device), ms.to(self.device), rewards.to(self.device), terms.to(self.device)

    def _update_target_network(self):
        #ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def save_points(self):
        run = wandb.init(
            project="online-training",
            config=self.args,
            dir="../scratch/wandb"
        )

    def upload_log(self, mylog):
        if self.args.save_points:
            wandb.log(mylog)

    def online_train2(self, goal_data, model_copy, update_all=False):
        obs, actions, skills, ms, rewards, terms = self.sample_data()

        #breakpoint()
        action_data_input = torch.cat(
            (((torch.ones((actions.shape[0], 1, self.action_dim))) * (-1)).to(self.device), actions[:, :-1, :]),
            dim=1)
        latent_k_input = torch.concat(
            [((torch.ones((skills.shape[0], 1, self.num_policies))) * (-1)).to(self.device), skills[:, :-1, :]],
            dim=1)
        latent_m_input = torch.concat(
            [((torch.ones((ms.shape[0], 1, 2))) * (-1)).to(self.device), ms[:, :-1, :]], dim=1)
        mask_input = torch.zeros([1, self.args.max_steps + 1, self.num_policies]).to(self.device)
        if self.args.use_our_skill:
            for i in [3, 9, 10, 72, 88]:
                mask_input[:, :, i] = 1
        else:
            for i in range(self.num_policies):
                mask_input[:, :, i] = 1
        mask_pad = (rewards!=-1).any(dim=-1)


        _, latent_m1_, k_prob_orig, m_prob_orig, _ = model_copy.discrete_policy.get_embedding(obs, action_data_input, goal_data[:,:ms.shape[1]], latent_k_input, latent_m_input,
            epsilon=0., rewards=rewards, temperature=1.0, deterministic=False, mask_input=mask_input[:, :ms.shape[1], :])

        latent_k1, latent_m1, k_prob, m_prob, hidden_out = self.discrete_policy.get_embedding(
            obs, action_data_input, goal_data[:, :ms.shape[1]], latent_k_input, latent_m_input,
            epsilon=0., rewards=rewards, temperature=1.0, deterministic=False,
            mask_input=mask_input[:, :ms.shape[1], :], online_training=self.args.not_update_trans)

        latent_k_loss = torch.nn.functional.kl_div(torch.log(k_prob[mask_pad]+ 1e-30), torch.log(k_prob_orig[mask_pad]+ 1e-30).detach(), reduction='batchmean',
                                                        log_target=True)

        latent_m_loss = torch.nn.functional.kl_div(torch.log(m_prob[mask_pad]+ 1e-30), torch.log(m_prob_orig[mask_pad]+ 1e-30).detach(), reduction='batchmean',
                                                    log_target=True)
        if self.args.include_goal:
            q1_pred = self.qf1(hidden_out).gather(2, skills.argmax(dim=-1, keepdim=True).long())[mask_pad] #.detach() FIXME
            q2_pred = self.qf2(hidden_out).gather(2, skills.argmax(dim=-1, keepdim=True).long())[mask_pad]
        else:
            q1_pred = self.qf1(hidden_out).gather(2, skills.argmax(dim=-1, keepdim=True).long())[
                mask_pad]  # .detach() FIXME
            q2_pred = self.qf2(hidden_out).gather(2, skills.argmax(dim=-1, keepdim=True).long())[mask_pad]


        with torch.no_grad():
            hidden_target_input = torch.concat(
                [hidden_out[:, 1:, :], ((torch.ones((ms.shape[0], 1, hidden_out.shape[2]))) * (-1)).to(self.device)],
                dim=1)
            target_q_values1 = self.target_qf1(hidden_target_input.detach())
            target_q_values2 = self.target_qf1(hidden_target_input.detach())
            next_k_prob = torch.concat(
                [k_prob[:, 1:, :], (torch.zeros((ms.shape[0], 1, k_prob.shape[2]))).to(self.device)],
                dim=1)
            rewards = rewards * self.args.reward_scale
            mask_input2 = torch.zeros([skills.shape[0], next_k_prob.shape[1], self.num_policies]).to(self.device)
            if self.args.use_our_skill:
                for i in [3, 9, 10, 72, 88]:
                    mask_input2[:, :, i] = 1
            else:
                for i in range(self.num_policies):
                    mask_input2[:, :, i] = 1
            next_log_pi = torch.zeros((skills.shape[0], next_k_prob.shape[1], self.num_policies)).to(self.device)
            next_log_pi[mask_input2==1] = torch.log(next_k_prob[mask_input2==1] + 1e-30)

            q_target = rewards[mask_pad] + (1. - terms[mask_pad]) * self.args.discount * (((torch.min(target_q_values1, target_q_values2) - self.alpha * next_log_pi) * next_k_prob).sum(dim=-1, keepdim=True))[mask_pad]
        # KL constraint on z if probabilistic

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        ###

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(hidden_out.detach()), self.qf2(hidden_out.detach()))

        # vf update
        log_pi = torch.zeros((skills.shape[0], next_k_prob.shape[1], self.num_policies)).to(self.device)
        log_pi[mask_input2 == 1] = torch.log(k_prob[mask_input2 == 1] + 1e-30)


        self._update_target_network()
        log_policy_target = min_q_new_actions


        policy_loss = ((
                self.alpha * log_pi - log_policy_target
        )*k_prob).sum(dim=-1, keepdim=True)[mask_pad].mean()

        kl_loss = self.args.kl_weight * (0.01 * latent_k_loss + latent_m_loss)

        self.optimizer.zero_grad()
        if update_all:
            kl_loss.backward(retain_graph=True)
        policy_loss.backward()

        self.optimizer.step()
        self.upload_log({"kl loss": np.mean(ptu.get_numpy(latent_k_loss)), "qf loss": np.mean(ptu.get_numpy(qf_loss)), "policy loss": np.mean(ptu.get_numpy(policy_loss)), "Log pi": np.mean(ptu.get_numpy(log_pi.sum(dim=-1))), "m mean": np.mean(ptu.get_numpy(latent_m1[:,:,:1].sum(dim=-1)))})





def main(args):
    torch.autograd.set_detect_anomaly(True)
    args = parse_arguments()

    image_directory = "./data/images_feats"
    goal_skill_feat_dir = "./data/lang_feats"
    switching_points_dir = "./data/switching_points_feats"
    goal_feat_dir = "./data/goal_feats"
    action_dir = "./data/action_sequences.pkl"
    mask_dir = "./data/masks_feats"


    if args.a6000:
        device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = CustomDataset(image_directory=image_directory, goal_skill_feat_dir=goal_skill_feat_dir,
                                 switching_points_dir=switching_points_dir, goal_feat_dir=goal_feat_dir,
                                 action_dir=action_dir, masked_action_dir=mask_dir,
                                 large_dataset=args.large_dataset, more_skills=args.more_skills)

    # Decide on the number of samples for train and test

    train_size = int(args.split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # You can now use these with DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                              collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)

    policy_manager = PolicyManager_Joint(args=args, train_dataloader=train_loader, test_dataloader=test_loader,
                                         train_size=train_size, test_size=test_size, device=device)
    policy_manager.setup()

    if args.train:
        if args.model:
            policy_manager.train(args.model)
        else:
            policy_manager.train()
    else:
        # self.policy_manager.train(self.args.model)
        policy_manager.evaluate(args.model)


if __name__ == '__main__':
    main(sys.argv)