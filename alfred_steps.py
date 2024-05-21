import json
from typing import List, Dict, Any
import numpy as np
from multiprocess import Pool
from collections import defaultdict
from tqdm import tqdm
import pdb
import re
import pathlib
import argparse
import time

from get_action_summary import ActionSummarySteps
from data_convert import write_data


def textify(name):
    # split on uppcase letters but keep them
    split_name = re.split("([A-Z][^A-Z]*)", name)
    split_name = [x.lower() for x in split_name if (x != "") and (x is not None) and (x != " ")]

    # remove object 
    split_name = [x for x in split_name if x.strip() != "object"]
    to_ret = " ".join(split_name)
    to_ret = re.sub("pickup", "pick up", to_ret)
    to_ret = re.sub("put", "put down", to_ret)

    return to_ret


class AlfredSteps(ActionSummarySteps):
    def __init__(self,
                 steps,
                 tasc_desc,
                 task_id,
                 preamble,
                 alignment_preamble,
                 temp_options=[0.7, 0.5, 0.3],
                 all_future_actions=False,
                 gold_alignment_data=None,
                 augment=False,
                 n_workers=1,
                 augment_kwargs=None,
                 hf_model_name=None,
                 n_shots=0,
                 shot_method="fixed"):
        super().__init__(steps,
                         tasc_desc,
                         task_id,
                         preamble,
                         alignment_preamble,
                         temp_options,
                         all_future_actions,
                         augment,
                         n_workers,
                         augment_kwargs,
                         hf_model_name=hf_model_name,
                         n_shots=n_shots,
                         shot_method=shot_method)
        self.gold_alignment_data = gold_alignment_data

    @classmethod
    def from_json(cls,
                  json_file: str,
                  preamble: str,
                  alignment_preamble: str,
                  temp_options: List[int] = [0.7, 0.5, 0.3],
                  use_num: bool = True,
                  plain_names: bool = False,
                  augment_kwargs: Dict[str, Any] = None,
                  hf_model_name: str = None,
                  n_shots: int = 0,
                  shot_method: str = "fixed") -> List[Any]:
        task_id = str(pathlib.Path(json_file).parent)
        # extract steps dict from line
        with open(json_file) as f1:
            data = json.load(f1)
        annotations = data['turk_annotations']['anns']
        task_and_high_descs = []
        # get mturk annotations
        for ann in annotations:
            task_desc = ann['task_desc']
            try:
                score = np.mean(ann['votes'])
            except KeyError:
                continue
            if score < 0.6:
                continue
            high_descs = ann['high_descs']
            task_and_high_descs.append((task_desc, high_descs))

        # get plan annotations with alignment
        plan_annotations = data['plan']['high_pddl']
        high_level_actions = {}
        low_level_actions = []
        low_to_high_alignment = {}
        high_to_low_alignment = defaultdict(list)
        # get high level actions by index
        for ann in plan_annotations:
            action_name = ann['discrete_action']['action']
            args = ann['discrete_action']['args']
            arg_str = ", ".join(args)
            action_string = f"{action_name}( {arg_str} )"
            high_level_actions[ann['high_idx']] = action_string
        # get low-level actions by index and add alignment
        low_level_annotations = data['plan']['low_actions']
        for i, ann in enumerate(low_level_annotations):
            action_name = ann['discrete_action']['action']
            if len(ann['discrete_action']['args']) > 0:
                # not sure what to do here
                # TODO (elias): it seems like not all annotators know all object names
                # and when they use different names in their task descriptions (e.g.
                # "ice cream scoop" instead of ladel, it can mess things up
                # since there's a mismatch between the low-level action name and the
                # task description
                if action_name == "PutObject":
                    arg = ann['api_action']['receptacleObjectId'].split("|")[0]
                else:
                    arg = ann['api_action']['objectId'].split("|")[0]
                # arg = ann['api_action']['objectId'].split("|")[0]
                action_name = f"{action_name} {arg}"
            low_level_actions.append(action_name)

            # compress repeated numerical low-level actions
        compressed_low_level = []
        curr_action_num = 0
        last_action_type = None
        last_action_is_numeric = False
        low_to_compressed_mapping = {}

        # group low_level_actions by action type
        grouped_low_level_actions = []
        curr_action_type = None

        for i, action in enumerate(low_level_actions):
            if re.match("\w+_\d+", action) is None:
                grouped_low_level_actions.append([action])
                # action_type = action
                curr_action_type = action
            else:
                action_type, num = action.split("_")
                if action_type == curr_action_type:
                    grouped_low_level_actions[-1].append(action)
                else:
                    grouped_low_level_actions.append([action])
                curr_action_type = action_type

        seq_counter = 0
        for i, action_group in enumerate(grouped_low_level_actions):
            # reduce action group to single action
            if len(action_group) == 1:
                action = action_group[0]
                if use_num:
                    compressed_low_level.append(action)
                else:
                    split_action = action.split("_")
                    if len(split_action) == 1:
                        if plain_names:
                            compressed_low_level.append(textify(action))
                        else:
                            compressed_low_level.append(action)
                    else:
                        if plain_names:
                            compressed_low_level.append(textify(split_action[0]))
                        else:
                            compressed_low_level.append(split_action[0])

                low_to_compressed_mapping[seq_counter] = i
                seq_counter += 1
            else:
                agg_num = 0
                for action in action_group:
                    try:
                        action_type, num = action_group[0].split("_")
                    except ValueError:
                        pdb.set_trace()
                    agg_num += int(num)
                    low_to_compressed_mapping[seq_counter] = i
                    seq_counter += 1
                if use_num:
                    compressed_low_level.append(f"{action_type}_{agg_num}")
                else:
                    if plain_names:
                        compressed_low_level.append(textify(action_type))
                    else:
                        compressed_low_level.append(f"{action_type}")

        for i, ann in enumerate(low_level_annotations):
            low_to_high_alignment[low_to_compressed_mapping[i]] = ann['high_idx']
            if low_to_compressed_mapping[i] not in high_to_low_alignment[ann['high_idx']]:
                high_to_low_alignment[ann['high_idx']].append(low_to_compressed_mapping[i])
        # check for errors

        lows = high_to_low_alignment.values()
        for i, l1 in enumerate(lows):
            for j, l2 in enumerate(lows):
                if i == j:
                    continue
                for e1 in l1:
                    if e1 in l2:
                        print(f"ERROR: {l1} {l2}")
                        pdb.set_trace()

        # convert to a steps List[Dict]
        steps = []
        for low_action in compressed_low_level:
            step = {}
            step['action'] = low_action
            step['observation'] = ""
            step['inventory'] = []
            step['freelook'] = ""
            steps.append(step)

        # each valid turker description counts as a separate
        # so we need to create multiple objects, as many as turkers
        to_ret = []
        for i, (task_desc, high_desc_seq) in enumerate(task_and_high_descs):
            gold_alignment_data = {"low_to_high_alignment": low_to_high_alignment,
                                   "high_to_low_alignment": high_to_low_alignment,
                                   "high_level_actions": high_level_actions,
                                   "high_desc_seq": high_desc_seq}
            obj = cls(steps,
                      task_desc,
                      task_id,
                      preamble=preamble,
                      alignment_preamble=alignment_preamble,
                      gold_alignment_data=gold_alignment_data,
                      temp_options=temp_options,
                      augment_kwargs=augment_kwargs,
                      hf_model_name=hf_model_name,
                      n_shots=n_shots,
                      shot_method=shot_method)
            to_ret.append(obj)
        return to_ret

    def to_dict(self):
        data = self.__dict__
        keys_to_keep = ["task_desc",
                        "task_id",
                        "gold_alignment_data",
                        "all_future_actions",
                        "action_sequence",
                        "action_prompt",
                        "summary_actions",
                        "action_summary_text",
                        "alignment_output_text",
                        "action_alignment"]
        data = {k: v for k, v in data.items() if k in keys_to_keep}
        if data['action_alignment'] is None:
            return None
        return data


def read_dir(data_dir):
    data_dir = pathlib.Path(data_dir)
    all_json_files = data_dir.glob('**/*.json')
    return all_json_files


def read_output_dir(output_dir):
    output_dir = pathlib.Path(output_dir)
    all_json_files = output_dir.glob("**/*.json")
    all_data = []
    for file in all_json_files:
        with open(file) as f1:
            data = json.load(f1)
            all_data.append(data)
    return all_data


def get_completed_task_ids(data):
    return [x['task_id'] for x in data]


def conversion_helper(json_file_list: List[str],
                      output_dir: pathlib.Path,
                      args: argparse.Namespace,
                      preamble: str,
                      alignment_preamble: str,
                      augment_kwargs: Dict[str, Any],
                      hf_model_name: str = None,
                      n_shots: int = 0,
                      shot_method: str = "fixed"):
    if args.overwrite:
        data = {"train": [], "dev": [], "test": []}
        completed_tasks = set()
    else:
        data = read_output_dir(output_dir)
        completed_task_ids = get_completed_task_ids(data)

    if args.overwrite:
        seqs_completed = 0
    else:
        seqs_completed = len(completed_task_ids)

    completed = 0
    timing = []
    split_paths = {"train": pathlib.Path(f'{output_dir}/train/'),
                   # "valid": pathlib.Path(f'{output_dir}/valid/'),
                   # "train_sample": pathlib.Path(f'{output_dir}/train_sample/'),
                   "valid_seen": pathlib.Path(f'{output_dir}/valid_seen/'),
                   # "tests_seen": pathlib.Path(f'{output_dir}/tests_seen')
                   }

    worker_data = []
    for json_file in tqdm(json_file_list, total=len(json_file_list)):
        if str(json_file.parent) in completed_task_ids and not args.overwrite:
            # print(f"skipping")
            continue

        print(f"running file {json_file}")

        objs_for_file = AlfredSteps.from_json(json_file,
                                              preamble,
                                              alignment_preamble,
                                              temp_options=[0.3],
                                              use_num=args.use_num,
                                              plain_names=args.plain_names,
                                              augment_kwargs=augment_kwargs,
                                              hf_model_name=hf_model_name,
                                              n_shots=n_shots,
                                              shot_method=shot_method)

        for i, obj in enumerate(objs_for_file):
            worker_data.append(obj)
            worker_data.extend(objs_for_file)
            data_to_write = obj.to_dict()
            if data_to_write is None:
                continue
            split = json_file.parent.parent.parent.stem
            split_paths[split].mkdir(parents=True, exist_ok=True)
            fname = f"{json_file.parent.parent.stem}_{json_file.parent.stem}_{i}.json"
            with open(split_paths[split] / fname, "w") as f1:
                json.dump(data_to_write, f1, indent=4)
                print(f"wrote to {split_paths[split] / fname}")
            completed += 1

        if args.limit is not None and completed > args.limit:
            break


def convert(data_dir: str,
            output_dir: str,
            args: argparse.Namespace,
            preamble: str,
            alignment_preamble: str,
            augment_kwargs: Dict[str, Any],
            hf_model_name: str = None,
            n_shots: int = 0,
            shot_method: str = "fixed"):
    all_json_files = read_dir(data_dir)
    all_json_files = list(all_json_files)

    if args.n_workers == 1:
        conversion_helper(all_json_files,
                          output_dir=output_dir,
                          args=args,
                          preamble=preamble,
                          alignment_preamble=alignment_preamble,
                          augment_kwargs=augment_kwargs,
                          hf_model_name=hf_model_name,
                          n_shots=n_shots,
                          shot_method=shot_method)

    else:
        # split all_json_files into n_workers batches 
        # and run in parallel
        n_per_worker = len(all_json_files) // args.n_workers
        batches = [all_json_files[i:i + n_per_worker] for i in range(0, len(all_json_files), n_per_worker)]
        arg_tuples = [
            (batch, output_dir, args, preamble, alignment_preamble, augment_kwargs, hf_model_name, n_shots, shot_method)
            for batch in batches]
        print(f"Creating {args.n_workers} workers each with {n_per_worker} files")
        with Pool(args.n_workers) as p:
            p.starmap(conversion_helper, arg_tuples)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../alfred/data/json_2.1.0", help="albert data directory")
    parser.add_argument("--output_dir", type=str, default="data_gpt4/", help="output directory")
    parser.add_argument("--limit", type=int, default=None, help="limit number of datapoints")
    parser.add_argument("--augment", action='store_true', help="augment input with GPT-4")
    parser.add_argument("--use_history", action="store_true", help="use history for augmenting") 
    parser.add_argument("--use_buffer", action="store_true", help="keep questions in buffer for augmenting")
    parser.add_argument("--max_tokens", type=int, default=8000, help="max tokens for GPT-4")
    parser.add_argument("--top_p", type=int, default=1, help="top p for GPT-4")
    parser.add_argument("--temperature", type=float, default=0.3, help="temperature for GPT-4")
    parser.add_argument("--num_questions", type=int, default=10, help="number of questions to ask")
    parser.add_argument("--n", type=int, default=1., help="n for GPT-4")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing augmented data")
    parser.add_argument("--n_workers", type=int, default=16, help="number of workers for multiprocessing")
    parser.add_argument("--use_num", action="store_true", help="use numbers in action sequences")
    parser.add_argument("--no_num", dest="use_num", action="store_false", help="do not use numbers in action sequences")
    parser.add_argument("--plain_names", action="store_true", help="use plain names in action sequences")
    parser.add_argument("--code_names", dest="plain_names", action="store_false", help="use code names in action sequences")
    parser.add_argument("--hf_model_name", type=str, default = None, help = "hf model to use instead of gpt4")
    parser.add_argument("--n_shots", type=int, default=0)
    parser.add_argument("--shot_method", type=str, default="fixed")
    parser.set_defaults(use_num=True)

    args = parser.parse_args()
    #preamble = "You are watching a robot do household tasks. For the following task and sequence of actions taken by the robot, segment the actions into no more than 30 skills where each skill corresponds to one part of the action sequence. The answer should a python dictionary in the form of: {(description of the first skill): (list of the actions (the number of which should be less than 5) that the robot took which correspond to the first skill), (description of the second skill): (list of the actions (the number of which should be less than 5) that the robot took which correspond to the second skill), etc.}. The number of actions assigned to each skill must not exceed 5 but should be larger than 1. The segmenatations should be as reasonable and fine-grained as possible. Please don't use newline character or other special character inside the list. There should not be any leftover actions and should recover the given sequence of actions in the exact same order if we concatenate these actions in the order of the skills."
    preamble = "You are watching a robot do household tasks. For the following task and sequence of actions taken by the robot, segment the actions into no more than 40 skills where each skill corresponds to one part of the action sequence. The answer should a python dictionary in the form of: {(description of the first skill): (list of the actions that the robot took which correspond to the first skill), (description of the second skill): (list of the actions that the robot took which correspond to the second skill), etc.}. The number of actions assigned to each skill should not exceed 5 but should be larger than 1. The segmenatations should be as reasonable and fine-grained as possible. Please don't use newline character or other special character inside the list. There should not be any leftover actions and should recover the given sequence of actions in the exact same order if we concatenate these actions in the order of the skills.\
    \n Example 1: \n Input Goal: turn on light on bureau top while holding clock \n Input Actions: LookDown_15, MoveAhead_150, RotateLeft_90, MoveAhead_50, LookDown_15, PickupObject AlarmClock, LookUp_15, RotateLeft_90, MoveAhead_50, RotateRight_90, MoveAhead_75, RotateRight_90, ToggleObjectOn DeskLamp \
               \n Output: {approach bureau (step 1): [LookDown_15, MoveAhead_150], \n approach bureau (step 2): RotateLeft_90, MoveAhead_50, LookDown_15], \n pick up clock: [PickupObject AlarmClock], \n move to desk lamp (step 1): [LookUp_15, RotateLeft_90, MoveAhead_50], \n move to desk lamp (step 2): [RotateRight_90, MoveAhead_75, RotateRight_90], \n turn on desk lamp: [ToggleObjectOn DeskLamp]} \
    \n Example 2: \n Input Goal: Place a chilled apple piece in a microwave  \n Input Actions: LookDown_15, RotateRight_90, MoveAhead_50, RotateRight_90, MoveAhead_225, RotateLeft_90, MoveAhead_75, RotateLeft_90, MoveAhead_25, RotateRight_90, MoveAhead_175, PickupObject ButterKnife, RotateRight_90, MoveAhead_100, RotateLeft_90, MoveAhead_25, SliceObject Apple, RotateLeft_180, MoveAhead_25, RotateLeft_90, MoveAhead_25, RotateRight_90, MoveAhead_225, LookDown_15, OpenObject Fridge, PutObject Fridge, CloseObject Fridge, LookUp_15, RotateRight_90, MoveAhead_25, RotateRight_90, MoveAhead_250, PickupObject Apple, RotateLeft_180, MoveAhead_25, RotateLeft_90, MoveAhead_25, RotateRight_90, MoveAhead_225, LookDown_15, OpenObject Fridge, PutObject Fridge, CloseObject Fridge, OpenObject Fridge, PickupObject Apple, CloseObject Fridge, LookUp_15, RotateRight_180, MoveAhead_150, RotateLeft_90, MoveAhead_125, LookUp_30, OpenObject Microwave, PutObject Microwave, CloseObject Microwave \
    \n Output: {navigate to kitchen (step 1): [LookDown_15, RotateRight_90, MoveAhead_50], \n navigate to kitchen (step 2): [RotateRight_90, MoveAhead_225], \n navigate to kitchen (step 3): [RotateLeft_90, MoveAhead_75], \n navigate to kitchen (step 4): [RotateLeft_90, MoveAhead_25], \n navigate to kitchen (step 5): [RotateRight_90, MoveAhead_175],\n pick up butterknife: [PickupObject ButterKnife], \n locate apple: [RotateRight_90, MoveAhead_100, RotateLeft_90], \n slice apple: [MoveAhead_25, SliceObject Apple], \n return butterknife and go to fridge (step 1): [RotateLeft_180, MoveAhead_25, RotateLeft_90, MoveAhead_25], \n return butterknife and go to fridge (step 2): [RotateRight_90, MoveAhead_225, LookDown_15], \n open fridge and put butterknife inside: [OpenObject Fridge, PutObject Fridge, CloseObject Fridge], \n navigate back to chilled apple piece: [LookUp_15, RotateRight_90, MoveAhead_25, RotateRight_90, MoveAhead_250], \n pick up chilled apple piece and navigate to the fridge (step 1): [PickupObject Apple, RotateLeft_180, MoveAhead_25], \n pick up chilled apple piece and navigate to the fridge (step 2): [RotateLeft_90, MoveAhead_25, RotateRight_90, MoveAhead_225, LookDown_15], \n open fridge and put apple: [OpenObject Fridge, PutObject Fridge, CloseObject Fridge], \n open fridge and pick up chilled apple piece: [OpenObject Fridge, PickupObject Apple, CloseObject Fridge, LookUp_15], \n navigate to microwave: [RotateRight_180, MoveAhead_150, RotateLeft_90, MoveAhead_125, LookUp_30], \n open microwave and place apple: [OpenObject Microwave, PutObject Microwave, CloseObject Microwave]} "


    alignment_preamble = "For each skill, list the actions that the robot took which correspond to that skill. There should not be any leftover actions and should recover the given sequence of actions if we concatenate these actions in the order of the skills. "  # Concatenating all your listed actions should recover the exact original action sequence."
    # preamble = "You are watching a robot do household tasks. For the following task and sequence of actions taken by the robot, summarize the actions in a way a person would understand, using no more than 8 high-level actions."
    # alignment_preamble = "For each distinct goal or summary action, list the actions that the robot took which correspond to that goal/summary action:"

    # Add  temps: 0.42 and 0.65 
    # 0.42 tends to give good augments but sometimes very short summaries 
    # 0.65 more verbose but sometimes tends to give augments that repeat action sequence exactly 
    # run 0.42 first, heuristically check if its good, if not run 0.65

    # actually, need to anneal the temp: Start high, see what you get. If you get the action sequence, lower until you don't 
    augment_kwargs = {"temp": args.temperature}
    convert(data_dir=args.data_dir, 
            output_dir=args.output_dir, 
            args = args,
            preamble=preamble,
            alignment_preamble=alignment_preamble,
            augment_kwargs=augment_kwargs,
            hf_model_name=args.hf_model_name,
            n_shots=args.n_shots,
            shot_method=args.shot_method) 