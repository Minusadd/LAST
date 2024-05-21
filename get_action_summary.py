import argparse
import json
import hashlib
import pdb
import time
import pathlib
import time
import os
import re
import traceback
from multiprocessing import Pool
from typing import List, Dict, Any

import openai
import numpy as np
from tqdm import tqdm
import Levenshtein as lev

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys

sys.path.append('../')
from api_tools.gpt4 import run_gpt_prompt, run_fake_gpt_prompt
from api_tools.hf import run_hf_prompt
from data_convert import Steps, write_data, read_data, read_dir, get_augmented_task_ids


# from data.send_sms import send_sms


all_prompts = []

pdbd = []


def clean_and_split_low_str(action_str):
    # remove punctuation and brackets 
    action_str = action_str.strip()
    action_str = re.sub("[\[\]'\"]", "", action_str)
    split_action_str = re.split("[,] ?", action_str)
    split_action_str = [x for x in split_action_str if x != ""]
    if len(split_action_str) > 1 and split_action_str[0] == split_action_str[1]:
        return split_action_str[1:]
    return split_action_str


def count_tokens(x, y):
    split_x = re.split("\s+", x)
    return len(split_x)


class ActionSummarySteps(Steps):
    def __init__(self,
                 steps: List[Dict],
                 tasc_desc: str,
                 task_id: str,
                 preamble: str,
                 alignment_preamble: str,
                 temp_options: List[int] = [0.1, 0.2, 0.3],
                 all_future_actions: bool = False,
                 augment: bool = False,
                 n_workers: int = 1,
                 augment_kwargs: Dict = None,
                 hf_model_name: str = None,
                 skip_init: bool = False,
                 n_shots: int = 0,
                 shot_method: str = "fixed",
                 skip_alignment: bool = False):
        super().__init__(steps, tasc_desc, task_id, all_future_actions, augment, n_workers, augment_kwargs)

        if hf_model_name is None:
            run_fxn = run_gpt_prompt

        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

            run_fxn = lambda x, y: run_hf_prompt(x, model=model, tokenizer=tokenizer, kwargs=y)

        total_tokens = 0
        temp = temp_options[0]
        self.summary_actions = None
        iteri = 0
        while self.summary_actions is None:
            # get action summary
            iteri += 1
            (self.action_summary_text,
             self.action_prompt) = self.get_action_summary(preamble,
                                                           temp=temp,
                                                           max_tokens=1024,
                                                           run_fxn=run_fxn,
                                                           n_shots=n_shots,
                                                           shot_method=shot_method)

            if skip_alignment:
                # don't perform alignment step
                self.summary_actions, self.action_alignment = None, None
            else:
                # TODO

                (self.summary_actions,
                self.action_alignment) = self.get_action_alignment(alignment_preamble, run_fxn = run_fxn, temp=augment_kwargs['temp'])
                # (self.summary_actions,
                #  self.action_alignment, self.task_desc) = self.get_action_alignment(alignment_preamble, run_fxn=run_fxn,
                #                                                                     temp=augment_kwargs['temp'])
            if iteri > 9:
                break

    def detect_bad_output(self, action_sequence, output, lev_threshold=0.5, avg_threshold=0.8):
        # check if output is roughly equal to action sequence
        output = re.split("\n", output)
        similar = [False for _ in range(len(output))]
        for i, (o, a) in enumerate(zip(output, action_sequence)):
            # if more than threshold percentage of the output is the same as the action, it's bad
            if lev.distance(o, a) / len(a) < lev_threshold:
                similar[i] = True
        if np.mean(similar) > avg_threshold:
            return True
        # check if action sequence is really long
        # if len(output) / len(action_sequence) > 0.7:
        #     return True

        # check for presence of low-level actions in the output for electric tasks
        low_level_action_stubs = ['terminal 1', 'terminal 2']
        for stub in low_level_action_stubs:
            for summary_action in output:
                if stub in summary_action:
                    return True
        return False

    def get_shots(self, n_shots, shot_method):
        if shot_method == "fixed":
            shot_str = "\nTask Description: Place a chilled potato slice in front of the toaster.\n" + \
                       """Actions: "LookDown_15", "RotateLeft_90", "MoveAhead_75", "RotateLeft_90", "MoveAhead_50", "RotateRight_90", "MoveAhead_75", "PickupObject ButterKnife", "RotateRight_90", "MoveAhead_25", "RotateRight_90", "MoveAhead_25", "RotateLeft_90", "MoveAhead_25", "RotateRight_90", "MoveAhead_125", "RotateRight_90", "SliceObject Potato", "RotateRight_90", "MoveAhead_100", "RotateRight_90", "MoveAhead_25", "LookDown_15", "OpenObject Fridge", "PutObject ButterKnife", "CloseObject Fridge", "LookUp_15", "RotateRight_90", "MoveAhead_100", "RotateRight_90", "MoveAhead_25", "PickupObject Potato", "RotateRight_90", "MoveAhead_100", "RotateRight_90", "MoveAhead_25", "LookDown_15", "OpenObject Fridge", "PutObject Potato", "CloseObject Fridge", "OpenObject Fridge", "PickupObject Potato", "CloseObject Fridge", "LookUp_15", "RotateLeft_180", "MoveAhead_150", "RotateRight_90", "MoveAhead_50", "PutObject Potato"\n""" + \
                       """Action Summary: First, the robot moves to the toaster. """ + \
                       """Then, it picks up a butter knife and moves it to the refrigerator. """ + \
                       """Next, it uses the knife to slice a potato. """ + \
                       """After that, it puts the potato in front of the toaster."""

            print(shot_str)
            pdb.set_trace()
            return shot_str

    def get_action_summary(self,
                           preamble: str,
                           temp=0.1,
                           max_tokens: int = 256,
                           top_p: float = 1.0,
                           run_fxn: Any = run_gpt_prompt,
                           n: int = 1,
                           n_shots: int = 0,
                           shot_method: str = 'fixed'):
        # query gpt3 for action summary
        # future_actions = [f'"{a}"' for a in self.get_future_actions(0)]
        future_actions = [f'{a}' for a in self.get_future_actions(0)]
        self.action_sequence = future_actions
        if n_shots > 0:
            shot_str = self.get_shots(n_shots, shot_method)
        else:
            shot_str = ""
        prompt_text = f"{preamble}{shot_str}\nGoal: {self.task_desc}\n" + \
            "Actions: " + ", ".join(future_actions) + "\n" \
        # + "Action Summary: First, the robot moves to"
        #todo
        # prompt_text = f"{preamble}{shot_str}\n" + \
        #               "Actions: " + ", ".join(future_actions) + "\n" \
            # anneal temp until action is not bad

        gpt_kwargs = {"max_tokens": max_tokens,
                      "top_p": top_p,
                      "temperature": temp,
                      "n": n}
        action_summary = run_fxn(prompt_text, gpt_kwargs)

        # check if it's a bad output, if it is, decrease the temp

        return action_summary, prompt_text

    def parse_action_summary(self):
        # parse action summary into list of actions
        # split on newline
        action_summary = re.split("\n", self.action_summary_text)
        action_summary = [a.strip() for a in action_summary]
        action_summary = [a for a in action_summary if a != ""]

        # split each summary on new sentence
        action_summary = [re.split("[\.]", x) for x in action_summary]
        # flatten and check for empty strings
        action_summary = [item for sublist in action_summary for item in sublist if item != ""]
        # if summary is just one or two sentences, split on commas
        # if len(action_summary) < 3:
        #     action_summary = [re.split("[,]", x) for x in action_summary]
        #     # flatten and check for empty strings
        #     action_summary = [item for sublist in action_summary for item in sublist if item != ""]

        # check for goal statement
        if "goal" in action_summary[0]:
            goal_statement = action_summary[0]
            action_summary = action_summary[1:]

        # get rid of numbers only
        action_summary = [x for x in action_summary if not x.isdigit()]

        summary_actions = []
        robot_actions = []
        for i, summary_action in enumerate(action_summary):
            summary_actions.append(summary_action)
            # check for "look around" as first action
            # if i == 0 and re.match("^.* look((s)|(ing))? around$", summary_action) is not None:
            #     # can add this to the prompt
            # robot_actions.append(self.action_sequence[0])

        summary_actions[0] = f"First, the robot moves to {summary_actions[0]}"
        return summary_actions, robot_actions

    def parse_action_summary_new(self):
        # parse action summary into list of actions
        # split on newline
        action_summary = re.split("\n", self.action_summary_text)
        action_summary = [a.strip() for a in action_summary if
                          a != "" and a != "{" and a != "}" and a != "[" and a != "]"]

        # get rid of numbers only
        action_summary = [x for x in action_summary if not x.isdigit()]

        summary_actions = []
        robot_actions = []
        for i, summary_action in enumerate(action_summary):
            temp_summary = re.split(":", summary_action)
            summary_actions.append(temp_summary[0].replace('"', '').replace('(', '').replace(')', ''))
            robot_actions.append(
                temp_summary[-1].rstrip(',').strip().replace('"', '').replace('[', '').replace(']', '').replace('(',
                                                                                                                '').replace(
                    ')', '').replace('{', '').replace('}', ''))

        return summary_actions, robot_actions

    def parse_action_summary_new2(self):
        # parse action summary into list of actions
        # split on newline
        action_summary = re.split("\n", self.action_summary_text)
        action_summary = [a.strip() for a in action_summary if
                          a != "" and a != "{" and a != "}" and a != "[" and a != "]"]
        task_desc = action_summary[0].replace('[', '').rstrip(',').replace('"', '').strip()

        # get rid of numbers only
        action_summary = [x for x in action_summary[1:] if not x.isdigit()]

        summary_actions = []
        robot_actions = []
        for i, summary_action in enumerate(action_summary):
            temp_summary = re.split(":", summary_action)
            summary_actions.append(
                temp_summary[0].replace('"', '').replace('(', '').replace(')', '').replace('[', '').replace(']',
                                                                                                            '').replace(
                    '{', '').replace('}', ''))
            robot_actions.append(
                temp_summary[-1].rstrip(',').strip().replace('"', '').replace('[', '').replace(']', '').replace('(',
                                                                                                                '').replace(
                    ')', '').replace('{', '').replace('}', ''))

        return summary_actions, robot_actions, task_desc

    def get_action_alignment(self, alignment_preamble: str, counter=0, temp=0.1, run_fxn=run_gpt_prompt):
        print("running one...")
        # TODO
        summary_actions, robot_actions = self.parse_action_summary_new()
        # summary_actions, robot_actions, task_desc = self.parse_action_summary_new2()
        alignment_summary_to_action = []
        assert len(summary_actions) == len(robot_actions)

        for i in range(len(summary_actions)):
            alignment_summary_to_action.append({"index": i,
                                                "summary_action_str": summary_actions[i],
                                                "robot_actions_str": robot_actions[i]})
        new_action_sequence = []
        for i in range(len(robot_actions)):
            new_action_sequence.extend(clean_and_split_low_str(robot_actions[i]))
            if len(clean_and_split_low_str(robot_actions[i]))>6:
                print(f"invalid because of too many actions in one skills: {len(clean_and_split_low_str(robot_actions[i]))} vs 6")
                # todo
                # return None, None, None
                return None, None
        if new_action_sequence == self.action_sequence:
            is_valid = True
        else:
            is_valid = False
            print(f"invalid because of mismatched actions: {new_action_sequence} vs {self.action_sequence}")

        if len(summary_actions) == 1:
            # need more than 1 summary action
            is_valid = False

        if not is_valid:
            # what to do if you manually force and still don't get a valid alignment
            # skip it for now
            print("NOT VALID")
            # todo
            #return None, None, None
            return None, None
        #return summary_actions, alignment_summary_to_action, task_desc
        # TODO
        return summary_actions, alignment_summary_to_action

    def parse_alignment_output(self, gpt_output: str, summary_actions: List[str]):
        gpt_output = re.sub("Corresponding actions:(\w)", "Corresponding actions: \g<1>", gpt_output)

        def clean_pred_summary(sum):
            # remove utterance-final periods
            sum = sum.strip()
            sum = re.sub("\.$", "", sum).strip()
            return sum

        def is_valid_alignment(pred_pairs, summary_actions, robot_actions):
            # check if the alignment is valid
            # a valid alignment has each low-level action matched to a single
            # summary action, and all summary actions included
            # buffer to consume low-level actions
            robot_action_buffer = [x for x in robot_actions]
            # buffer to consume summary actions
            summary_action_buffer = [x for x in summary_actions]
            sum_idx = 0
            robo_idx = 0
            for pred_sum, pred_low in pred_pairs:
                pred_sum = clean_pred_summary(pred_sum)
                # summary actions don't match up
                try:
                    if pred_sum.strip() == summary_action_buffer[sum_idx].strip():
                        # if summary action matches, consume it
                        sum_idx += 1
                    else:
                        print(
                            f"invalid because of mismatched summary action: {pred_sum} vs {summary_action_buffer[sum_idx]}")
                        return False, None
                except IndexError:
                    print(f"Index error!")
                    return False, None
                pred_low_seq = clean_and_split_low_str(pred_low)
                i = 0
                while (i < len(pred_low_seq) and
                       robo_idx < len(robot_action_buffer) and
                       robot_action_buffer[robo_idx].strip() == pred_low_seq[i].strip()):
                    i += 1
                    robo_idx += 1

                # this means the above loop quite because the actions didn't match
                if i != len(pred_low_seq):
                    print(
                        f"invalid because of incomplete low-level action plan: {pred_low_seq} vs {robot_action_buffer[robo_idx:]}")
                    return False, None
            # check if we've consumed all the low-level actions
            if robo_idx < len(robot_action_buffer):
                print(f"invalid because of leftover low-level actions: {robot_action_buffer[robo_idx:]}")

                # if we're in ScienceWorld, we can try tacking the actions wait and lookaround on to the end
                if robot_action_buffer[robo_idx:] == ['wait1', 'look around']:
                    print("adding wait1, lookaround!")
                    return True, ["wait1", "look around"]
                if robot_action_buffer[robo_idx:] == ['look around']:
                    print("adding lookaround!")
                    return True, ["look around"]

                return False, None
            return True, None

        def clean_prediction(text_pair):
            if len(text_pair) == 2:
                summary_text, correspond_text = text_pair
            elif len(text_pair) == 3:
                remaining, summary_text, correspond_text = text_pair
            else:
                try:
                    summary_text, correspond_text = text_pair[-2:]
                except ValueError:
                    # cut off because of max tokens
                    summary_text = "NONE"
                    correspond_text = "NONE"

            summary_text = re.sub("Summary action [0-9]+: ", "", summary_text)
            correspond_text = re.sub("Corresponding actions:", "", correspond_text)
            correspond_text = ", ".join(clean_and_split_low_str(correspond_text))
            return (summary_text, correspond_text)

        def chunk_action_pairs(flat_list):
            all_pairs = []
            curr_pair = []
            last_was_summary = False
            for line in flat_list:
                if line.startswith("Summary action"):
                    curr_pair.append(line)
                    last_was_summary = True
                elif last_was_summary:
                    # add the actions and reset
                    curr_pair.append(line)
                    last_was_summary = False
                    all_pairs.append(curr_pair)
                    curr_pair = []
                elif line.startswith("Remaining actions"):
                    # skip these lines
                    last_was_summary = False
                    continue
                else:
                    continue
            return all_pairs

        # split on "Summary action"
        top_split_str = re.sub("(Summary action \d+:)", "<SPLITHERE>\\1", gpt_output)
        top_split = re.split("<SPLITHERE>", top_split_str)
        top_split = [x for x in top_split if x != ""]
        # split each one of these
        predicted_action_pairs = []
        for x in top_split:
            split_unit = re.split("\n+", x)
            split_unit = [x for x in split_unit if x != ""]
            try:
                pred_pair = chunk_action_pairs(split_unit)[0]
            except IndexError:
                # pdb.set_trace()
                continue
            predicted_action_pairs.append(pred_pair)

        if len(predicted_action_pairs) == 0:
            return [], False
        predicted_action_pairs = list(map(clean_prediction, predicted_action_pairs))

        is_valid, suffix = is_valid_alignment(predicted_action_pairs, summary_actions, self.action_sequence)
        if suffix is not None:
            predicted_action_pairs[-1] = list(predicted_action_pairs[-1])
            predicted_action_pairs[-1][1] = f"{predicted_action_pairs[-1][1]}, {', '.join(suffix)}"

        return predicted_action_pairs, is_valid

    def to_dict(self):
        data = self.__dict__
        keys_to_keep = ["task_desc",
                        "task_id",
                        "all_future_actions",
                        "action_sequence",
                        "action_prompt",
                        "summary_actions",
                        "action_summary_text",
                        "alignment_output_text",
                        "action_alignment"]
        data = {k: v for k, v in data.items() if k in keys_to_keep}
        if data['action_summary_text'] is None:
            return None
        return data


def get_augmented_task_ids(data: List[Dict]) -> List[str]:
    task_ids = set()
    for split, examples in data.items():
        for ex in examples:
            if ex['action_summary_text'] is not None:
                task_ids.update({ex['task_id']})
    return task_ids


def read_output_dir(dir):
    dir = pathlib.Path(dir)
    data = []
    for file in dir.glob("*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                data.append(json.loads(line))

    other_dir = dir / "raw"
    for fname in other_dir.glob("*json"):
        with open(fname) as f1:
            # print(fname)
            line = json.loads(f1.read())
            data.append(line)
    return data


def get_hash(text_id):
    hash_val = int(hashlib.sha1(text_id.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    return hash_val


def get_completed_task_ids(data):
    completed = []
    for ex in data:
        # text_id = f"{ex['task_id']}" # _{ex['task_desc']}"
        text_id = f"{ex['task_id']}_{ex['task_desc']}"
        is_valid = ex['summary_actions'] is not None
        hash_val = get_hash(text_id)
        # text_id = re.sub("\s+","_", text_id)
        # text_id = re.sub("[\.\,]", "_", text_id)
        # cut off to stop OS error
        # text_id = text_id[0:80]
        if is_valid:
            completed.append(hash_val)
    return completed


def shorten_repeated_actions(steps):
    compressed_steps = []
    last_step = {"action": None}
    curr_step = None
    action_counter = 0
    for step in steps:
        # if action is repeated, increment counter and don't add anything
        if step['action'] == last_step['action']:
            action_counter += 1
            curr_step = last_step
            last_step = step
            continue
        else:
            # last action was repeated
            if action_counter > 0:
                repeated_action = last_step['action']
                # handle wait
                if repeated_action == "wait1":
                    repeated_action = f"wait{action_counter + 1}"
                else:
                    # handle other actions
                    repeated_action = f"{repeated_action} {action_counter + 1} times"
                last_step['action'] = repeated_action
                compressed_steps.append(last_step)

                last_step = step
            else:
                compressed_steps.append(last_step)
                last_step = step

            action_counter = 0
    # add last step
    compressed_steps.append(last_step)
    # cut off None
    compressed_steps = compressed_steps[1:]

    # print([x['action'] for x in steps])
    # print([x['action'] for x in compressed_steps])
    return compressed_steps


def conversion_helper(example):
    seq_sample = example['seq_sample']
    task_id = example['task_id']
    curr_task = example['curr_task']
    variation_idx = example['variation_idx']
    task_var_id = example['task_var_id']
    skip_alignment = example['skip_alignment']
    output_dir = example['output_dir']
    task_desc = seq_sample['taskDescription']

    identifier = f"{task_id}_{variation_idx}_{task_desc}"
    if example['done']:
        print("skipping")
        return None
        # continue

    steps = seq_sample['path']
    steps = shorten_repeated_actions(steps)
    try:
        fold = seq_sample['fold']
    except KeyError:
        fold = None
        return None

    t0 = time.time()
    steps = ActionSummarySteps(steps,
                               task_desc,
                               task_var_id,
                               preamble=preamble,
                               alignment_preamble=alignment_preamble,
                               all_future_actions=True,
                               augment=False,
                               n_workers=1,
                               augment_kwargs={"temp": 0.7},
                               hf_model_name=None,
                               skip_alignment=skip_alignment)

    data_to_write = steps.to_dict()
    if data_to_write is None:
        return None

    to_write = json.dumps(data_to_write)

    # text_id = re.sub("\s+","_", identifier)
    # text_id = re.sub("[\.\,]", "_", text_id)
    # cut off to stop OS error
    # text_id = text_id[0:80]
    text_id = get_hash(identifier)

    print(f"writing to :{output_dir}/raw/{text_id}.json")
    with open(f"{output_dir}/raw/{text_id}.json", "w") as f:
        f.write(to_write)

    return to_write
    # write_pointers[fold].write(json.dumps(data_to_write) + "\n")


def convert(data_dir: str,
            output_dir: str,
            do_augment: bool,
            limit: int,
            overwrite: bool,
            n_workers: int,
            preamble: str,
            alignment_preamble: str,
            augment_kwargs: Dict,
            hf_model_name: str = None,
            skip_alignment: bool = False) -> None:
    raw_data_list = read_dir(data_dir)
    if overwrite:
        data = {"train": [], "dev": [], "test": []}
        completed_tasks = set()
    else:
        data = read_output_dir(output_dir)
        completed_task_ids_and_descs = get_completed_task_ids(data)

    if overwrite:
        seqs_completed = 0
    else:
        seqs_completed = len(completed_task_ids_and_descs)

    print(f"completed {seqs_completed} already")
    timing = []

    split_paths = {"train": f'{output_dir}/sciworld_formatted_train.jsonl',
                   "dev": f'{output_dir}/sciworld_formatted_dev.jsonl',
                   "test": f'{output_dir}/sciworld_formatted_test.jsonl'}

    skipped = 0
    if overwrite:
        mode = "w"
    else:
        mode = "a"
        # try:
    total_tokens = 0
    with open(split_paths['train'], mode) as train_f, \
            open(split_paths['dev'], mode) as dev_f, \
            open(split_paths['test'], mode) as test_f:

        write_pointers = {'train': train_f, 'dev': dev_f, 'test': test_f}
        # prepare all data for processing by putting it into a list
        all_data = []
        for raw_data in raw_data_list:
            for task_id in raw_data.keys():
                curr_task = raw_data[task_id]
                for idx, seq_sample in enumerate(curr_task['goldActionSequences']):
                    variation_idx = seq_sample['variationIdx']
                    task_var_id = f"{task_id}_{variation_idx}"

                    task_desc = curr_task['goldActionSequences'][0]['taskDescription']
                    identifier = f"{task_id}_{variation_idx}_{task_desc}"
                    id_hash = get_hash(identifier)
                    # identifier = f"{task_id}_{variation_idx}_{task_desc}"

                    if limit is not None and seqs_completed >= limit:
                        break

                    done = id_hash in completed_task_ids_and_descs

                    data_for_proc = {"task_id": task_id,
                                     "curr_task": curr_task,
                                     "seq_sample": seq_sample,
                                     "variation_idx": variation_idx,
                                     "task_var_id": task_var_id,
                                     "skip_alignment": skip_alignment,
                                     "output_dir": output_dir,
                                     "done": done}
                    all_data.append(data_for_proc)
                    seqs_completed += 1

        # pdb.set_trace()
        # write_pointers[fold].write(json.dumps(data_to_write) + "\n")
        # for example in tqdm(all_data, desc="Converting and augmenting data"):
        pool = Pool(n_workers)
        # chunk all data up so we can iteratively write it
        # if n_workers > 1:
        #    chunk_size = 3 * n_workers
        # else:
        #    chunk_size = 1
        chunk_size = n_workers
        print(f"total: {len(all_data)}")
        all_data = [x for x in all_data if not x['done']]
        print(f"total: {len(all_data)}")

        all_ids = [x['task_var_id'] for x in all_data]
        # chunked_data = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
        # for chunk in chunked_data:
        results = pool.map(conversion_helper, all_data)
        # results = map(conversion_helper, chunk)
        # for inp, result in zip(chunk, results):
        for inp, result in zip(all_data, results):
            if result is None:
                skipped += 1
                continue
            try:
                fold = inp['fold']
            except KeyError:
                fold = None
                continue
            write_pointers[fold].write(result + "\n")

    print(f"skipped {skipped}")
    print(f"Average time per sequence: {np.mean(timing)}")
    print(f"total sequences: {seqs_completed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../alignments_alfred_full_11-03-2022/train",
                        help="gold paths directory")
    parser.add_argument("--output_dir", type=str, default="data_subtasks/", help="output directory")
    parser.add_argument("--limit", type=int, default=None, help="limit number of datapoints")
    parser.add_argument("--augment", action='store_true', help="augment input with GPT-4")
    parser.add_argument("--use_history", action="store_true", help="use history for augmenting")
    parser.add_argument("--use_buffer", action="store_true", help="keep questions in buffer for augmenting")
    parser.add_argument("--max_tokens", type=int, default=500, help="max tokens for GPT-4")
    parser.add_argument("--top_p", type=int, default=1, help="top p for GPT-4")
    parser.add_argument("--temp", type=float, default=0.7, help="temperature for GPT-4")
    parser.add_argument("--num_questions", type=int, default=10, help="number of questions to ask")
    parser.add_argument("--n", type=int, default=1, help="n for GPT-4")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing augmented data")
    parser.add_argument("--n_workers", type=int, default=1, help="number of workers for multiprocessing")
    parser.add_argument("--hf_model_name", type=str, default=None, help="name of the hf model to use")
    parser.add_argument("--skip_alignment", action="store_true", help="set to true to skip alignment step")
    args = parser.parse_args()
    augment_kwargs = {k: args.__dict__[k] for k in
                      ['max_tokens', 'top_p', 'temp', 'n', 'use_buffer', 'use_history', "num_questions"]}

    preamble = "You are watching a robot play a text game about science. For the following task and sequence of actions taken by the robot, summarize the actions into high-level goals and steps."
    alignment_preamble = "For each distinct goal or summary action, list the actions that the robot took which correspond to that goal/summary action:"

    # Add  temps: 0.42 and 0.65 
    # 0.42 tends to give good augments but sometimes very short summaries 
    # 0.65 more verbose but sometimes tends to give augments that repeat action sequence exactly 
    # run 0.42 first, heuristically check if its good, if not run 0.65

    # actually, need to anneal the temp: Start high, see what you get. If you get the action sequence, lower until you don't 

    while True:
        try:
            convert(args.data_dir,
                    args.output_dir,
                    preamble=preamble,
                    alignment_preamble=alignment_preamble,
                    do_augment=args.augment,
                    limit=args.limit,
                    overwrite=args.overwrite,
                    n_workers=args.n_workers,
                    augment_kwargs=augment_kwargs,
                    skip_alignment=args.skip_alignment)
        except:
            # get error
            print(traceback.format_exc())
            print("SLEEPING AND RETRYING")
            time.sleep(300)

