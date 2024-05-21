import argparse
import json
import pdb 
import time 
import os
from multiprocessing import Pool
from typing import List, Dict
import traceback

import numpy as np
from tqdm import tqdm

from api_tools.gpt4 import run_gpt_prompt, run_fake_gpt_prompt
from prompts import GPTPrompt


all_prompts = []
class Buffer:
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.buffer = None 

class UnlimitedBuffer(Buffer):
    def __init__(self):
        self.buffer = []
    def get_top(self):
        # unlimited so everything is top 
        return [x for x in set(self.buffer)]

class Steps:
    def __init__(self, 
                steps: List[Dict],
                task_desc: str,
                task_id: str, 
                all_future_actions: bool = False,
                augment: bool = False,
                n_workers: int = 1,
                augment_kwargs: Dict = None):
        self.steps = steps 
        self.task_desc = task_desc
        self.task_id = task_id
        self.all_future_actions = all_future_actions
        self.augment = augment
        self.augment_kwargs = augment_kwargs
        self.n_workers = n_workers
        # deprecated
        #if augment_kwargs is not None and augment_kwargs['use_buffer']: 
        #    self.buffer = UnlimitedBuffer()
        #else:
        #    self.buffer = None

    def datapoint_helper(self, idx):
        input_str, label, future_actions, qa_augments = self.get_input_and_label(idx) 

        curr_dat = {'task_id': self.task_id, 'input': Steps.clean(input_str), 
                    "augment": qa_augments, 'target': Steps.clean(label), 
                    "future_actions": future_actions} 

        return curr_dat 

    def get_datapoints(self):
        datapoints = []
        if self.n_workers == 1: 
            for i in range(len(self.steps)):
                datapoints.append(self.datapoint_helper(i))
        else:
            all_idxs = [x for x in range(len(self.steps))]
            with Pool(self.n_workers) as p: 
                datapoints = p.map(self.datapoint_helper, all_idxs) 

        return datapoints

    @staticmethod
    def clean(s:str) -> str:
        clean_toks = ['\n', '\t']
        for tok in clean_toks:
            s = s.replace(tok, ' ')
        return s

    def get_future_actions(self, curr_idx: int) -> List[str]:
        return [s['action'] for s in self.steps[curr_idx:]]

    def get_all_actions(self) -> List[str]:
        return [s['action'] for s in self.steps]

    def get_input_and_label(self,
                            curr_idx: int):
        # get current step 
        curr_step = self.steps[curr_idx]
        # get future action string if needed 
        future_actions = self.get_future_actions(curr_idx)
        curr_action = curr_step['action']
        label = "<extra_id_0> " + curr_action + ' <extra_id_1>'

        # get input string 
        inventory = curr_step['inventory']
        look = curr_step['freelook']
        if curr_idx != 0:
            prev_step = self.steps[curr_idx-1]
            prev_action = prev_step['action']
            prev_obs = prev_step['observation']
            
            input_str = self.task_desc + ' </s> ' + ' ' + inventory + ' ' + look + ' </s> <extra_id_0>'\
                + ' </s> ' + prev_action + ' </s> ' + prev_obs + ' </s>'
        else:
            input_str = self.task_desc + '</s>' + ' ' + inventory + ' ' + look + ' </s> <extra_id_0>'\
                + ' </s>' + ' </s> ' + '</s>'

        qa_augments = None
        if self.augment:
            # get augmented input from GPT-3
            qa_augments = self.get_qa_augments(curr_idx, **self.augment_kwargs)

        return input_str, label, future_actions, qa_augments

    def get_qa_augments(self,
                        curr_idx: int, 
                        use_history: bool = True,
                        use_buffer: bool = True, 
                        max_tokens: int = 500,
                        top_p: int = 1,
                        temperature: float = 0.7,
                        num_questions: int = 10,
                        n: int = 1) -> str:
        if not use_history:
            steps = [self.steps[curr_idx]]
        else:
            steps = self.steps[:curr_idx]

        prompt = GPTPrompt(context = "You are playing a computer game with the following locations: OUTSIDE, FOUNDRY, KITCHEN, GREEN HOUSE, WORKSHOP, ART STUDIO, LIVING ROOM, BEDROOM, BATHROOM.", 
                        prompt = f"Can you think of {num_questions} questions that could give you clues for solving the task? Ask each question and then write an answer for it below.\nQuestion 1:",
                        # prompt = "Can you think of a question that could give you clues for solving the task? Ask a question and then write an answer for it below.\nQuestion:",
                        task_desc = self.task_desc,
                        steps = steps) 
        prompt = str(prompt)
        pdb.set_trace()
        gpt_kwargs= {"max_tokens": max_tokens, 
                    "top_p": top_p,
                    "temperature": temperature,
                    "n": n}
        # output = run_gpt_prompt(prompt, gpt_kwargs)
        output = "" 
        # use fake gpt for development 
        # all_prompts.append(prompt)
        # output = run_fake_gpt_prompt(prompt, gpt_kwargs)
        # add output to buffer 
        self.buffer.buffer.append(output)
        if use_buffer:
            # return all QAs so far 
            to_ret = self.buffer.get_top() 
        else:
            to_ret = [output]

        return to_ret 

def write_data(path: str, data: List[Dict]) -> None:
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def read_dir(data_dir: str) -> List[Dict]:
    raw_data_list = []
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename), 'r') as f:
            raw_data_list.append(json.load(f))
    return raw_data_list

def read_data(path: str)-> List[Dict]:
    try:
        with open(path) as f1:
            return [json.loads(x) for x in f1]
    except FileNotFoundError:
        return []

def read_output_dir(output_dir: str) -> Dict[str, List]:
    data = {"train": [], "dev": [], "test": []}
    data['train'] = read_data(f'{output_dir}/sciworld_formatted_train.jsonl') 
    data['dev'] = read_data(f'{output_dir}/sciworld_formatted_dev.jsonl') 
    data['test'] = read_data(f'{output_dir}/sciworld_formatted_test.jsonl') 
    return data 

def get_augmented_task_ids(data: List[Dict]) -> List[str]:
    task_ids = set()
    for split, examples in data.items():
        for ex in examples:
            if ex['augment'] is not None:
                task_ids.update({ex['task_id']})
    return task_ids

def convert(data_dir: str, output_dir: str, do_augment: bool, limit: int, overwrite: bool, n_workers: int, augment_kwargs) -> None:
    raw_data_list = read_dir(data_dir)
    if overwrite:
        data = {"train": [], "dev": [], "test": []}
        completed_tasks = set()
    else:
        data = read_output_dir(output_dir)
        completed_tasks = get_augmented_task_ids(data)

    if overwrite:
        seqs_completed = 0
        steps_completed = 0
    else:
        seqs_completed = len(completed_tasks)
        steps_completed = sum([len(data[split]) for split in data.keys()])

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
    with open(split_paths['train'], mode) as train_f, \
        open(split_paths['dev'], mode) as dev_f, \
        open(split_paths['test'], mode) as test_f:

        write_pointers = {'train': train_f, 'dev': dev_f, 'test': test_f}
        # prepare all data for processing by putting it into a list 
        all_data = []
        for raw_data in raw_data_list:
            for task_id in raw_data.keys():
                curr_task = raw_data[task_id]
                for variation_idx, seq_sample in enumerate(curr_task['goldActionSequences']):
                    task_var_id = f"{task_id}_{variation_idx}" 
                    if limit is not None and seqs_completed >= limit:  
                        break

                    if not overwrite and task_var_id in completed_tasks:
                        # check if sequence has been completed, if it has, we can skip  
                        skipped += 1
                        continue

                    data_for_proc = {"task_id": task_id, "curr_task": curr_task, "seq_sample": seq_sample, "variation_idx": variation_idx, "task_var_id": task_var_id}
                    all_data.append(data_for_proc)
                    seqs_completed += 1

        for example in tqdm(all_data, desc="Converting and augmenting data"):
            seq_sample = example['seq_sample']
            task_id = example['task_id']
            curr_task = example['curr_task']
            variation_idx = example['variation_idx']
            task_var_id = example['task_var_id']

            task_desc = seq_sample['taskDescription']
            steps = seq_sample['path']
            if len(steps) < 2:
                continue
            fold = seq_sample['fold']

            t0 = time.time()
            steps = Steps(steps, 
                            task_desc, 
                            task_var_id,
                            all_future_actions=True, 
                            augment=do_augment, 
                            n_workers=n_workers,
                            augment_kwargs=augment_kwargs)
            datapoints = steps.get_datapoints()
            data[fold] += datapoints
            for d in datapoints:
                write_pointers[fold].write(json.dumps(d) + "\n")
                
            t1 = time.time()
            elapsed = t1-t0
            timing.append(elapsed)
            seqs_completed += 1
            steps_completed += len(steps.steps)
    # except:
        # Except all errors and print them
        # print(f"Error with task")
        # trace = traceback.format_exc()
        # send_sms(f"Error with task. Script needs restarting", "+19292499275")
        # print(trace)
                
    # if we have reached this point, we can savely overwrite with the final data 
    write_data(split_paths['train'], data['train'])
    write_data(split_paths['dev'], data['dev'])
    write_data(split_paths['test'], data['test'])

    print(f"skipped {skipped}")
    print(f"Average time per sequence: {np.mean(timing)}")
    print(f"total sequences: {seqs_completed}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/goldpaths-all/", help="gold paths directory")
    parser.add_argument("--output_dir", type=str, default="data/", help="output directory")
    parser.add_argument("--limit", type=int, default=None, help="limit number of datapoints")
    parser.add_argument("--augment", action='store_true', help="augment input with GPT-4")
    parser.add_argument("--use_history", action="store_true", help="use history for augmenting") 
    parser.add_argument("--use_buffer", action="store_true", help="keep questions in buffer for augmenting")
    parser.add_argument("--max_tokens", type=int, default=500, help="max tokens for GPT-4")
    parser.add_argument("--top_p", type=int, default=1, help="top p for GPT-4")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for GPT-4")
    parser.add_argument("--num_questions", type=int, default=10, help="number of questions to ask")
    parser.add_argument("--n", type=int, default=1, help="n for GPT-4")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing augmented data")
    parser.add_argument("--n_workers", type=int, default=1, help="number of workers for multiprocessing")

    args = parser.parse_args()
    augment_kwargs = {k:args.__dict__[k] for k in ['max_tokens', 'top_p', 'temperature', 'n', 'use_buffer', 'use_history', "num_questions"]}

    convert(args.data_dir, 
            args.output_dir, 
            do_augment=args.augment, 
            limit=args.limit, 
            overwrite=args.overwrite, 
            n_workers=args.n_workers,
            augment_kwargs=augment_kwargs)
