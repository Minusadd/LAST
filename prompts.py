import pdb 
from typing import List, Dict

class Prompt:
    def __init__(self, context: str):
        self.context = context

class GPTPrompt(Prompt):
    def __init__(self, 
                context: str, 
                prompt: str, 
                task_desc: str,
                steps: List[Dict]): 
        super().__init__(context)
        self.task_desc = task_desc
        self.step_history = steps
        self.prompt = prompt         

    def __str__(self): 
        content = f"{self.context}\n"
        content += f"{self.task_desc}\n"

        for i in range(len(self.step_history)):
            action = self.step_history[i]['action'].lower()
            content += f"> {action}\n"
            observation = self.step_history[i]['observation'].lower()
            content += f"{observation}\n"
            # for now, we won't use 
            inventory = self.step_history[i]['inventory']
            look = self.step_history[i]['freelook']

        content += self.prompt
        return content