import os
import pdb 
import openai
import hashlib
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")

openai.api_version = "2023-03-15-preview"

@retry(
    reraise=True,
    stop=stop_after_attempt(100),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
    ),
)
def run_gpt_prompt(text, kwargs): 
    prompt = {
      "prompt": text,
    }
    if kwargs is not None:
      prompt.update(kwargs) 
    #print(prompt)

    response = openai.ChatCompletion.create(deployment_id="gpt-4",
                                        messages=[{"role": "user", "content":prompt['prompt']}],
                                        #prompt=prompt['prompt'],
                                        max_tokens=kwargs['max_tokens'],
                                        top_p=kwargs['top_p'],
                                        temperature=kwargs['temperature'],
                                        n=kwargs['n'])

    response = response["choices"][0]["message"]["content"].strip()
 
    # text = " ".join([x['choices'][0]['text'] for x in responses])
    return response

@retry(
    reraise=True,
    stop=stop_after_attempt(100),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
    ),
)
def run_fake_gpt_prompt(text, kwargs): 
    hash_object = hashlib.md5(text.encode("utf-8"))
    return hash_object.hexdigest()

if __name__ == "__main__":
    input = ("You are watching an agent doing different tasks in gridworld. The tasks involve go to a specific location in the gridworld and sometimes need to pick up something or use a key to open the door. The agent needs to avoid the other object or hitting the wall in this process. For the following task goal and sequence of actions taken by the agent, segment the actions into no more \
             than 20 skills where each skill corresponds to one part of the action sequence. The answer should a python dictionary in the form of: {(description of the first skill): (list of the actions that the robot took which correspond to the first skill), (description of the second skill): (list of the actions that the robot took which correspond to the second skill), etc.}. The number of actions assigned to each skill should not exceed 5 but should be larger than 1. The segmentation should be as reasonable and fine-grained as possible. Please don't use newline character or other special character inside the list. There should not be any leftover actions and should recover the given sequence of actions in the exact same order if we concatenate these actions in the order of the skills. \n \
             Goal: pick up a purple key, then pick up the grey box and open the red door \n Actions: ['Turn Left', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Turn Left', 'Open a door', 'Move Forward', 'Pick up an object', 'Turn right', 'Turn right', 'Move Forward', 'Drop an object', 'Turn right', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Pick up an object', 'Drop an object', 'Turn right', 'Move Forward', 'Turn Left', 'Move Forward', 'Pick up an object', 'Drop an object', 'Turn right', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Turn Left', 'Open a door', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Open a door', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Open a door', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Open a door', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Open a door', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Turn Left', 'Open a door', 'Turn Left', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Pick up an object', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn Left', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Turn Left', 'Move Forward', 'Move Forward', 'Move Forward', 'Turn right', 'Move Forward', 'Move Forward', 'Open a door']"
             )
    response = openai.ChatCompletion.create(deployment_id="gpt-4",
                                            messages=[{"role": "user", "content": input}],
                                            # prompt=
                                            # response_format={'type':'json_object'},
                                            max_tokens=1028,
                                            top_p=1,
                                            temperature=0.3,
                                            n=1)
    print(response)