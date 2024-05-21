import pdb 
def run_hf_prompt(text, model, tokenizer, kwargs):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, 
                             max_length = 4096,
                             temperature=kwargs['temperature']) 
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # pdb.set_trace()
    return output 
