# Copied from qllm-infer/lib/kivi/long_context_example.py


import json
import warnings
warnings.filterwarnings("ignore")


def niah_example(model, tokenizer):
    
    haystack_iter = 5

    haystack_path = "utils/needle_in_a_haystack/haystack_example.jsonl"
    needle_path = "utils/needle_in_a_haystack/needle_examples.jsonl"

    prompt_prefix = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information there.\n"
    prompt_postfix = "What is the pass key? The pass key is "

    model.eval()

    with open(haystack_path, "r") as file:
        haystack_example = json.loads(file.read())

    for line in open(needle_path, "r"):
        needle_example = json.loads(line)

        prompt = prompt_prefix + haystack_iter*haystack_example["input"] + needle_example["input"] + haystack_iter*haystack_example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        target_token_len = tokenizer(needle_example["target"], return_tensors="pt").input_ids.cuda().shape[1]
        
        print("-----------------------------------")
        print("# Input Tokens: {}\n".format(input_ids.shape[1]))
        print("Target: {}".format(needle_example["target"]))

        tokens = model.generate(input_ids, max_new_tokens=target_token_len, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)

        print("Answer: {}".format(answer))
        print("-----------------------------------\n")
