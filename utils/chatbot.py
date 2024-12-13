# Chatbot Playground # JHLEE
import logging
from transformers import TextStreamer

def generate(model, text, tokenizer, max_new_tokens, device):
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs['attention_mask'],
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=1,
        # use_cache=False,
        use_cache=True,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def chatbot_play(model, tokenizer, max_new_tokens, device='cuda'):
    logging.info('Start prompting, give a prompt as you want! (enter exit to close, c to continue to benchmark)')
    while True:
        prompt = input('User: ')
        if prompt=='c':
            logging.info('Exit')
            return
        elif prompt=='exit':
            logging.info('Exit')
            import sys; sys.exit()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        generate(model, text, tokenizer, max_new_tokens, device)
