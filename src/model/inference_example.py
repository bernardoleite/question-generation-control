from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    T5Tokenizer
)

import argparse
import sys
sys.path.append('../')

from models import T5FineTuner
from utils import currentdate
from utils import find_string_between_two_substring
import time
import os
import json
import torch

def generate(args, device, qgmodel: T5FineTuner, tokenizer: T5Tokenizer, prompt: str) -> str:

    # enconding info
    source_encoding = tokenizer(
        prompt,
        max_length=args.max_len_input,
        padding='max_length',
        truncation = 'only_second',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    # Put this in GPU (faster than using cpu)
    input_ids = source_encoding['input_ids'].to(device)
    attention_mask = source_encoding['attention_mask'].to(device)

    generated_ids = qgmodel.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=args.num_return_sequences, # defaults to 1
        num_beams=args.num_beams, # defaults to 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! myabe experiment with 5
        max_length=args.max_len_output,
        repetition_penalty=args.repetition_penalty, # defaults to 1.0, #last value was 2.5
        length_penalty=args.length_penalty, # defaults to 1.0
        early_stopping=True, # defaults to False
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    generated_text = ''.join(preds)
 
    return generated_text

def run(args, prompt):
    # Load args (needed for model init) and log json
    params_dict = dict(
        checkpoint_model_path = args.checkpoint_model_path,
        model_name = args.model_name,
        tokenizer_name = args.tokenizer_name,
        batch_size = args.batch_size,
        max_len_input = args.max_len_input,
        max_len_output = args.max_len_output,
        num_beams = args.num_beams,
        num_return_sequences = args.num_return_sequences,
        repetition_penalty = args.repetition_penalty,
        length_penalty = args.length_penalty,
        seed_value = args.seed_value
    )
    params = argparse.Namespace(**params_dict)

    # Load T5 base Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name, model_max_length=512)
    t5_tokenizer.add_tokens(['<skill>','<question>','<answer>','<answertype>','<text>'], special_tokens=True)

    # Load T5 base Model
    if "mt5" in args.model_name:
        t5_model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        t5_model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # https://stackoverflow.com/questions/69191305/how-to-add-new-special-token-to-the-tokenizer
    # https://discuss.huggingface.co/t/adding-new-tokens-while-preserving-tokenization-of-adjacent-tokens/12604
    t5_model.resize_token_embeddings(len(t5_tokenizer))

    # Load T5 fine-tuned model for QG
    checkpoint_model_path = args.checkpoint_model_path
    qgmodel = T5FineTuner.load_from_checkpoint(checkpoint_model_path, hparams=params, t5model=t5_model, t5tokenizer=t5_tokenizer)

    # Put model in freeze() and eval() model. Not sure the purpose of freeze
    # Not sure if this should be after or before changing device for inference.
    qgmodel.freeze()
    qgmodel.eval()

    # Put model in gpu (if possible) or cpu (if not possible) for inference purpose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qgmodel = qgmodel.to(device)
    print ("Device for inference:", device)

    # Generate questions and append predictions
    start_time_generate = time.time()

    generated_text = generate(args, device, qgmodel, t5_tokenizer, prompt)

    print("Inference completed!")

    end_time_generate = time.time()
    gen_total_time = end_time_generate - start_time_generate
    print("Inference time: ", gen_total_time)

    print("Generated Text:\n")
    print(generated_text)

if __name__ == '__main__':
    
    story_text = "Once there were a hare and tortoise. The hare was very proud of his fast speed. He asked the tortoise to have race. The tortoise was slow but he agreed. The race started. The hare ran very fast. The tortoise was left behind. The hare thought he should take some rest and fell asleep. The tortoise did not take a rest. He reached the goal. The tortoise won the race."
    
    prompt_skill_type = '<skill>' + 'setting'
    prompt_answer_type = '<answertype>' + 'implicit'
    prompt_text = '<text>' + story_text
    prompt = prompt_skill_type + prompt_answer_type + prompt_text

    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Generate questions and save them to json file.')

    # Add arguments
    # Add arguments
    parser.add_argument('-cmp','--checkpoint_model_path', type=str, metavar='', default="../../checkpoints/qg_t5_base_512_128_32_10_skill-answertype-text_question-answer_seed_44/model-epoch=04-val_loss=0.95.ckpt", required=False, help='Model folder checkpoint path.')

    parser.add_argument('-mn','--model_name', type=str, metavar='', default="t5-base", required=False, help='Model name.')
    parser.add_argument('-tn','--tokenizer_name', type=str, metavar='', default="t5-base", required=False, help='Tokenizer name.')

    parser.add_argument('-bs','--batch_size', type=int, metavar='', default=1, required=False, help='Batch size.')
    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=128, required=False, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=20, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=1, required=False, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')
    parser.add_argument('-sv','--seed_value', type=int, default=44, metavar='', required=False, help='Seed value.')

    # Parse arguments
    args = parser.parse_args()

    # Start tokenization, encoding and generation
    run(args, prompt)