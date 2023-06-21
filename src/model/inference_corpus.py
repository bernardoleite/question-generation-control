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

validation_global = 0

def generate(args, device, qgmodel: T5FineTuner, tokenizer: T5Tokenizer, question: dict) -> str:
    global validation_global

    # enconding info
    if args.encoder_info == "text":
        input_concat = ' '.join(question['sections_texts'])
    elif args.encoder_info == "answer_text":
        input_concat = '<answer>' + question['answers_reference'][0] + '<text>' + ' '.join(question['sections_texts'])
    elif args.encoder_info == "skill_text":
        input_concat = '<skill>' + question['attributes'][0] + '<text>' + ' '.join(question['sections_texts'])
    elif args.encoder_info == "skill_answer_text":
        input_concat = '<skill>' + question['attributes'][0] + '</skill>' + '<answer>' + question['answers_reference'][0] + '</answer>' + '<text>' + ' '.join(question['sections_texts']) + '</text>'
    elif args.encoder_info == "question_text": #qa1
        input_concat = '<question>' + question['questions_reference'][0] + '<text>' + ' '.join(question['sections_texts'])
    elif args.encoder_info == "questiongen_text": #qa2
        input_concat = '<question>' + question['gen_question'] + '<text>' + ' '.join(question['sections_texts'])
    elif args.encoder_info == "answertype_text":
        input_concat = '<answertype>' + question['ex-or-im1'] + '<text>' + ' '.join(question['sections_texts'])
    elif args.encoder_info == "skill_answertype_text":
        input_concat = '<skill>' + question['attributes'][0] + '<answertype>' + question['ex-or-im1'] + '<text>' + ' '.join(question['sections_texts'])
    else:
        print("Error with encoder_info.")
        sys.exit()

    source_encoding = tokenizer(
        input_concat,
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
        num_return_sequences=args.num_return_sequences
        num_beams=args.num_beams,
        max_length=args.max_len_output,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        early_stopping=True, # defaults to False
        use_cache=True
    )

    generated_question, generated_answer = '', ''
    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    if args.decoder_info == 'question': #qg1
        generated_question = ''.join(preds)
    elif args.decoder_info == 'question_answer': #qg2
        generated_str = ''.join(preds)
        res_validation = validate_generated_str(generated_str)
        
        if res_validation == 1:
            generated_question = find_string_between_two_substring(generated_str, '<question>', '<answer>')
            generated_answer = find_string_between_two_substring(generated_str, '<answer>', '<END>')  
        else:
            print("Error during inference, nr. of <question> or <answer> tags is different to 1 !")
            return "ERROR_QUESTION", "ERROR_ANSWER" # this will force any outlier qg result to be zero

    elif args.decoder_info == 'answer': #qa
        generated_answer = ''.join(preds)
    else:
        print("Error during inference: question_answer (2)!")
        sys.exit()
    
    validation_global = validation_global + 1
    return generated_question, generated_answer

def validate_generated_str(str):
    question_begin = str.count("<question>")
    answer_begin = str.count("<answer>")

    if question_begin == 1 and answer_begin == 1:
        return 1
    else:
        return -1

def show_result(generated: str, answer: str, context:str, original_question: str = ''):
    print('Generated: ', generated)
    if original_question:
        print('Original: ', original_question)

    print()
    print('Answer: ', answer)
    print('Context: ', context)
    print('-----------------------------')

def run(args):
    # Load args (needed for model init) and log json
    params_dict = dict(
        checkpoint_model_path = args.checkpoint_model_path,
        predictions_save_path = args.predictions_save_path,
        test_path = args.test_path,
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

    # Read test data
    with open(args.test_path, "r", encoding='utf-8') as read_file:
        test_list = json.load(read_file)

    predictions = []

    # Put model in gpu (if possible) or cpu (if not possible) for inference purpose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qgmodel = qgmodel.to(device)
    print ("Device for inference:", device)

    # Generate questions and append predictions
    start_time_generate = time.time()
    printcounter = 0
    for index, row in enumerate(test_list):

        target_attribute = 'null'
        ex_or_im1 = 'null'
        qa_answer = 'null'
        if "target_attribute" in row:
            target_attribute = row['target_attribute']
        if "ex-or-im1" in row:
            ex_or_im1 = row['ex-or-im1']

        gen_question, gen_answer = generate(args, device, qgmodel, t5_tokenizer, row)

        if args.encoder_info == "questiongen_text":
            qa_answer = gen_answer
            gen_answer = [row['gen_answer']]
            gen_question = row['gen_question']
            if gen_question == 'ERROR_QUESTION' or gen_question == 'ERROR_ANSWER':
                qa_answer = '8T*^19$0@&bR2' # this will force any outlier QA result to be zero

        predictions.append(
            {'sections_uuids': row['sections_uuids'],
            'questions_reference': row['questions_reference'],
            'answers_reference': row['answers_reference'],
            'ex-or-im1': ex_or_im1,
            'sections_texts': row['sections_texts'],
            'attributes': row['attributes'],
            'target_attribute': target_attribute,
            'gen_question': gen_question,
            'gen_answer': gen_answer,
            'qa_answer': qa_answer} 
        )
        printcounter += 1
        if (printcounter == 500):
            print(str(printcounter) + " questions have been generated.")
            printcounter = 0
        #show_result(generated, row['answer'], row['context'], row['question'])

    print("All predictions are completed.")
    print("Number of predictions (q-a-c triples): ", len(predictions))

    end_time_generate = time.time()
    gen_total_time = end_time_generate - start_time_generate
    print("Inference time: ", gen_total_time)

    print("Nr. of RIGHT cases: ", validation_global)

    # Save questions and answers to json file

    #https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    prediction_json_path = args.predictions_save_path
    from pathlib import Path
    Path(prediction_json_path).mkdir(parents=True, exist_ok=True)

    # Save json to json file
    # https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
    with open(prediction_json_path + 'predictions.json', 'w', encoding='utf-8') as file:
        json.dump(predictions, file)

    # Save json params to json file next to predictions
    with open(prediction_json_path + 'params.json', 'w', encoding='utf-8') as file:
        file.write(
            '[' +
            ',\n'.join(json.dumps(str(key)+': '  + str(value)) for key,value in params_dict.items()) +
            ']\n')

    print("Predictions were saved in ", prediction_json_path)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Generate questions and save them to json file.')

    # Add arguments
    parser.add_argument('-cmp','--checkpoint_model_path', type=str, metavar='', default="../../checkpoints/qg_t5_base_512_128_32_10_skill-text_question-answer_seed_44/model-epoch=XX-val_loss=YY.ckpt", required=False, help='Model folder checkpoint path.')
    parser.add_argument('-psp','--predictions_save_path', type=str, metavar='', default="../../predictions/qg_t5_base_512_128_32_10_skill-text_question-answer_seed_44/model-epoch=XX-val_loss=YY/", required=False, help='Folder path to save predictions after inference.')
    parser.add_argument('-tp','--test_path', type=str, metavar='', default="../../data/FairytaleQA_Dataset/processed_ctrl_sk_a/test.json", required=False, help='Test json path.')

    parser.add_argument('-mn','--model_name', type=str, metavar='', default="t5-base", required=False, help='Model name.')
    parser.add_argument('-tn','--tokenizer_name', type=str, metavar='', default="t5-base", required=False, help='Tokenizer name.')

    parser.add_argument('-bs','--batch_size', type=int, metavar='', default=32, required=False, help='Batch size.')
    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=128, required=False, help='Max len output for encoding.')

    parser.add_argument('-enci','--encoder_info', type=str, metavar='', default="skill_text", required=False, help='Information for encoding.')
    parser.add_argument('-deci','--decoder_info', type=str, metavar='', default="question_answer", required=False, help='Information for decoding (generation).')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=5, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=1, required=False, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')
    parser.add_argument('-sv','--seed_value', type=int, default=44, metavar='', required=False, help='Seed value.')

    # Parse arguments
    args = parser.parse_args()

    # Start tokenization, encoding and generation
    run(args)