import json
import sys
sys.path.append('../')
import random
import torch
from operator import itemgetter

import numpy as np

from transformers import BertTokenizerFast as BertTokenizer
from models import NarrativeTagger

def predict_attributes(text_narratives, tokenizer, trained_model, device, nr_attributes):
    LABEL_COLUMNS = ['character','setting','action','feeling','causal','outcome','prediction']
    attributes_predicted = {'character': 0,'setting': 0,'action': 0,'feeling': 0,'causal': 0,'outcome': 0,'prediction': 0}

    #THRESHOLD = 0.5

    encoding = tokenizer.encode_plus(
        text_narratives,
        truncation=True,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )

    _, test_prediction = trained_model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
    test_prediction = test_prediction.cpu().flatten().numpy() # https://stackoverflow.com/questions/53900910/typeerror-can-t-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tens

    for label, prediction in zip(LABEL_COLUMNS, test_prediction):
        attributes_predicted[label] = prediction
    
    # https://www.geeksforgeeks.org/python-n-largest-values-in-dictionary/
    # N largest values in dictionary
    # Using sorted() + itemgetter() + items()
    attributes_predicted_best = dict(sorted(attributes_predicted.items(), key = itemgetter(1), reverse = True)[:nr_attributes])
    
    return list(attributes_predicted_best.keys())

def get_dataset(split_cl, type_precl, tokenizer, trained_model, device, random_seed):

    #random.shuffle(split_cl)
    
    # array for saving new dataset
    split_precl = []

    possible_attributes = ['character','setting','action','feeling','causal','outcome','prediction']

    for question in split_cl:

        sections_uuids = question["sections_uuids"]
        sections_uuids_concat = question["sections_uuids_concat"]
        sections_texts = question["sections_texts"]
        questions_reference = question["questions_reference"]
        answers_reference = question["answers_reference"]
        attributes = question["attributes_per_question"]

        # Removing duplicates in attributes with set()
        attributes_unique_gold = list(set(attributes))
        attributes_unique_random = random.sample(possible_attributes, len(attributes_unique_gold))

        if type_precl == "gold":
            attributes_unique = attributes_unique_gold
        elif type_precl == "random":
            attributes_unique = attributes_unique_random
        elif type_precl == "random_dist":
            weights=[0.12, 0.07, 0.27, 0.11, 0.27, 0.11, 0.05]
            weights = [w/sum(weights) for w in weights]
            attributes_unique = np.random.choice(possible_attributes, size=len(attributes_unique_gold), replace=False, p=weights)
            attributes_unique = attributes_unique.tolist()
        elif type_precl == "predicted":
            attributes_unique_predicted = predict_attributes(' '.join(question["sections_texts"]), tokenizer, trained_model, device, len(attributes_unique_gold))
            attributes_unique = attributes_unique_predicted
        else:
            print("Error!")
            sys.exit()

        for att_unique in attributes_unique:
            new_elem = {
            "sections_uuids": sections_uuids,
            "sections_uuids_concat": sections_uuids_concat,
            "sections_texts": sections_texts,
            "questions_reference": questions_reference,
            "answers_reference": answers_reference,
            "attributes_per_question": attributes,
            "target_attribute": att_unique # new element!
            }
            split_precl.append(new_elem)

            #print(attributes_unique)
    
    print("Len of dataset created: ", len(split_precl))
    return split_precl

def run(train_cl, val_cl, test_cl, type_precl, random_seed):
    LABEL_COLUMNS = ['character','setting','action','feeling','causal','outcome','prediction']

    trained_model = NarrativeTagger.load_from_checkpoint(
        "../../model_classifier/best-checkpoint-512.ckpt",
        n_classes=len(LABEL_COLUMNS)
    )
    trained_model.eval()
    trained_model.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)

    BERT_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    train_precl = get_dataset(train_cl, type_precl, tokenizer, trained_model, device, random_seed)
    val_precl = get_dataset(val_cl, type_precl, tokenizer, trained_model, device, random_seed)
    test_precl = get_dataset(test_cl, type_precl, tokenizer, trained_model, device, random_seed)

    return train_precl, val_precl, test_precl

if __name__ == '__main__':

    # read json data (processed_gen)
    with open("../../data/FairyTaleQA_Dataset/processed_cl/train.json", "r", encoding='utf-8') as read_file:
        train_cl = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_cl/val.json", "r", encoding='utf-8') as read_file:
        val_cl = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_cl/test.json", "r", encoding='utf-8') as read_file:
        test_cl = json.load(read_file)

    type_precl = "random_dist"
    radom_seed = 3
    train_precl, val_precl, test_precl  = run(train_cl, val_cl, test_cl, type_precl, radom_seed)

    attributes_counter = {'character':0,'setting':0,'action':0,'feeling':0,'causal':0,'outcome':0,'prediction':0}
    for elem in test_precl:
        attributes_counter[elem["target_attribute"]] = attributes_counter[elem["target_attribute"]] + 1

    attributes_counter_ordered = dict(sorted(attributes_counter.items(), key = itemgetter(1), reverse = True)[:7])
    #a_new = {k: v / total for total in (sum(attributes_counter_ordered.values()),) for k, v in attributes_counter_ordered.items()}

    #print(attributes_counter_ordered)
    #sys.exit()

    if "random" in type_precl:
        type_precl = type_precl + str(radom_seed)

    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    path_folder = '../../data/FairyTaleQA_Dataset/processed_precl' + '_' + type_precl
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    
    # save faitytaleqa processed splits to json files
    path_train = '../../data/FairyTaleQA_Dataset/processed_precl' + '_' + type_precl + '/train.json'
    with open(path_train, 'w', encoding='utf-8') as fout:
        json.dump(train_precl , fout)

    path_val = '../../data/FairyTaleQA_Dataset/processed_precl' + '_' + type_precl + '/val.json'
    with open(path_val, 'w', encoding='utf-8') as fout:
        json.dump(val_precl , fout)

    path_test = '../../data/FairyTaleQA_Dataset/processed_precl' + '_' + type_precl + '/test.json'
    with open(path_test, 'w', encoding='utf-8') as fout:
        json.dump(test_precl , fout)

    print("PRE-CL splits have been successfully created.")