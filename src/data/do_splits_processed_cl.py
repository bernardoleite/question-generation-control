import json
import sys
import random

def get_dataset(split_gen):

    random.shuffle(split_gen)
    
    # array for saving new dataset
    split_cl = []

    # format of each dict
    empy_elem = {
    "sections_uuids": [],
    "sections_uuids_concat": 'null', 
    "questions_reference": [],
    "answers_reference": [],
    "sections_texts_concat": 'null',
    "attributes_per_question": [],
    "character": 0,
    "setting": 0,
    "action": 0,
    "feeling": 0,
    "causal": 0,
    "outcome": 0,
    "prediction": 0
    }

    # append new element
    split_cl.append(empy_elem)

    for question in split_gen:

        sections_uuids = question["sections_uuids"]
        sections_uuids_concat = ''.join(question["sections_uuids"])
        sections_texts = question["sections_texts"]
        sections_texts_concat = ' '.join(sections_texts)
        question_reference = question["questions_reference"][0] # there is only 1 question_reference in _gen splits
        answer_reference = question["answer1"]
        attributes = [question["attributes"][0]] # only first attribute exists in this list, but this is done to match with other splits variable names

        new_section = 1

        for elem in split_cl:
            if sections_uuids_concat == elem['sections_uuids_concat']:
                elem["questions_reference"].append(question_reference)
                elem["answers_reference"].append(answer_reference)
                elem["attributes_per_question"].append(attributes[0])
                new_section = 0
        
        # create new element if it is a new <section,skill> pair
        if new_section == 1:
            questions_reference = [question_reference]
            answers_reference = [answer_reference]
            new_elem = {
            "sections_uuids": sections_uuids,
            "sections_uuids_concat": sections_uuids_concat,
            "sections_texts": sections_texts,
            "sections_texts_concat": sections_texts_concat,
            "questions_reference": questions_reference,
            "answers_reference": answers_reference,
            "attributes_per_question": attributes,
            "character": 0,
            "setting": 0,
            "action": 0,
            "feeling": 0,
            "causal": 0,
            "outcome": 0,
            "prediction": 0
            }
            split_cl.append(new_elem)

    split_cl.pop(0) # remove first empty element

    for elem in split_cl:
        # Removing duplicates in attributes with set()
        attributes_unique_gold = list(set(elem["attributes_per_question"]))
        for att in attributes_unique_gold:
            if att == 'character':
                elem["character"] = 1
            if att == 'setting':
                elem["setting"] = 1
            if att == 'action':
                elem["action"] = 1
            if att == 'feeling':
                elem["feeling"] = 1
            if att == 'causal':
                elem["causal"] = 1
            if att == 'outcome':
                elem["outcome"] = 1
            if att == 'prediction':
                elem["prediction"] = 1

    print("Len of dataset created: ", len(split_cl))

    return split_cl

def run(train_gen, val_gen, test_gen):

    train_cl = get_dataset(train_gen)
    val_cl = get_dataset(val_gen)
    test_cl = get_dataset(test_gen)

    return train_cl, val_cl, test_cl

if __name__ == '__main__':

    # read json data (processed_gen)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/train.json", "r", encoding='utf-8') as read_file:
        train_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/val.json", "r", encoding='utf-8') as read_file:
        val_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/test.json", "r", encoding='utf-8') as read_file:
        test_gen = json.load(read_file)

    train_cl, val_cl, test_cl  = run(train_gen, val_gen, test_gen)

    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    Path('../../data/FairyTaleQA_Dataset/processed_cl').mkdir(parents=True, exist_ok=True)
    
    # save faitytaleqa processed splits to json files
    with open('../../data/FairyTaleQA_Dataset/processed_cl/train.json', 'w', encoding='utf-8') as fout:
        json.dump(train_cl , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_cl/val.json', 'w', encoding='utf-8') as fout:
        json.dump(val_cl , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_cl/test.json', 'w', encoding='utf-8') as fout:
        json.dump(test_cl , fout)

    print("CL splits have been successfully created.")