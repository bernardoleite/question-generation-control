# section1 | [quest1,quest2,quest3,quest4] | [ans1,ans2,ans3,ans4] | [skillA, skillB, skillC,skillC]

import json
import sys
import random

def get_dataset(split_gen):

    random.shuffle(split_gen)
    
    attributes_counter = {'character':0,'setting':0,'action':0,'feeling':0,'causal':0,'outcome':0,'prediction':0}

    # array for saving new dataset
    split_ctrl = []

    # format of each dict
    empy_elem = {
    "sections_uuids": [],
    "sections_uuids_concat": 'null', 
    "questions_reference": [],
    "answers_reference": [],
    "sections_texts_concat": 'null',
    "attributes": []
    }

    # append new element
    split_ctrl.append(empy_elem)

    for question in split_gen:

        sections_uuids = question["sections_uuids"]
        sections_uuids_concat = ''.join(question["sections_uuids"])
        sections_texts = question["sections_texts"]
        question_reference = question["questions_reference"][0] # there is only 1 question_reference in _gen splits
        attributes = [question["attributes"][0]] # only first answer exists in this list, but this is done to match with other splits variable names
        answer_reference = question["answer1"]
        
        sections_uuids_exists = 0

        for elem in split_ctrl:
            if sections_uuids_concat == elem['sections_uuids_concat']:
                sections_uuids_exists = 1
                # append to current elem if it exists
                if attributes[0] == elem["attributes"][0]:
                    elem["questions_reference"].append(question_reference)
                    elem["answers_reference"].append(answer_reference)
        
        # create new element if section uuid does not exist
        if sections_uuids_exists == 0:
            questions_reference = [question_reference]
            answers_reference = [answer_reference]
            new_elem = {
            "sections_uuids": sections_uuids,
            "sections_uuids_concat": sections_uuids_concat, 
            "questions_reference": questions_reference,
            "answers_reference": answers_reference,
            "sections_texts": sections_texts,
            "attributes": attributes
            }
            attributes_counter[attributes[0]] = attributes_counter[attributes[0]] + 1
            split_ctrl.append(new_elem)

    split_ctrl.pop(0) # remove first empty element
    print("Len of dataset created: ", len(split_ctrl))
    print(attributes_counter)
    
    return split_ctrl

def run(train_gen, val_gen, test_gen):

    train_ctrl = get_dataset(train_gen)
    val_ctrl = get_dataset(val_gen)
    test_ctrl = get_dataset(test_gen)

    return train_ctrl, val_ctrl, test_ctrl


if __name__ == '__main__':

    # read json data (processed_gen)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/train.json", "r", encoding='utf-8') as read_file:
        train_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/val.json", "r", encoding='utf-8') as read_file:
        val_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/test.json", "r", encoding='utf-8') as read_file:
        test_gen = json.load(read_file)

    train_ctrl, val_ctrl, test_ctrl  = run(train_gen, val_gen, test_gen)

    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    Path('../../data/FairyTaleQA_Dataset/processed_ctrl').mkdir(parents=True, exist_ok=True)
    
    # save faitytaleqa processed splits to json files
    with open('../../data/FairyTaleQA_Dataset/processed_ctrl/train.json', 'w', encoding='utf-8') as fout:
        json.dump(train_ctrl , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_ctrl/val.json', 'w', encoding='utf-8') as fout:
        json.dump(val_ctrl , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_ctrl/test.json', 'w', encoding='utf-8') as fout:
        json.dump(test_ctrl , fout)

    print("CTRL splits have been successfully created.")