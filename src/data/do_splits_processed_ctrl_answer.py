# section1 | [quest1,quest2,quest3,quest4] | [ans1,ans2,ans3,ans4] | [skillA, skillB, skillC,skillC]

import json
import sys
import random

def get_dataset(split_gen):

    #random.shuffle(split_gen)
    
    answer_counter = {'explicit':0,'implicit':0}
    skill_answer_counter = {'character_explicit':0, 'character_implicit':0,
                            'setting_explicit':0, 'setting_implicit':0,
                            'action_explicit':0, 'action_implicit':0,
                            'feeling_explicit':0, 'feeling_implicit':0,
                            'causal_explicit':0, 'causal_implicit':0,
                            'outcome_explicit':0, 'outcome_implicit':0,
                            'prediction_explicit':0, 'prediction_implicit':0}
    
    # array for saving new dataset
    split_ctrl_answer = []

    # format of each dict
    empy_elem = {
    "sections_uuids": [],
    "sections_uuids_concat": 'null',
    "sections_texts": 'null',
    "questions_reference": [],
    "answers_reference": [],
    "answertype": 'null',
    "attributes": []
    }

    # append new element
    split_ctrl_answer.append(empy_elem)

    for question in split_gen:

        sections_uuids = question["sections_uuids"]
        sections_uuids_concat = ''.join(question["sections_uuids"])
        sections_texts = question["sections_texts"]
        question_reference = question["questions_reference"][0] # there is only 1 question_reference in _gen splits
        answer_reference = question["answer1"]
        answertype = question["ex-or-im1"]
        attributes = [question["attributes"][0]]
        
        new_section_answertype = 1

        for elem in split_ctrl_answer:
            if sections_uuids_concat == elem['sections_uuids_concat']:
                # append to current elem if it exists
                if answertype == elem["ex-or-im1"]:
                    elem["questions_reference"].append(question_reference)
                    elem["answers_reference"].append(answer_reference)
                    elem["attributes"].append(question["attributes"][0])
                    new_section_answertype = 0

        # create new element if section uuid does not exist
        if new_section_answertype == 1:
            questions_reference = [question_reference]
            answers_reference = [answer_reference]
            new_elem = {
            "sections_uuids": sections_uuids,
            "sections_uuids_concat": sections_uuids_concat, 
            "sections_texts": sections_texts,
            "questions_reference": questions_reference,
            "answers_reference": answers_reference,
            "ex-or-im1": answertype,
            "attributes": attributes
            }
            # stats...
            answer_counter[answertype] = answer_counter[answertype] + 1
            skill_answer_type =  new_elem["attributes"][0] + "_" + new_elem["ex-or-im1"]
            skill_answer_counter[skill_answer_type] = skill_answer_counter[skill_answer_type] + 1

            split_ctrl_answer.append(new_elem)

    split_ctrl_answer.pop(0) # remove first empty element
    print("Len of dataset created: ", len(split_ctrl_answer))
    print(answer_counter)
    print(skill_answer_counter)
    #print(answer_type_valid_counter)
    print("\n")
    
    return split_ctrl_answer

def run(train_gen, val_gen, test_gen):

    train_ctrl_answer = get_dataset(train_gen)
    val_ctrl_answer = get_dataset(val_gen)
    test_ctrl_answer = get_dataset(test_gen)

    return train_ctrl_answer, val_ctrl_answer, test_ctrl_answer

if __name__ == '__main__':

    # read json data (processed_gen)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/train.json", "r", encoding='utf-8') as read_file:
        train_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/val.json", "r", encoding='utf-8') as read_file:
        val_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/test.json", "r", encoding='utf-8') as read_file:
        test_gen = json.load(read_file)

    train_ctrl_answer, val_ctrl_answer, test_ctrl_answer  = run(train_gen, val_gen, test_gen)

    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    Path('../../data/FairyTaleQA_Dataset/processed_ctrl_answer').mkdir(parents=True, exist_ok=True)
    
    # save faitytaleqa processed splits to json files
    with open('../../data/FairyTaleQA_Dataset/processed_ctrl_answer/train.json', 'w', encoding='utf-8') as fout:
        json.dump(train_ctrl_answer , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_ctrl_answer/val.json', 'w', encoding='utf-8') as fout:
        json.dump(val_ctrl_answer , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_ctrl_answer/test.json', 'w', encoding='utf-8') as fout:
        json.dump(test_ctrl_answer , fout)

    print("CTRL_answer splits have been successfully created.")