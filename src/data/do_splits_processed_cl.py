# section1 | [quest1,quest2,quest3,quest4] | [ans1,ans2,ans3,ans4] | [skillA, skillB, skillC,skillC]

import json
import sys
import random

def get_dataset_v2(split_gen):

    all_sections_uuids_repeated = []
    all_sections_uuids_unique = []
    split_cl = []

    for question in split_gen:
        sections_uuids = question["sections_uuids"]
        all_sections_uuids_repeated.extend(sections_uuids)

    all_sections_uuids_unique = list(set(all_sections_uuids_repeated))
    
    for section_uuid in all_sections_uuids_unique:
        new_dict = {"section_uuid": section_uuid}
        for question in split_gen:
            if section_uuid in question["sections_uuids"]:
                split_cl.append()

    return split_cl

def get_dataset(split_gen):

    random.shuffle(split_gen)
    
    attributes_counter = {'character':0,'setting':0,'action':0,'feeling':0,'causal':0,'outcome':0,'prediction':0}

    split_ctrl = []

    empy_elem = {
    "sections_uuids_concat": 'null', 
    "questions_reference": [],
    "answers_reference": [],
    "sections_texts_concat": 'null',
    "attribute": 'null'
    }

    split_ctrl.append(empy_elem)

    for question in split_gen:

        sections_uuids_concat = ''.join(question["sections_uuids"])
        sections_texts_concat = ' '.join(question["sections_texts"])
        question_reference = question["question_reference"]
        attribute = question["attributes"][0]
        answer_reference = question["answer1"]
        sections_uuids_exists = 0

        for elem in split_ctrl:
            if sections_uuids_concat == elem['sections_uuids_concat']:
                sections_uuids_exists = 1
                if attribute == elem["attribute"]:
                    elem["questions_reference"].append(question_reference)
                    elem["answers_reference"].append(answer_reference)
        
        if sections_uuids_exists == 0:
            questions_reference = [question_reference]
            answers_reference = [answer_reference]
            new_elem = {
            "sections_uuids_concat": sections_uuids_concat, 
            "questions_reference": questions_reference,
            "answers_reference": answers_reference,
            "sections_texts_concat": sections_texts_concat,
            "attribute": attribute
            }
            attributes_counter[attribute] = attributes_counter[attribute] + 1
            split_ctrl.append(new_elem)

    split_ctrl.pop(0) # remove first empty element
    print(len(split_ctrl))
    print(attributes_counter)
    
    sys.exit()
    return split_ctrl

def run(train_gen, val_gen, test_gen):

    train_cl = get_dataset(train_gen)
    val_cl = get_dataset(val_gen)
    test_cl = get_dataset(test_gen)

    return train_cl, val_cl, test_cl


if __name__ == '__main__':

    # read data
    with open("../../data/FairyTaleQA_Dataset/processed_gen/fairytaleqa_train.json", "r", encoding='utf-8') as read_file:
        train_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/fairytaleqa_val.json", "r", encoding='utf-8') as read_file:
        val_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/fairytaleqa_test.json", "r", encoding='utf-8') as read_file:
        test_gen = json.load(read_file)

    train_cl, val_cl, test_cl  = run(train_gen, val_gen, test_gen)