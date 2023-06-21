import json
import sys
import random

def get_dataset(split_gen):

    # array for saving new dataset
    split_grouped = []

    # format of each dict
    empy_elem = {
    "sections_uuids": [],
    "sections_uuids_concat": 'null',
    "sections_texts": 'null',
    "questions_reference": [],
    "answers_reference": [],
    "local_or_sum": [],
    "ex_or_im1": [],
    "attributes": [],
    "story_names": []
    }

    # append new element
    split_grouped.append(empy_elem)

    for question in split_gen:

        sections_uuids = question["sections_uuids"]
        sections_uuids_concat = ''.join(question["sections_uuids"])
        sections_texts = question["sections_texts"]
        question_reference = question["questions_reference"][0] # there is only 1 question_reference in _gen splits
        answer_reference = question["answers_reference"][0] # there is only 1 question_reference in _gen splits
        local_or_sum = question["local-or-sum"]
        ex_or_im1 = question["ex-or-im1"]
        attribute = question["attributes"][0] # there is only 1 attribute in _gen splits
        story_name = question["story_name"]

        new_section = 1

        for elem in split_grouped:
            if sections_uuids_concat == elem['sections_uuids_concat']:
                elem["questions_reference"].append(question_reference)
                elem["answers_reference"].append(answer_reference)
                elem["local_or_sum"].append(local_or_sum)
                elem["ex_or_im1"].append(ex_or_im1)
                elem["attributes"].append(attribute)
                elem["story_names"].append(story_name)
                new_section = 0
        
        # create new element if it is a new section appears...
        if new_section == 1:
            questions_reference = [question_reference]
            answers_reference = [answer_reference]
            local_or_sum = [local_or_sum]
            ex_or_im1 = [ex_or_im1]
            attributes = [attribute]
            story_names = [story_name]

            new_elem = {
            "sections_uuids": sections_uuids,
            "sections_uuids_concat": sections_uuids_concat,
            "sections_texts": sections_texts,
            "questions_reference": questions_reference,
            "answers_reference": answers_reference,
            "local_or_sum": local_or_sum,
            "ex_or_im1": ex_or_im1,
            "attributes": attributes,
            "story_names": story_names
            }
            split_grouped.append(new_elem)

    split_grouped.pop(0) # remove first empty element
    
    return split_grouped

def run(train_gen_v2, val_gen_v2, test_gen_v2):

    train_processed_grouped = get_dataset(train_gen_v2)
    val_processed_grouped = get_dataset(val_gen_v2)
    test_processed_grouped = get_dataset(test_gen_v2)

    return train_processed_grouped, val_processed_grouped, test_processed_grouped

if __name__ == '__main__':

    # read json data (processed_gen_v2)
    with open("../../data/FairyTaleQA_Dataset/processed_gen_v2/train.json", "r", encoding='utf-8') as read_file:
        train_gen_v2 = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen_v2/val.json", "r", encoding='utf-8') as read_file:
        val_gen_v2 = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen_v2/test.json", "r", encoding='utf-8') as read_file:
        test_gen_v2 = json.load(read_file)

    train_processed_grouped, val_processed_grouped, test_processed_grouped  = run(train_gen_v2, val_gen_v2, test_gen_v2)

    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    Path('../../data/FairyTaleQA_Dataset/processed_grouped').mkdir(parents=True, exist_ok=True)
    
    # save faitytaleqa processed splits to json files
    with open('../../data/FairyTaleQA_Dataset/processed_grouped/train.json', 'w', encoding='utf-8') as fout:
        json.dump(train_processed_grouped , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_grouped/val.json', 'w', encoding='utf-8') as fout:
        json.dump(val_processed_grouped , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_grouped/test.json', 'w', encoding='utf-8') as fout:
        json.dump(test_processed_grouped , fout)

    print("GROUPED splits have been successfully created.")