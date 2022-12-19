import pandas as pd

import json
import sys
import csv
import numpy as np
sys.path.append('../')
import uuid

fairytaleqa_path_questions = '../../data/FairyTaleQA_Dataset/split_for_training/'
headers_stories= ['section_id', 'section_text']
#headers_questions = ['question_id', 'local-or-sum', 'section_id', 'attribute1', 'attribute2', 'question', 'ex-or-im1', 'answer1', 'answer2', 'answer3', 'ex-or-im2', 'answer4', 'answer5', 'answer6']

def get_files_names(path_questions):
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    # List directory files
    from os import listdir
    from os.path import isfile, join
    files_names = [f for f in listdir(path_questions) if isfile(join(path_questions, f))]

    # get only "-story.csv" files
    files_names_story = []
    for file_name in files_names:
        if "-story.csv" in file_name:
            files_names_story.append(file_name)

    # get only "-questions.csv" files
    files_names_questions = []
    for file_name in files_names:
        if "-questions.csv" in file_name:
            files_names_questions.append(file_name)

    return files_names_story, files_names_questions


def get_story_sections(file_path_story):
    # convert xxx-story.csv to list of lists -> [ ['section1','text1'], ['section2','text2'], ... ]
    with open(file_path_story, encoding="utf8") as fp:
        reader = csv.reader(fp, delimiter=",")
        next(reader, None)  # skip the headers
        story_sections = [row for row in reader]

    # check if there are empty lists
    for story_section in story_sections:
        if not story_section:
            print("-> FOUND ERROR IN <-")
    # remove empty list bug from child-of-mary-story.csv
    # https://stackoverflow.com/questions/4842956/python-how-to-remove-empty-lists-from-a-list
    story_sections = [x for x in story_sections if x != []]

    story_sections = [dict(zip(headers_stories, l)) for l in story_sections]

    for story_sec in story_sections:
        # assign unique section_uuid for each story section
        story_sec['section_uuid'] = uuid.uuid4().hex
        # convert section_id (which is str) to int
        story_sec['section_id'] = int(story_sec['section_id'])

    return story_sections

# include attribute information into queestions
def merge_questions_attributes(story_sections):
    possible_attributes = ['character','setting','action','feeling','causal relationship','outcome resolution','prediction']
    attributes = []
    for row in story_sections:
        if row['attribute1'] in possible_attributes:
            attributes.append(row['attribute1'])
        else:
            print("error! attribute1 is not identified!")
            sys.exit()
        if row['attribute2'] in possible_attributes:
            attributes.append(row['attribute2'])

        # replace attributes name's spaces with '_'
        for idx, att in enumerate(attributes):
            if att == 'causal relationship':
                attributes[idx] = 'causal'
            if att == 'outcome resolution':
                attributes[idx] = 'outcome'

        row['attributes'] = attributes
        attributes = []
    
    return story_sections

def change_headers_names(headers_questions):
    for idx, header in enumerate(headers_questions):
        if header == 'cor_section':
            headers_questions[idx] = 'section_id'
        if header == 'question':
            headers_questions[idx] = 'question_reference'
    return headers_questions

def get_story_questions(file_path_questions):
    # convert xxx-questions.csv to list of lists
    with open(file_path_questions, encoding="utf8") as fp:
        reader = csv.reader(fp, delimiter=",")
        #next(reader, None)  # skip the headers
        story_questions = [row for row in reader]
    
    # (bug) must take headers manually because some files change column order (e.g., the-enchanted-moccasins-questions.csv)
    headers_questions = change_headers_names(story_questions[0])
    story_questions.pop(0)

    # delete last 'comments' column from 2 stories (it is a bug from fairytaleqa)
    for row in story_questions:
        if len(row) > 14:
            del row[-1]

    story_questions = [dict(zip(headers_questions, l)) for l in story_questions]

    return story_questions

def merge_questions_sections(file_name_story, story_sections, story_questions):
    # add section stories to story_questions
    for index, question_story in enumerate(story_questions):
        #assign unique id to question
        question_story['question_uuid'] = uuid.uuid4().hex

        # convert section_id = '1,2,3' to sections_ids = ['1','2','3'] or section_id = '1' to sections_ids = ['1']
        section_ids_str = question_story['section_id'].split(',')
        # https://stackoverflow.com/questions/7368789/convert-all-strings-in-a-list-to-int
        section_ids_int = list(map(int, section_ids_str))
        
        # merge questions with sections
        sections_uuids, sections_texts = [],[]
        for section_id in section_ids_int:
            section_story = next(item for item in story_sections if item['section_id'] == section_id)
            sections_uuids.append(section_story['section_uuid'])
            sections_texts.append(section_story['section_text'])
        
        # assign new columns 'sections_uuids' and 'sections_texts'
        question_story['sections_uuids'] = sections_uuids
        question_story['sections_texts'] = sections_texts
        # assign new column 'story_name'
        question_story['story_name'] = file_name_story
    
    return story_questions

def get_dataset(dataset_split):
    # get files names
    fairytaleqa_path_questions_split = fairytaleqa_path_questions + dataset_split
    files_names_story, files_names_questions = get_files_names(fairytaleqa_path_questions_split)

    faitytaleqa = []

    for file_name_questions in files_names_questions:
        # get xxx-questions.csv path
        file_path_questions = fairytaleqa_path_questions_split + "/" + file_name_questions

        # get xxx-story.csv path
        file_name_story = file_name_questions.replace("-questions.csv", "-story.csv")
        file_path_story = fairytaleqa_path_questions_split + "/" + file_name_story

        # get processed story_sections and story_questions
        story_sections = get_story_sections(file_path_story)
        story_questions = get_story_questions(file_path_questions)

        # merge attributes 1 and 2 (if exists)
        story_questions = merge_questions_attributes(story_questions)

        # merge story_sections and story_questions in one final dataset
        merged_questions_sections = merge_questions_sections(file_name_story, story_sections, story_questions)

        faitytaleqa.extend(merged_questions_sections)

    return faitytaleqa

def run():
    fairytaleqa_train = get_dataset("train")
    fairytaleqa_val = get_dataset("val") 
    fairytaleqa_test = get_dataset("test")

    return fairytaleqa_train, fairytaleqa_val, fairytaleqa_test

if __name__ == '__main__':
    fairytaleqa_train, fairytaleqa_val, fairytaleqa_test  = run()

    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    Path('../../data/FairyTaleQA_Dataset/processed_gen').mkdir(parents=True, exist_ok=True)
    
    # save faitytaleqa processed splits to json files
    with open('../../data/FairyTaleQA_Dataset/processed_gen/train.json', 'w', encoding='utf-8') as fout:
        json.dump(fairytaleqa_train , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_gen/val.json', 'w', encoding='utf-8') as fout:
        json.dump(fairytaleqa_val , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_gen/test.json', 'w', encoding='utf-8') as fout:
        json.dump(fairytaleqa_test , fout)
    
    print("GEN splits have been successfully created.")

    # read data
    #with open('../../data/FairyTaleQA_Dataset/processed/fairytaleqa_test.json', "r", encoding='utf-8') as read_file:
        #data = json.load(read_file)

# https://stackoverflow.com/questions/43175382/python-create-a-pandas-data-frame-from-a-list
#df_questions = pd.DataFrame(all_list_questions, columns = headers)
#sys.exit()

# https://stackoverflow.com/questions/63980549/convert-list-of-lists-to-json-format-first-list-is-header
#all_list_questions.insert(0, headers)
#out = {'fairytaleqa':[dict(zip(all_list_questions[0], row)) for row in all_list_questions[1:]]}

# Getting a list of values from a list of dicts
# https://stackoverflow.com/questions/7271482/getting-a-list-of-values-from-a-list-of-dicts
#questions = [d['question'] for d in out["fairytaleqa"]]
#print(questions)
