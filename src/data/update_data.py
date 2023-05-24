import json
import re
import sys

def update_question(string):
    # Ensure the first character is uppercase
    if string[0].islower():
        string = string[0].upper() + string[1:]

    # Ensure the last character is a question mark
    if string[-1] != '?':
        string += '?'
    
    return string

def update_answer(string):
    # Ensure the first character is uppercase
    if string[0].islower():
        string = string[0].upper() + string[1:]

    # Ensure the last character is a dot
    if string[-1] != '.':
        string += '.'

    return string

def is_question_valid(string):
    pattern = r'^[A-Z].*\?$'
    return re.match(pattern, string) is not None

def is_answer_valid(string):
    pattern = r'^[A-Z].*\.$'
    return re.match(pattern, string) is not None

def get_invalid_questions(data_set):
    invalid_questions = []
    for elem in data_set:
        for q in elem["questions_reference"]:
            question_valid = is_question_valid(q)
            if question_valid == False:
                invalid_questions.append(q)
            else:
                pass
    return invalid_questions

def get_invalid_answers(data_set):
    invalid_answers = []
    for elem in data_set:
        for a in elem["answers_reference"]:
            answer_valid = is_answer_valid(a)
            if answer_valid == False:
                invalid_answers.append(a)
            else:
                pass
    return invalid_answers

def update_qas(data_set):
    #questions
    for i, elem in enumerate(data_set):
        for j, q in enumerate(elem["questions_reference"]):
            question_valid = is_question_valid(q)
            if question_valid == False:
                question_updated = update_question(q)
                data_set[i]["questions_reference"][j] = question_updated

    #answers
    for i, elem in enumerate(data_set):
        for j, a in enumerate(elem["answers_reference"]):
            answer_valid = is_answer_valid(a)
            if answer_valid == False:
                answer_updated = update_answer(a)
                data_set[i]["answers_reference"][j] = answer_updated

    return data_set

def print_stats(train_gen, val_gen, test_gen):
    train_gen_invalid_questions = get_invalid_questions(train_gen)
    print("Nr of train_gen_invalid_questions: ", len(train_gen_invalid_questions))
    val_gen_invalid_questions = get_invalid_questions(val_gen)
    print("Nr of val_gen_invalid_questions: ", len(val_gen_invalid_questions))
    test_gen_invalid_questions = get_invalid_questions(test_gen)
    print("Nr of test_gen_invalid_questions: ", len(test_gen_invalid_questions))

    train_gen_invalid_answers = get_invalid_answers(train_gen)
    print("Nr of train_gen_invalid_answers: ", len(train_gen_invalid_answers))
    val_gen_invalid_answers = get_invalid_answers(val_gen)
    print("Nr of val_gen_invalid_answers: ", len(val_gen_invalid_answers))
    test_gen_invalid_answers = get_invalid_answers(test_gen)
    print("Nr of test_gen_invalid_answers: ", len(test_gen_invalid_answers))

def run_gen():

    # read processed_gen splits
    with open("../../data/FairyTaleQA_Dataset/processed_gen/train.json", "r", encoding='utf-8') as read_file:
        train_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/val.json", "r", encoding='utf-8') as read_file:
        val_gen = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_gen/test.json", "r", encoding='utf-8') as read_file:
        test_gen = json.load(read_file)

    # get processed_gen stats regarding upper/lower-case and punctuation
    print_stats(train_gen, val_gen, test_gen)

    # update processed_gen questions and answers
    train_gen_v2 = update_qas(train_gen)
    val_gen_v2 = update_qas(val_gen)
    test_gen_v2 = update_qas(test_gen)

    # get processed_gen stats regarding upper/lower-case and punctuation
    print_stats(train_gen_v2, val_gen_v2, test_gen_v2)

    # save new processed_gen splits
    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    Path('../../data/FairyTaleQA_Dataset/processed_gen_v2').mkdir(parents=True, exist_ok=True)
    
    # save faitytaleqa processed splits to json files
    with open('../../data/FairyTaleQA_Dataset/processed_gen_v2/train.json', 'w', encoding='utf-8') as fout:
        json.dump(train_gen_v2 , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_gen_v2/val.json', 'w', encoding='utf-8') as fout:
        json.dump(val_gen_v2 , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_gen_v2/test.json', 'w', encoding='utf-8') as fout:
        json.dump(test_gen_v2 , fout)

    print("GEN V2 splits have been successfully created.")

def run_ctrl_sk_a():

    # read processed_gen splits
    with open("../../data/FairyTaleQA_Dataset/processed_ctrl_sk_a/train.json", "r", encoding='utf-8') as read_file:
        train_ctrl_sk_a = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_ctrl_sk_a/val.json", "r", encoding='utf-8') as read_file:
        val_ctrl_sk_a = json.load(read_file)
    with open("../../data/FairyTaleQA_Dataset/processed_ctrl_sk_a/test.json", "r", encoding='utf-8') as read_file:
        test_ctrl_sk_a = json.load(read_file)

    # get processed_gen stats regarding upper/lower-case and punctuation
    print_stats(train_ctrl_sk_a, val_ctrl_sk_a, test_ctrl_sk_a)

    # update processed_gen questions and answers
    train_ctrl_sk_a_v2 = update_qas(train_ctrl_sk_a)
    val_ctrl_sk_a_v2 = update_qas(val_ctrl_sk_a)
    test_ctrl_sk_a_v2 = update_qas(test_ctrl_sk_a)

    # get processed_gen stats regarding upper/lower-case and punctuation
    print_stats(train_ctrl_sk_a_v2, val_ctrl_sk_a_v2, test_ctrl_sk_a_v2)

    # save new processed_gen splits
    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    from pathlib import Path
    Path('../../data/FairyTaleQA_Dataset/processed_ctrl_sk_a_v2').mkdir(parents=True, exist_ok=True)
    
    # save faitytaleqa processed splits to json files
    with open('../../data/FairyTaleQA_Dataset/processed_ctrl_sk_a_v2/train.json', 'w', encoding='utf-8') as fout:
        json.dump(train_ctrl_sk_a_v2 , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_ctrl_sk_a_v2/val.json', 'w', encoding='utf-8') as fout:
        json.dump(val_ctrl_sk_a_v2 , fout)

    with open('../../data/FairyTaleQA_Dataset/processed_ctrl_sk_a_v2/test.json', 'w', encoding='utf-8') as fout:
        json.dump(test_ctrl_sk_a_v2 , fout)

    print("CTRL_SK_A V2 splits have been successfully created.")

if __name__ == '__main__':
    run_gen()
    run_ctrl_sk_a()