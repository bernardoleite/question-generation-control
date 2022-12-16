import argparse
import json
import sys
sys.path.append('../')
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from rouge import Rouge
from rouge import FilesRouge
from statistics import mean

def get_corpus_bleu(references, predictions, lower_case=True, language="english"):
    list_of_references = []
    hypotheses = []

    for ref in references:
        ref_processed = word_tokenize(ref, language=language) # tokenize
        if lower_case:
            ref_processed = [each_string.lower() for each_string in ref_processed] # lowercase
        list_of_references.append([ref_processed])

    for pred in predictions:
        pred_processed = word_tokenize(pred, language=language) # tokenize
        if lower_case:
            pred_processed = [each_string.lower() for each_string in pred_processed] # lowercase
        hypotheses.append(pred_processed)

    bleu_1 = corpus_bleu(list_of_references, hypotheses, weights = [1,0,0,0])
    bleu_2 = corpus_bleu(list_of_references, hypotheses, weights = [0.5,0.5,0,0])
    bleu_3 = corpus_bleu(list_of_references, hypotheses, weights = [1/3,1/3,1/3,0])
    bleu_4 = corpus_bleu(list_of_references, hypotheses, weights = [0.25,0.25,0.25,0.25])

    return {"Bleu_1": bleu_1, "Bleu_2": bleu_2, "Bleu_3": bleu_3, "Bleu_4": bleu_4}

def get_rouge_option_1(references, predictions, lower_case=True, language="english"):
    list_of_references = []
    hypotheses = []

    for ref in references:
        ref_processed = word_tokenize(ref, language=language) # tokenize
        if lower_case:
            ref_processed = [each_string.lower() for each_string in ref_processed] # lowercase
        list_of_references.append([ref_processed])

    for pred in predictions:
        pred_processed = word_tokenize(pred, language=language) # tokenize
        if lower_case:
            pred_processed = [each_string.lower() for each_string in pred_processed] # lowercase
        hypotheses.append(pred_processed)

    all_rougeL_p = []
    all_rougeL_r = []
    all_rougeL_f = []
    rouge = Rouge() #init rouge
        
    for index, ref in enumerate(references):
        ref_processed = word_tokenize(ref, language=language) # tokenize
        if lower_case:
            ref_processed = [each_string.lower() for each_string in ref_processed] # lowercase
        ref_processed = ' '.join(ref_processed)

        pred_processed = word_tokenize(predictions[index], language=language) # tokenize
        if lower_case:
            pred_processed = [each_string.lower() for each_string in pred_processed] # lowercase
        pred_processed = ' '.join(pred_processed)

        curr_score = rouge.get_scores(pred_processed, ref_processed)
        all_rougeL_p.append(curr_score[0]["rouge-l"]["p"])
        all_rougeL_r.append(curr_score[0]["rouge-l"]["r"])
        all_rougeL_f.append(curr_score[0]["rouge-l"]["f"])
    
    return {"r": mean(all_rougeL_r), "p": mean(all_rougeL_p), "f": mean(all_rougeL_f)}

def get_rouge_option_2(path_predictions, path_references):
    files_rouge = FilesRouge()
    scores_files_rouge = files_rouge.get_scores(path_predictions, path_references, avg=True)
    return scores_files_rouge

def print_bleu_stats(preds, method, lower_case=True, language="portuguese"):

    for index, pred in enumerate(preds):
        # reference
        ref_processed = word_tokenize(pred["gt_question"], language=language) # tokenize
        if lower_case:
            ref_processed = [each_string.lower() for each_string in ref_processed] # lowercase
        reference = [ref_processed]
        ref_rouge = ' '.join(ref_processed)

        # candidate
        pred_processed = word_tokenize(pred["gen_question"], language=language) # tokenize
        if lower_case:
            pred_processed = [each_string.lower() for each_string in pred_processed] # lowercase
        pred_rouge = ' '.join(pred_processed)

        # bleu
        sentence_bleu_1 = sentence_bleu(reference, pred_processed, weights = [1,0,0,0])
        sentence_bleu_2 = sentence_bleu(reference, pred_processed, weights = [0.5,0.5,0,0])
        sentence_bleu_3 = sentence_bleu(reference, pred_processed, weights = [1/3,1/3,1/3,0])
        sentence_bleu_4 = sentence_bleu(reference, pred_processed, weights = [0.25,0.25,0.25,0.25])

        # rouge
        rouge = Rouge() #init rouge
        curr_score = rouge.get_scores(pred_rouge, ref_rouge)

        preds[index]["bleu_1"] = sentence_bleu_1
        preds[index]["bleu_2"] = sentence_bleu_2
        preds[index]["bleu_3"] = sentence_bleu_3
        preds[index]["bleu_4"] = sentence_bleu_4
        preds[index]["rouge_l"] = curr_score[0]["rouge-l"]["r"]

    if method == 'high':
        new_preds = sorted(preds, key=lambda d: d['bleu_4']) 
    else:
        new_preds = sorted(preds, key=lambda d: d['bleu_4']) 
    
    for pred in new_preds:
        print(pred,"\n")

    return 1

def run(args):
    # Read predictions file
    with open(args.predictions_path + "predictions.json") as file:
        references_predictions = json.load(file)
    
    references = [ref['gt_question'] for ref in references_predictions]
    predictions = [pred['gen_question'] for pred in references_predictions]

    # Get BLEU (results are the same as reported from Du et. al (2017))
    score_corpus_bleu = get_corpus_bleu(references, predictions, lower_case=True, language=args.language)
    print("Score Corpus Bleu: ", score_corpus_bleu)

    result = print_bleu_stats(references_predictions, method="high", lower_case=True, language=args.language)

    # Get BLEU (results are the same as reported from Du et. al (2017))
    score_corpus_bleu = get_corpus_bleu(references, predictions, lower_case=False, language=args.language)
    print("Score Corpus Bleu: ", score_corpus_bleu)

    # Get ROUGE Option 1
    mean_rouge_scores = get_rouge_option_1(references, predictions, lower_case=True, language=args.language)
    print("Mean RougeL Scores: ", mean_rouge_scores)

    # Get ROUGE Option 2
    rouge_files = get_rouge_option_2(args.predictions_path + "gen_questions.txt", args.predictions_path + "gt_questions.txt")
    print("RougeL Files Score: ", rouge_files['rouge-l'])

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Generate questions and save them to json file.')

    # Add arguments
    parser.add_argument('-pp','--predictions_path', type=str, metavar='', default="../predictions/br_v2/", required=True, help='Predictions path.')
    parser.add_argument('-lg','--language', type=str, metavar='', default="portuguese", required=True, help='Language for tokenize.')

    # Parse arguments
    args = parser.parse_args()

    # Start evaluation
    run(args)