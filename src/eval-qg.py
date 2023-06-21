import argparse
import json
import sys
sys.path.append('../')
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from rouge_score import rouge_scorer, scoring
from statistics import mean
#from bert_score import score
from bleurt import score

import evaluate
from evaluate import load

from collections import Counter
from nltk import ngrams

import itertools

from language_tool_python import LanguageTool

#from nlgeval import NLGEval
#nlgeval = NLGEval()  # loads the models

def get_nlgeval(references, predictions, lower_case=True, language="english"):
    list_of_references = []
    hypotheses = []

    for question_refs in references:
        tmp_list_of_refs = []
        for ref in question_refs:
            ref_processed = ref
            if lower_case:
                ref_processed = ref.lower() # lowercase
            tmp_list_of_refs.extend([ref_processed])
        list_of_references.append(tmp_list_of_refs)

    for pred in predictions:
        pred_processed = pred
        if lower_case:
            pred_processed = pred.lower() # lowercase
        hypotheses.append(pred_processed)

    #metrics_dict = nlgeval.compute_metrics(list_of_references, hypotheses)
    return metrics_dict

def get_rouge_option_rouge_scorer(references, predictions, lower_case=True, language="english"):
    rougeL_p_scores, rougeL_r_scores, rougeL_f_scores = [],[],[]
    P_INDEX, R_INDEX, F_INDEX = 0,1,2

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for index, question_refs in enumerate(references):
        references = question_refs
        gen = predictions[index]
        if lower_case:
            references = [ref.lower() for ref in references]
            gen = gen.lower()
        scores = scorer.score_multi(references, gen)

        rougeL_p_scores.append(scores['rougeL'][P_INDEX])
        rougeL_r_scores.append(scores['rougeL'][R_INDEX])
        rougeL_f_scores.append(scores['rougeL'][F_INDEX])

    return {"r": round(mean(rougeL_r_scores),5), "p": round(mean(rougeL_p_scores),5), "f": round(mean(rougeL_f_scores),5)}

def get_corpus_bleu(references, predictions, lower_case=False, language="english"):
    list_of_references = []
    hypotheses = []

    for question_refs in references:
        tmp_list_of_refs = []
        for ref in question_refs:
            ref_processed = word_tokenize(ref, language=language) # tokenize
            if lower_case:
                ref_processed = [each_string.lower() for each_string in ref_processed] # lowercase
            tmp_list_of_refs.extend([ref_processed])
        list_of_references.append(tmp_list_of_refs)

    for pred in predictions:
        pred_processed = word_tokenize(pred, language=language) # tokenize
        if lower_case:
            pred_processed = [each_string.lower() for each_string in pred_processed] # lowercase
        hypotheses.append(pred_processed)

    bleu_1 = corpus_bleu(list_of_references, hypotheses, weights = [1,0,0,0])
    bleu_2 = corpus_bleu(list_of_references, hypotheses, weights = [0.5,0.5,0,0])
    bleu_3 = corpus_bleu(list_of_references, hypotheses, weights = [1/3,1/3,1/3,0])
    bleu_4 = corpus_bleu(list_of_references, hypotheses, weights = [0.25,0.25,0.25,0.25])

    return {"Bleu_1": round(bleu_1,5), "Bleu_2": round(bleu_2,5), "Bleu_3": round(bleu_3,5), "Bleu_4": round(bleu_4,5)}

def get_bert_score_old(references, predictions, lower_case=False, language="english"):
    list_of_references = []

    for index, ref_group in enumerate(references):
        refs_bertscore = []
        candidate = predictions[index]
        for ref in ref_group:
            (P, R, F), hashname = score([candidate], [ref], lang="en", return_hash=True)
            bertscore_f1 = F.item()
            refs_bertscore.append(bertscore_f1)
        best_bertscore_f1 = max(refs_bertscore)
        idx_best_bertscore_f1 = refs_bertscore.index(best_bertscore_f1)
        chosen_ref = refs_bertscore[idx_best_bertscore_f1]
        list_of_references.append(chosen_ref)

    (P_all, R_all, F_all), hashname = score(predictions, list_of_references, lang="en", return_hash=True)

    return F_all.item()

def get_bert_score(references, predictions, lower_case=False, language="english"):
    (P, R, F), hashname = score(predictions, references, lang="en", return_hash=True)
    all_bert_scores_f1 = F.tolist()

    mean_all_bert_score_f1 = round(mean(all_bert_scores_f1),4)

    return mean_all_bert_score_f1

def get_bleurt_score(references, predictions, lower_case=False, language="english"):
    bleurt_checkpoint = "PATH_TO_BLEURT/bleurt-master/bleurt/BLEURT-20" #update this

    list_of_references = []
    scorer = score.BleurtScorer(bleurt_checkpoint)

    for index, ref_group in enumerate(references):
        refs_bertscore = []
        candidate = predictions[index]
        for ref in ref_group:

            scores = scorer.score(references=[ref], candidates=[candidate])
            assert isinstance(scores, list) and len(scores) == 1
            refs_bertscore.append(scores[0])
        #print(index, refs_bertscore)
        best_bertscore_f1 = max(refs_bertscore)
        idx_best_bertscore_f1 = refs_bertscore.index(best_bertscore_f1)
        chosen_ref = ref_group[idx_best_bertscore_f1]
        list_of_references.append(chosen_ref)
    

    all_scores = scorer.score(references=list_of_references, candidates=predictions)
    #with open('bleurt_answertype-text_question-answer.json', 'w') as f:
        #json.dump(all_scores, f)

    return mean(all_scores)

def get_ppl_score(predictions, lower_case=False, language="english"):
    perplexity = evaluate.load("perplexity", module_type="metric")
    results_perplexity = perplexity.compute(model_id='gpt2', add_start_token=False, predictions=predictions)
    return results_perplexity

def calc_distinct_n_score(text, n):
    # Split the text into individual words
    words = text.split()

    # Calculate n-grams
    ngrams_list = list(ngrams(words, n))

    # Count the occurrences of each n-gram
    ngrams_count = Counter(ngrams_list)

    # Calculate the number of distinct n-grams
    distinct_ngrams = len(ngrams_count)

    # Calculate the distinct-n score
    distinct_n_score = distinct_ngrams / len(words)

    return distinct_n_score

def get_distinct_n_score(predictions, n):
    all_distinct_scores = []

    for pred in predictions:
        distinct_n_score = calc_distinct_n_score(pred, n)
        all_distinct_scores.append(distinct_n_score)

    return mean(all_distinct_scores)

def get_grammar_error_score(predictions):
    tool = LanguageTool('en-US')  # Specify the language ('en-US' for American English)

    total_errors = 0
    for sentence in predictions:
        matches = tool.check(sentence)  # Check for errors in the sentence
        filtered_matches = [match for match in matches if match.ruleId != 'MORFOLOGIK_RULE_EN_US']
        total_errors += len(filtered_matches)  # Count the number of errors in the sentence

        #if len(filtered_matches) > 0:
            #print(filtered_matches)
            #print("\n")

    average_errors = total_errors / len(predictions)

    return average_errors

def run(args, preds_path):

    for pred_path in preds_path:

        print(pred_path)

        # Read predictions file
        with open(pred_path + "predictions.json") as file:
            predictions = json.load(file)
        
        references = [ref['questions_reference'] for ref in predictions]
        predictions_answers = [pred['gen_answer'] for pred in predictions]
        predictions = [pred['gen_question'] for pred in predictions]

        references_flat = list(itertools.chain.from_iterable(references))

        # Get BLEU (results are the same as reported from Du et. al (2017))
        score_corpus_bleu = get_corpus_bleu(references, predictions, lower_case=False, language=args.language)
        print("Score Corpus Bleu: ", round(score_corpus_bleu['Bleu_4'],3))

        rouge_scores =  get_rouge_option_rouge_scorer(references, predictions, lower_case=True, language=args.language)
        print("Mean rouge_scorer: ", round(rouge_scores['f'],3))

        #bert_score = get_bert_score(references, predictions, lower_case=True, language=args.language)
        #print("Mean BERT_score: ", bert_score)

        #bleurt_score = get_bleurt_score(references, predictions, lower_case=True, language=args.language)
        #print("Mean BLEURT_score: ", bleurt_score)

        ppl_score_references = get_ppl_score(references_flat, lower_case=False, language=args.language)
        print("Mean Perplexity score (references): ", ppl_score_references["mean_perplexity"])

        #
        # Linguistic Quality for QUESTIONS #
        #

        #ppl_score_predictions = get_ppl_score(predictions, lower_case=False, language=args.language)
        #print("Mean Perplexity score (predictions): ", round(ppl_score_predictions["mean_perplexity"],3))

        #distinct_score = get_distinct_n_score(predictions, 3)
        #print("Mean Distinct-3 score: ", round(distinct_score,3))

        #grammar_error_score_preds = get_grammar_error_score(predictions)
        #print("Avg. Grammar Error score (predictions): ", round(grammar_error_score_preds,3))

        #
        # Linguistic Quality for ANSWERS #
        #

        ppl_score_predictions = get_ppl_score(predictions_answers, lower_case=False, language=args.language)
        print("Mean Perplexity score (predictions): ", round(ppl_score_predictions["mean_perplexity"],3))

        distinct_score = get_distinct_n_score(predictions_answers, 3)
        print("Mean Distinct-3 score: ", round(distinct_score,3))

        grammar_error_score_preds = get_grammar_error_score(predictions_answers)
        print("Avg. Grammar Error score (predictions): ", round(grammar_error_score_preds,3))

        print("\n\n")

if __name__ == '__main__':
    
    preds_path = [
    #"../predictions/t5-base/qg_t5_base_512_128_32_10_answer-text_question_seed_44/model-epoch=01-val_loss=1.05/",
    #"../predictions/t5-base/qg_t5_base_512_128_32_10_text_question-answer_seed_44/model-epoch=03-val_loss=1.02/",
    #"../predictions/t5-base/qg_t5_base_512_128_32_10_answertype-text_question-answer_seed_44/model-epoch=04-val_loss=0.99/",
    #"../predictions/t5-base/qg_t5_base_512_128_32_10_skill-text_question-answer_seed_44/model-epoch=04-val_loss=0.95/",
    #"../predictions/t5-base/qg_t5_base_512_128_32_10_skill-answertype-text_question-answer_seed_44/model-epoch=04-val_loss=0.95/",
    
    #"../predictions/t5-large/qg_t5_large_512_128_8_10_answer-text_question_seed_45/model-epoch=00-val_loss=0.94/",
    "../predictions/t5-large/qg_t5_large_512_128_8_10_text_question-answer_seed_45/model-epoch=01-val_loss=0.92/",
    "../predictions/t5-large/qg_t5_large_512_128_8_10_answertype-text_question-answer_seed_45/model-epoch=01-val_loss=0.91/",
    "../predictions/t5-large/qg_t5_large_512_128_8_10_skill-text_question-answer_seed_45/model-epoch=00-val_loss=0.85/",
    "../predictions/t5-large/qg_t5_large_512_128_8_10_skill-answertype-text_question-answer_seed_45/model-epoch=01-val_loss=0.86/",

    "../predictions/gpt3/gpt3_ctrl_text/",
    "../predictions/gpt3/gpt3_ctrl_a_same/",
    "../predictions/gpt3/gpt3_ctrl_sk/",
    "../predictions/gpt3/gpt3_ctrl_sk_a/"

    ]

    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Evaluation script for QG.')

    # Add arguments
    #parser.add_argument('-pp','--predictions_path', type=str, metavar='', default="../predictions/qg_t5_base_512_128_32_10_skill-answertype-text_question-answer_seed_44/model-epoch=04-val_loss=0.95/", required=False, help='Predictions path.')
    parser.add_argument('-lg','--language', type=str, metavar='', default="english", required=False, help='Language for tokenize.')

    # Parse arguments
    args = parser.parse_args()

    # Start evaluation
    run(args, preds_path)