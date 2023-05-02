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

from evaluate import load

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

    return {"Bleu_1": round(bleu_1,4), "Bleu_2": round(bleu_2,4), "Bleu_3": round(bleu_3,4), "Bleu_4": round(bleu_4,4)}

def run(args):
    # Read predictions file
    with open(args.predictions_path + "predictions.json") as file:
        predictions = json.load(file)
    
    # Main change for QA evaluation
    answers_reference_all = [ref['gen_answer'] for ref in predictions]
    answers_generated_all = [pred['qa_answer'] for pred in predictions]

    # https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    answers_reference_all_flat = [item for sublist in answers_reference_all for item in sublist] # for exact_match

    # https://stackoverflow.com/questions/29051573/python-filter-list-of-dictionaries-based-on-key-value
    predictions_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions))
    predictions_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions))

    answers_reference_explicit = [ref['gen_answer'] for ref in predictions_explicit]
    answers_generated_explicit = [pred['qa_answer'] for pred in predictions_explicit]
    answers_reference_implicit = [ref['gen_answer'] for ref in predictions_implicit]
    answers_generated_implicit = [pred['qa_answer'] for pred in predictions_implicit]

    # https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    answers_reference_explicit_flat = [item for sublist in answers_reference_explicit for item in sublist] # for exact_match
    answers_reference_implicit_flat = [item for sublist in answers_reference_implicit for item in sublist] # for exact_match

    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=answers_generated_all, references=answers_reference_all_flat)
    print("EM (ALL): ", round(results["exact_match"], 3))
    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=answers_generated_explicit, references=answers_reference_explicit_flat)
    print("EM (EXPLICIT): ", round(results["exact_match"], 4))
    results = exact_match_metric.compute(predictions=answers_generated_implicit, references=answers_reference_implicit_flat)
    print("EM (IMPLICIT): ", round(results["exact_match"], 3))

    #score_corpus_bleu = get_corpus_bleu(answers_reference_all, answers_generated_all, lower_case=False, language=args.language)
    #print("Score Corpus Bleu (ALL): ", score_corpus_bleu)
    rouge_scores =  get_rouge_option_rouge_scorer(answers_reference_all, answers_generated_all, lower_case=True, language=args.language)
    print("Mean rouge_scorer (ALL): ", rouge_scores)

    #score_corpus_bleu = get_corpus_bleu(answers_reference_explicit, answers_generated_explicit, lower_case=False, language=args.language)
    #print("Score Corpus Bleu (EXPLICIT): ", score_corpus_bleu)
    rouge_scores =  get_rouge_option_rouge_scorer(answers_reference_explicit, answers_generated_explicit, lower_case=True, language=args.language)
    print("Mean rouge_scorer (EXPLICIT): ", rouge_scores)

    #score_corpus_bleu = get_corpus_bleu(answers_reference_implicit, answers_generated_implicit, lower_case=False, language=args.language)
    #print("Score Corpus Bleu (IMPLICIT): ", score_corpus_bleu)
    rouge_scores =  get_rouge_option_rouge_scorer(answers_reference_implicit, answers_generated_implicit, lower_case=True, language=args.language)
    print("Mean rouge_scorer (IMPLICIT): ", rouge_scores)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Evaluation script fo QA.')

    # Add arguments
    parser.add_argument('-pp','--predictions_path', type=str, metavar='', default="../predictions/qa_questiongen_t5_base_512_128_32_10_answertype-text_question-answer_seed_44/model-epoch=04-val_loss=0.99/", required=False, help='Predictions path.')
    parser.add_argument('-lg','--language', type=str, metavar='', default="english", required=False, help='Language for tokenize.')

    # Parse arguments
    args = parser.parse_args()

    # Start evaluation
    run(args)