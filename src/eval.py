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
from rouge_score import rouge_scorer, scoring
from statistics import mean

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
    
    #list_of_references = [['ref1a bla bla bla bla bla', 'ref1b bla bla bla bla bla'], ['ref2a bla bla bla bla bla', 'ref2b bla bla bla bla bla']]
    #hypotheses = ['gen 1 bla bla bla bla bla bla', 'gen 2 bla bla bla bla bla bla']

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

    return {"r": mean(rougeL_r_scores), "p": mean(rougeL_p_scores), "f": mean(rougeL_f_scores)}

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

    return {"Bleu_1": bleu_1, "Bleu_2": bleu_2, "Bleu_3": bleu_3, "Bleu_4": bleu_4}

def run(args):
    # Read predictions file
    with open(args.predictions_path + "predictions.json") as file:
        predictions = json.load(file)
    
    references = [ref['questions_reference'] for ref in predictions]
    predictions = [pred['gen_question'] for pred in predictions]

    # Get BLEU (results are the same as reported from Du et. al (2017))
    score_corpus_bleu = get_corpus_bleu(references, predictions, lower_case=False, language=args.language)
    print("Score Corpus Bleu: ", score_corpus_bleu)

    rouge_scores =  get_rouge_option_rouge_scorer(references, predictions, lower_case=True, language=args.language)
    print("Mean rouge_scorer: ", rouge_scores)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Generate questions and save them to json file.')

    # Add arguments
    parser.add_argument('-pp','--predictions_path', type=str, metavar='', default="../predictions/qg_t5_small_512_64_8_10_skilltext_questionanswer_precl_random_dist3_seed_42/model-epoch=04-val_loss=1.12/", required=False, help='Predictions path.')
    parser.add_argument('-lg','--language', type=str, metavar='', default="english", required=False, help='Language for tokenize.')

    # Parse arguments
    args = parser.parse_args()

    # Start evaluation
    run(args)