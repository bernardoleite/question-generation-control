"""
Description: This file contains the implementation of the PINC metric. The script is not our own work, but is taken from the following repository:
Reference: https://github.com/kstats/MultiQuestionGeneration/blob/main/metrics/metric_diversity.py
"""

import re

def expand_contractions(text, contractions_dict):
    contractions_pattern = re.compile(
        "({})".format("|".join(contractions_dict.keys())), flags=re.IGNORECASE | re.DOTALL
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contractions_dict.get(match) if contractions_dict.get(match) else contractions_dict.get(match.lower())
        )
        expanded_contraction = expanded_contraction
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I had",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "iit will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}




import string
from nltk import ngrams


class PINCscore:
    def __init__(self, max_n_gram):
        self.max_n_gram = max_n_gram

    def ngram(self, document, max_n_gram):
        ngrams_list = []
        for i in range(1, max_n_gram + 1):
            splitted = ngrams(document.split(), i)
            ngrams_list.append(set(splitted))
        return ngrams_list

    def preprocess(self, text):
        # helper funfction for preprocessing text
        pre_processed_text = []
        for i in range(len(text)):
            expanded_text = (expand_contractions(text[i], contractions_dict)).lower()
            no_punc_text = expanded_text.translate(str.maketrans("", "", string.punctuation))
            pre_processed_text.append(no_punc_text)
        return pre_processed_text

    def score(self, contexts, questions, answers, lengths=None, extra=None):
        # Rest of the code remains the same
        # You can simply add a placeholder for the 'answers' parameter here
        # It's not used in this context
        pass

    def score(self, contexts, questions, lengths=None, extra=None):
        """
        The score function returns the PINC score for two documents.
        With a maximum_lengths constraint, the function tokenizes the two
        document and measure the level of similarity  between them.
        The original implementation can be found here:
        https://www.cs.utexas.edu/~ml/papers/chen.acl11.pdf
        """
        pre_processed_contexts = self.preprocess(contexts)
        pre_processed_questions = self.preprocess(questions)

        PINC_score_list = []
        for i in range(len(pre_processed_questions)):
            # the N in the N-gram tokenization cannot exceed the number of words in the document
            max_n_gram = min(
                len(pre_processed_questions[i].split()), len(pre_processed_contexts[i].split()), self.max_n_gram
            )

            # if question is blank, then score is 0
            if max_n_gram == 0:
                PINC_score_list.append(0)
                continue

            context_ngram_list = self.ngram(pre_processed_contexts[i], max_n_gram)
            question_ngram_list = self.ngram(pre_processed_questions[i], max_n_gram)
            # we tokenize the groundtruth document and the prediction sentences
            # and create a 1-D array which contains all the n grams, where n ranges
            # from 1 to N
            PINC_score = 0
            for j in range(max_n_gram):
                overlap_count = 0
                for elem in question_ngram_list[j]:
                    if elem in context_ngram_list[j]:
                        overlap_count += 1
                PINC_score += 1 - overlap_count / len(question_ngram_list[j])
            PINC_score *= 1 / max_n_gram
            PINC_score_list.append(PINC_score)
        return PINC_score_list

    def score_two_questions(self, question_ones, question_twos, lengths=None, extra=None):
        """
        The PINC scoring function specifically for two question generation.
        Instead of evaluating the level of similarity betweena context and the
        generated questions. This function instead evaluates the level of similarity
        between the two sets of generated functions
        """
        assert len(question_ones) == len(question_twos), "The number of questions must be equal"
        pre_processed_first_questions = self.preprocess(question_ones)
        pre_processed_second_questions = self.preprocess(question_twos)

        PINC_score_list = []
        for i in range(len(pre_processed_second_questions)):
            # the N in the N-gram tokenization cannot exceed the number of words in the document
            max_n_gram = min(
                len(pre_processed_second_questions[i].split()),
                len(pre_processed_first_questions[i].split()),
                self.max_n_gram,
            )

            # if question is blank, then score is 0
            if max_n_gram == 0:
                PINC_score_list.append(0)
                continue

            question_ones_ngram_list = self.ngram(pre_processed_first_questions[i], max_n_gram)
            question_twos_ngram_list = self.ngram(pre_processed_second_questions[i], max_n_gram)
            # we tokenize the groundtruth document and the prediction sentences
            # and create a 1-D array which contains all the n grams, where n ranges
            # from 1 to N
            PINC_score = 0
            # Question2 -> Question 1 PINC score
            PINC_score_reverse = 0
            # Question1 -> Question 2 PINC score
            for j in range(max_n_gram):
                overlap_count = 0
                overlap_count_reverse = 0
                for elem in question_twos_ngram_list[j]:
                    if elem in question_ones_ngram_list[j]:
                        overlap_count += 1
                for elem in question_ones_ngram_list[j]:
                    if elem in question_twos_ngram_list[j]:
                        overlap_count_reverse += 1
                PINC_score += 1 - overlap_count / len(question_twos_ngram_list[j])
                PINC_score_reverse += 1 - overlap_count_reverse / len(question_ones_ngram_list[j])
            PINC_score *= 1 / max_n_gram
            PINC_score_reverse *= 1 / max_n_gram
            PINC_score_list.append((PINC_score + PINC_score_reverse) / 2)
        return PINC_score_list