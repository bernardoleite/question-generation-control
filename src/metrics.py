"""
Description: This file is part of the implementation of the Self-Bleu metric. The script is not our own work, but is taken from the following repository:
Reference: https://github.com/geek-ai/Texygen/blob/3104e22ac75f3cc2070da2bf5e2da6d2bef149ad/utils/metrics/Metrics.py
"""

from abc import abstractmethod


class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass