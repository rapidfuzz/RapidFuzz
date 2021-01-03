# SPDX-License-Identifier: MIT
# Copyright (C) 2021 Max Bachmann

from rapidfuzz import string_metric
import logging

def distance(s1, s2):
    logger = logging.getLogger(__name__)
    logger.warn(
        'This function is deprecated and will be removed in v2.0.0.\n'
        'Use string_metric.levenshtein(s1, s2) instead')
    return string_metric.levenshtein(s1, s2)

def normalized_distance(s1, s2, score_cutoff=0):
    logger = logging.getLogger(__name__)
    logger.warn(
        'This function is deprecated and will be removed in v2.0.0.\n'
        'Use string_metric.normalized_levenshtein(s1, s2, score_cutoff=%f) instead' % score_cutoff)
    return string_metric.normalized_levenshtein(s1, s2, score_cutoff=score_cutoff)

def weighted_distance(s1, s2, insert_cost=1, delete_cost=1, replace_cost=1):
    logger = logging.getLogger(__name__)
    logger.warn(
        'This function is deprecated and will be removed in v2.0.0.\n'
        'Use string_metric.normalized_levenshtein(s1, s2, insert_cost=%d, delete_cost=%d, replace_cost=%d) instead'
        % (insert_cost, delete_cost, replace_cost))
    return string_metric.levenshtein(s1, s2, insert_cost, delete_cost, replace_cost)

def normalized_weighted_distance(s1, s2, score_cutoff=0):
    logger = logging.getLogger(__name__)
    logger.warn(
        'This function is deprecated and will be removed in v2.0.0.\n'
        'Use string_metric.normalized_levenshtein(s1, s2, insert_cost=1, delete_cost=1, replace_cost=2, score_cutoff=%f) instead'
        %  score_cutoff)
    return string_metric.normalized_levenshtein(s1, s2, insert_cost=1, delete_cost=1, replace_cost=2, score_cutoff=score_cutoff)

def hamming(s1, s2):
    logger = logging.getLogger(__name__)
    logger.warn(
        'This function is deprecated and will be removed in v2.0.0.\n'
        'Use string_metric,hamming(s1, s2) instead')
    return string_metric,hamming(s1, s2)