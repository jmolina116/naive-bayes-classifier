# -*- mode: Python; coding: utf-8 -*-

from __future__ import division
from classifier import Classifier
from numpy import log
from collections import defaultdict as dd

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""

    def __init__(self, model=dd()):
        super(NaiveBayes, self).__init__(model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, instances):
        probs = dd()  # joint probabilities
        features = set()  # set of all the features found

        # get counts
        for instance in instances:
            label = instance.label
            """There is one blank-labeled blank-data document in the balanced
            corpus that skews F1 calculations away from the simple M/F"""
            if label != '':
                if label not in probs:
                    probs[label] = dd(float)
                for feature in instance.features():
                    features.add(feature)
                    probs[label][feature] += 1.0

        # do smoothing
        smooth = 0.1
        for label in probs:
            probs[label]['UNK'] = smooth
            for feature in features:
                probs[label][feature] += smooth

        # convert to log probabilities
        sum = 0.0
        for label in probs:
            for feature in probs[label]:
                sum += probs[label][feature]
        for label in probs:
            for feature in probs[label]:
                probs[label][feature] /= sum
                probs[label][feature] = log(probs[label][feature])

        self.model = probs

    def classify(self, instance):
        probs = self.model
        p_lf = dd(float)  # probabilities of labels given features
        for label in probs:
            for feature in instance.features():
                if feature in probs[label]:
                    p_lf[label] += probs[label][feature]
                else:
                    p_lf[label] += probs[label]['UNK']
        return max(p_lf, key=lambda x: p_lf[x])
