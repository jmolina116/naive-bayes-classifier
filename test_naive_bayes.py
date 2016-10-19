# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from naive_bayes import NaiveBayes

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip

from nltk.tokenize import word_tokenize as tokenize
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords
import string

class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return [self.data % 2 == 0]

class BagOfWords(Document):
    def features(self):
        """Ρemoving stopwords worsens results.
        Lemmatizing worsens results.
        Trigrams worsen results."""

        # tokenizing and removing punctuation slightly improves results
        puncs = [p for p in string.punctuation]
        words = [w for w in tokenize(self.data.lower()) if w not in puncs]

        # bigrams significantly improve results
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]

        # length feature slightly improve results
        avg_w_len = 0
        total_w_len = 0
        for word in words:
            total_w_len += len(word)
        if len(words) != 0:
            avg_w_len = total_w_len / len(words)
        length_features = [('length', len(words)), ('avg_w_len', avg_w_len)]

        return words + bigrams + length_features

class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        name = self.data
        # performed best with just these two features and no others
        return [('first', name[0].lower()), ('last', name[-1].lower())]

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        # modified to show 2 decimal places
        acc = 100.0 * sum(correct) / len(correct)
        split_acc = str(acc).split('.')
        s = "accuracy: " + "{}.{:.2}%\n".format(split_acc[0], split_acc[1])
        print >> verbose, s,
    return sum(correct) / len(correct)

def recall(classifier, test, verbose=sys.stderr):
    r = {}
    for label in classifier.model:
        true_pos = [classifier.classify(x) == x.label and x.label == label for x in test]
        false_neg = [classifier.classify(x) != x.label and x.label == label for x in test]
        if sum(true_pos + false_neg) != 0:
            r[label] = sum(true_pos) / sum(true_pos + false_neg)
        else:
            r[label] = 0.0
    if verbose:
        for label in r:
            split_r = str(100.0 * r[label]).split('.')
            s = str(label) + " recall: "
            s += "{}.{:.2}%\n".format(split_r[0], split_r[1])
            print >> verbose, s,
    return r

def precision(classifier, test, verbose=sys.stderr):
    p = {}
    for label in classifier.model:
        true_pos = [classifier.classify(x) == x.label and x.label == label for x in test]
        false_pos = [classifier.classify(x) == label and x.label != label for x in test]
        if sum(true_pos + false_pos) != 0:
            p[label] = sum(true_pos) / sum(true_pos + false_pos)
        else:
            p[label] = 0.0
    if verbose:
        for label in p:
            split_p = str(100.0 * p[label]).split('.')
            s = str(label) + " precision: "
            s += "{}.{:.2}%\n".format(split_p[0], split_p[1])
            print >> verbose, s,
    return p

def f1(classifier, test, verbose=sys.stderr):
    r = recall(classifier, test, verbose=verbose)
    p = precision(classifier, test, verbose=verbose)
    f = {}
    for label in r:
        if (r[label] + p[label]) != 0.0:
            f[label] = (2 * r[label] * p[label]) / (r[label] + p[label])
        else:
            f[label] = 0.0
    if verbose:
        for label in f:
            split_f = str(100.0 * f[label]).split('.')
            s = str(label) + " f1: "
            s += "{}.{:.2}%\n".format(split_f[0], split_f[1])
            print >> verbose, s,
    return f

def avg_f1(classifier, test, verbose=sys.stderr):
    f = f1(classifier, test, verbose=verbose)
    if verbose:
        split_f = str(100.0 * sum(f.values()) / len(f)).split('.')
        s = "avg f1: " + "{}.{:.2}%\n".format(split_f[0], split_f[1])
        print >> verbose, s,
    return sum(f.values()) / len(f)

class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd(self):
        """Classify numbers as even or odd"""
        classifier = NaiveBayes()
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
        self.assertEqual(accuracy(classifier, test), 1.0)
        avg_f1(classifier, test, verbose=False)
        classifier.save('models/even_odd_model')

    def split_names_corpus(self, document_class=Name):
        """Split the names corpus into training and test sets"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943)  # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:6000], names[6000:])

    def test_names_nltk(self):
        """Classify names using NLTK features"""
        train, test = self.split_names_corpus()
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.70)
        avg_f1(classifier, test, verbose=False)
        classifier.save('models/names_model')

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_bag(self):
        """Classify blog authors using bag-of-words"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)
        avg_f1(classifier, test, verbose=False)
        classifier.save('models/blogs_bag_model')

    def split_blogs_corpus_imba(self, document_class):
        blogs = BlogsCorpus(document_class=document_class)
        imba_blogs = blogs.split_imbalance()
        return (imba_blogs[:1600], imba_blogs[1600:])

    def test_blogs_imba(self):
        train, test = self.split_blogs_corpus_imba(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        # you don't need to pass this test
        self.assertGreater(accuracy(classifier, test), 0.1)
        avg_f1(classifier, test, verbose=False)
        classifier.save('models/blogs_imba_model')

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
