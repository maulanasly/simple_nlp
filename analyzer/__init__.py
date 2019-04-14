import csv
import nltk.classify
from sklearn import svm

from .sanitizers import clean, get_stop_words
from .feature_vectors import get_feature_vector, get_svm_feature_vector_and_labels, get_svm_feature_vector


FEATURE_LIST = []


def extract_file(filename):
    with open(filename, 'rb') as fn:
        return fn.read().splitlines()


def extract_features(sentence):
    words = set(sentence)
    features = {}
    for word in FEATURE_LIST:
        features['contains(%s)' % word] = (word in words)
    return features


def training_set(filename, stop_words):
    raw_sentences = csv.reader(extract_file(filename), quotechar='|')
    sentences = []
    for text in raw_sentences:
        sentiment, word = text

        cleaned_tweet = clean(word)
        feature_vector = get_feature_vector(cleaned_tweet, stop_words)
        if cleaned_tweet not in FEATURE_LIST:
            FEATURE_LIST.extend(feature_vector)
        sentences.append((feature_vector, sentiment))

    return (nltk.classify.util.apply_features(extract_features, sentences), sentences)


def naive_bayes(sentence, training_set, stop_words):
    # Train the Naive Bayes classifier
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
    cleaned_sentence = clean(sentence)
    return NBClassifier.classify(extract_features(get_feature_vector(cleaned_sentence, stop_words)))


def max_entropy_classifier(sentence, training_set, stop_words):
    # Max Entropy Classifier
    MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set,
                                                                   'GIS',
                                                                   trace=0,
                                                                   encoding=None,
                                                                   labels=None,
                                                                   gaussian_prior_sigma=0,
                                                                   max_iter=10)
    cleaned_sentence = clean(sentence)
    return MaxEntClassifier.classify(extract_features(get_feature_vector(cleaned_sentence, stop_words)))


def get_featured_labels(sentences):
    return get_svm_feature_vector_and_labels(sentences, FEATURE_LIST)


def svm_classifier(featured_labels, training_set, sentence, stop_words):
    sentiment_label = {0: 'positive', 1: 'negative', 2: 'neutral'}
    classifier = svm.SVC(probability=True, gamma='scale')

    featured_vector = get_svm_feature_vector(sentence, FEATURE_LIST, stop_words)

    classifier.fit(featured_labels['feature_vector'], featured_labels['labels'])
    prediction = classifier.predict([featured_vector])
    # print classifier.predict_proba([featured_vector])
    return sentiment_label[prediction[0]]


__all__ = [
    'extract_file', 'extract_features', 'extract_features', 'training_set', 'naive_bayes', 'max_entropy_classifier',
    'get_featured_labels', 'svm_classifier', 'get_stop_words'
]
