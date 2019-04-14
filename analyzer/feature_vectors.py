import re

from .sanitizers import remove_duplications, clean


def get_feature_vector(sentence, stop_words):
    feature_vector = []
    for word in sentence.split():
        word = remove_duplications(word)
        word = word.strip('\'"?,.')  # cleanup punctution
        # check if the word stats with an alphabet
        if re.search(r'^[a-zA-Z][a-zA-Z0-9]*$', word) is None or word in stop_words:
            continue

        feature_vector.append(word.lower())
    return feature_vector


def get_svm_feature_vector_and_labels(sentences, feature_list):
    sorted_feature = sorted(feature_list)
    feature_vector = []
    labels = []
    sentiment_label = {
        'positive': 0,
        'negative': 1,
        'neutral': 2
    }
    for sentence in sentences:
        mapping = {w: 0 for w in sorted_feature}
        text, sentiment = sentence
        for word in text:
            word = remove_duplications(word)
            word = word.strip('\'"?,.')
            if word in mapping:
                mapping[word] = 1

        feature_vector.append(mapping.values())
        labels.append(sentiment_label[sentiment])
    return {'feature_vector': feature_vector, 'labels': labels}


def get_svm_feature_vector(sentence, feature_list, stop_words):
    cleaned_sentence = clean(sentence)
    sentence = get_feature_vector(cleaned_sentence, stop_words)
    sorted_feature = sorted(feature_list)
    mapping = {w: 0 for w in sorted_feature}

    for word in sentence:
        if word in mapping:
            mapping[word] = 1
        values = mapping.values()
    return values
