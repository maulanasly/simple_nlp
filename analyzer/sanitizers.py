import re


def clean(sentence):
    # process the sentence
    # Convert to lower case
    sentence = sentence.lower()
    # Convert www.* or https?://* to URL
    sentence = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'URL', sentence)
    # Convert @username to AT_USER
    sentence = re.sub(r'@[^\s]+', 'AT_USER', sentence)
    # Remove additional white spaces
    sentence = re.sub(r'[\s]+', ' ', sentence)
    # Replace #word with word
    sentence = re.sub(r'#([^\s]+)', r'\1', sentence)
    # trim
    sentence = sentence.strip('\'"')
    return sentence


def remove_duplications(word):
    # look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", word)


def get_stop_words(file_name):
    stop_words = []
    with open(file_name, 'r') as fn:
        stop_words = fn.read().splitlines()
    stop_words.extend(['AT_USER', 'URL'])
    return stop_words
