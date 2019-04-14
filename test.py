import csv
import logging

from analyzer import get_stop_words, training_set, naive_bayes, max_entropy_classifier, get_featured_labels, svm_classifier

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.getLogger(__name__)

if __name__ == "__main__":
    stop_words = get_stop_words('../datasets/feature_list/stopwordsID.txt')
    training_set, sentences = training_set('../datasets/sampleTweetsID.csv', stop_words)

    # tweet_sample = 'Hari yang mengecewakan. Menghadiri pameran mobil untuk mencari pendanaan, harganya malah lebih mahal'
    # tweet_sample = 'apakah kamu membagikan #lelucon #kutipan #musik #foto atau #berita #artikel '
    tweet_sample = 'Dear #Telkomel kapan daerah kami akan mendapatkan jaringan yang bagus seperti daerah lain. #telkomsel #miris'

    sentiment = naive_bayes(tweet_sample, training_set, stop_words)
    logging.info("Naive bayes | %s | %s" % (tweet_sample, sentiment))

    sentiment = max_entropy_classifier(tweet_sample, training_set, stop_words)
    logging.info("Max entropy | %s | %s" % (tweet_sample, sentiment))

    featured_labels = get_featured_labels(sentences)
    sentiment = svm_classifier(featured_labels, training_set, tweet_sample, stop_words)
    logging.info("SVM | %s | %s" % (tweet_sample, sentiment))
