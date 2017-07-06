import nltk
import random
import string
from nltk.corpus import movie_reviews, stopwords
import pickle

stop_words = set(stopwords.words("english"))

documents = [(list(movie_reviews.words(fileid)), category)
			  for category in movie_reviews.categories()
			  for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = [w for w in all_words if w not in stop_words]
all_words = [''.join(c for c in s if c not in string.punctuation) for s in all_words]
all_words = [s for s in all_words if s]

all_words = nltk.FreqDist(all_words)

word_features = [w[0] for w in list(all_words.most_common(3000))]

#print(word_features)

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features


featureSets = [(find_features(rev), category) for (rev, category) in  documents]

training_set = featureSets[:1900]
testing_set = featureSets[1900:]


classifer = nltk.NaiveBayesClassifier.train(list(training_set))

#classifer_f = open("naivebayes.pickle", "rb")
#classifer = pickle.load(classifer_f)
#classifer_f.close()

print("Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifer, testing_set))*100)

classifer.show_most_informative_features(15)

#save_classifier = open("naivebayes.pickle", "wb")
#pickle.dump(classifer, save_classifier)
#save_classifier.close()

