import nltk
import random
import string
from nltk.corpus import movie_reviews, stopwords
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode 


# Own Classifer
class VoteClassifer(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes =  votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf


short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

documents = []

for r in short_pos.split('\n'):
	documents.append( (r, "pos") )

for r in short_neg.split('\n'):
	documents.append( (r, "neg") )

print(documents[:5])

all_words = []
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
	all_words.append(w.lower())

for w in short_neg_words:
	all_words.append(w.lower())

print(all_words[:10])

all_words = nltk.FreqDist(all_words)

#word_features = [w[0] for w in list(all_words.most_common(5000))]
word_features = list(all_words.keys())[:5000]

#print(word_features)

def find_features(document):
	words = set(word_tokenize(document))
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features


featureSets = [(find_features(rev), category) for (rev, category) in  documents]

random.shuffle(featureSets)

training_set = featureSets[:10000]
testing_set = featureSets[10000:]


classifer = nltk.NaiveBayesClassifier.train(training_set)

#classifer_f = open("naivebayes.pickle", "rb")
#classifer = pickle.load(classifer_f)
#classifer_f.close()

print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifer, testing_set))*100)

classifer.show_most_informative_features(15)

#save_classifier = open("naivebayes.pickle", "wb")
#pickle.dump(classifer, save_classifier)
#save_classifier.close()

MNB_Classifier = SklearnClassifier(MultinomialNB())
MNB_Classifier.train(list(training_set))
print("MNB_Classifier accuracy percent: ", (nltk.classify.accuracy(MNB_Classifier, testing_set))*100)

'''
GNB_Classifier = SklearnClassifier(GaussianNB())
GNB_Classifier.train(list(training_set))
print("GNB_Classifier accuracy percent: ", (nltk.classify.accuracy(GNB_Classifier, testing_set))*100)
'''

BNB_Classifier = SklearnClassifier(BernoulliNB())
BNB_Classifier.train(list(training_set))
print("BNB_Classifier accuracy percent: ", (nltk.classify.accuracy(BNB_Classifier, testing_set))*100)

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(list(training_set))
print("LogisticRegression_Classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_Classifier, testing_set))*100)

SGD_Classifier = SklearnClassifier(SGDClassifier())
SGD_Classifier.train(list(training_set))
print("SGD_Classifier accuracy percent: ", (nltk.classify.accuracy(SGD_Classifier, testing_set))*100)

# Leave SVC since it gives poor results as compared to others
'''
SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(list(training_set))
print("SVC_Classifier accuracy percent: ", (nltk.classify.accuracy(SVC_Classifier, testing_set))*100)
'''
LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(list(training_set))
print("LinearSVC_Classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_Classifier, testing_set))*100)

NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(list(training_set))
print("NuSVC_Classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_Classifier, testing_set))*100)


voted_classifier = VoteClassifer(classifer, MNB_Classifier, BNB_Classifier, LogisticRegression_Classifier, SGD_Classifier, LinearSVC_Classifier, NuSVC_Classifier)

print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %: ", voted_classifier.confidence(testing_set[1][0])*100)

print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %: ", voted_classifier.confidence(testing_set[2][0])*100)

print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %: ", voted_classifier.confidence(testing_set[3][0])*100)

print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %: ", voted_classifier.confidence(testing_set[4][0])*100)

print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %: ", voted_classifier.confidence(testing_set[5][0])*100)
