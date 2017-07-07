import nltk
import random
import string
from nltk.corpus import movie_reviews, stopwords
import pickle
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
