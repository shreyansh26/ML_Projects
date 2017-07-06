import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

'''
POS tag list:

CC 	  Coordinating conjunction
CD 	  Cardinal number
DT 	  Determiner
EX 	  Existential there
FW 	  Foreign word
IN 	  Preposition or subordinating conjunction
JJ 	  Adjective
JJR   Adjective, comparative
JJS   Adjective, superlative
LS 	  List item marker
MD 	  Modal
N 	  Noun, singular or mass
NS 	  Noun, plural
NNP   Proper noun, singular
NNPS  Proper noun, plural
PDT   Predeterminer
POS   Possessive ending
PRP   Personal pronoun
PRP$  Possessive pronoun
RB 	  Adverb
RBR   Adverb, comparative
RBS   Adverb, superlative
RP 	  Particle
SYM   Symbol
TO 	  to
UH 	  Interjection
VB 	  Verb, base form
VBD   Verb, past tense
VBG   Verb, gerund or present participle
VBN   Verb, past participle
VBP   Verb, non-3rd person singular present
VBZ   Verb, 3rd person singular present
WDT   Wh-determiner
P 	  Wh-pronoun
WP$   Possessive wh-pronoun
WRB   Wh-adverb 
'''

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)

			print(tagged)

	except Exception as e:
		print(str(e))

process_content()

