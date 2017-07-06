import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)

			namedEnt = nltk.ne_chunk(tagged)  #(nltk.ne_chunk(tagged, binary=True))

			namedEnt.draw()

	except Exception as e:
		print(str(e))

process_content()	

"""
NE Type Examples

ORGANISATION Georgia-Tech, WHO
PERSON       Shreyansh
LOCATION     Paris, India
DATE         June, 22-10-16
TIME         two fifty am, 2:50 pm
MONEY        17 usd, 1200 million candian dollars
PERCENT      twenty pct, 18.75%
FACILITY     Washington Monument, Stonehenge
GPE (Geographical)     South East Asia
"""
