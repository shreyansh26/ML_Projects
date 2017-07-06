# I was taking a ride in the car
# I was riding in the car

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

#for w in example_words:
#	print(ps.stem(w))

new_text = "It is ver important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly atleast once."

words = word_tokenize(new_text)

for w in words:
	print(ps.stem(w))