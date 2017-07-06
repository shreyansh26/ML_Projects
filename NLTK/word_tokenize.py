from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. Ths sky is pinkish blue. You should not eat cardboard."

print(sent_tokenize(example_text))
print(word_tokenize(example_text))