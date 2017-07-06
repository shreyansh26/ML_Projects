from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

'''
print(lemmatizer.lemmatize("cats")) #cat
print(lemmatizer.lemmatize("cacti")) #cactus
print(lemmatizer.lemmatize("geese")) #goose
print(lemmatizer.lemmatize("rocks")) #rock
print(lemmatizer.lemmatize("python")) #python
'''
print(lemmatizer.lemmatize("better")) #better
print(lemmatizer.lemmatize("better", pos="a")) # Since adjective, good (pos is "part of speech")
print(lemmatizer.lemmatize("best", pos="a")) # Since adjective, good
print(lemmatizer.lemmatize("ran")) #ran
print(lemmatizer.lemmatize("ran", pos="v")) # Since verb, run	



