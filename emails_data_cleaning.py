import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# raw emails csv
url = 'https://raw.githubusercontent.com/cmansoo/email-spam/main/emails.csv'
df = pd.read_csv(url)

# text length
length = [len(df['text'].iloc[i]) for i in range(len(df))]
df['length'] = length

# word counts
word_count = df['text'].str.split().apply(len) # returns pd.Series obj
df['word_count'] = word_count

# stopwords
mystopwords = stopwords.words('english')

for i in string.punctuation:
    mystopwords.append(i)

# tokenize (or basically text.split()) & remove stopwords
clean_text = []
for i in range(len(df)):
    text = word_tokenize(df['text'].iloc[i])
    processed = [word for word in text if word.lower() not in mystopwords]
    result = ' '.join(processed)
    clean_text.append(result)

df['clean_text'] = clean_text

# export to csv
df.to_csv('clean.csv', index=False)
