import pandas as pd
import spacy
from tqdm import tqdm

"""
before running, make sure you have run:
`python -m spacy download en_core_web_sm`
"""

# load pipeline that doesn't look for named entities or dependency parsing (we just wanna lemmatize)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

df = pd.read_csv("data/airport_clean.csv") # read in clean data

df['content'] = df['content'].astype(str) # enforce string type for review content column

"""
 preprocessing logic:
 	takes in spacy doc object, returns lemmatized string.
 """
def preprocess_text(doc):
	return " ".join([
        token.lemma_.lower() for token in doc # lemmatize and lowercase
		if not token.is_stop # removes stop words
		and not token.is_punct # removes punctuation
		and not token.like_num # removes numbers
		and not token.is_space # removes whitespace
    ])

"""
actually preprocess the text
"""
print("Preprocessing text...")

# applies preprocessing function defined earlier
df['processed_content'] = [
	preprocess_text(doc)
	for doc in tqdm(nlp.pipe(df['content'], batch_size=1000), total=len(df))
]

# save to new csv
df.to_csv("data/step2_preprocessed.csv", index=False)

print("Preprocessing complete. Saved to data/step2_preprocessed.csv")