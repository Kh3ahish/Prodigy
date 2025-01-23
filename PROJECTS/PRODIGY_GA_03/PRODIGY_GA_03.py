
import markovify
import pandas as pd

# Load dataset
df = pd.read_csv('/content/airport_reviews.csv.zip')

# Join reviews into a single large text string and build a Markov chain model
from itertools import chain

N = 100
review_subset = df["content"][:N]
text = "".join(chain.from_iterable(review_subset))
markov_chain_model = markovify.Text(text)

# Generate 5 sentences using the Markov chain model
print("Generated sentences:")
for _ in range(5):
    print(markov_chain_model.make_sentence())

# Generate 3 sentences with a length of no more than 140 characters
print("\nGenerated short sentences:")
for _ in range(3):
    print(markov_chain_model.make_short_sentence(140))
