# https://github.com/cytora/pycon-nlp-in-10-lines/blob/master/tutorial_easy.ipynb
import spacy
nlp = spacy.load('en')

doc = nlp('Hello, world. Natural Language Processing in 10 lines of code.')

for token in doc:
    print( token )

for sentence in doc.sents:
    print( sentence )

# part of speech tag
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
for token in doc:
    print( token, "\t\t", token.pos_ )


#
# Write a function that walks up the syntactic tree of the given token and
# collects all tokens to the root token (including root token).
def tokens_to_root(token):
    """
    Walk up the syntactic tree, collecting tokens to the root of the given `token`.
    :param token: Spacy token
    :return: list of Spacy tokens
    """
    tokens_to_r = []
    while token.head is not token:
        tokens_to_r.append(token)
        token = token.head
        
    tokens_to_r.append(token)
    return tokens_to_r

# For every token in document, print it's tokens to the root
for token in doc:
    print('{} --> {}'.format(token, tokens_to_root(token)))

print()
# Print dependency labels of the tokens
for token in doc:
    print('-> '.join(['{}-{}'.format(dependent_token, dependent_token.dep_) for
        dependent_token in tokens_to_root(token)]))


#
# Print all named entities with named entity types
doc_2 = nlp("I went to Paris where I met my old friend Jack from uni.")
for ent in doc_2.ents:
    print('{} - {}'.format(ent, ent.label_))

doc.ents

# Noun chunks are the phrases based upon nouns recovered from tokenized text
# using the speech tags
# Print noun chunks for doc_2
print([chunk for chunk in doc_2.noun_chunks])

for c in doc.noun_chunks:
    print(c)


# For every token in doc_2, print log-probability of the word, estimated from
# counts from a large corpus 
for token in doc_2:
    print(token, ',', token.prob)

doc[4].prob

#
# Word Embedding
doc = nlp("Apples and oranges are similar. Boots and hippos aren't.")

apples = doc[0]
oranges = doc[2]
boots = doc[6]
hippos = doc[8]

print( apples.similarity( oranges) )
print( boots.similarity( hippos ) )

apples_sents, boots_sents = doc.sents

fruit = doc.vocab['fruit']

apples_sents.similarity(fruit)

boots_sents.similarity(fruit)

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    with open(file_name, 'r') as file:
        return file.read()

# Process `text` with Spacy NLP Parser
text = read_file('/tmp/pride_and_prejudice.txt')
processed_text = nlp(text)

sentences = [s for s in processed_text.sents]

len(sentences)

print( sentences[10:14] )


from collections import Counter, defaultdict

def find_actor_occurences(doc):
    actors = Counter()
    for ent in processed_text.ents:
        if ent.label_ == 'PERSON':
            actors[ent.lemma_] += 1
    return actors.most_common()

count = 0
for ent in processed_text.ents:
    if ent.label_ == 'PERSON':
        print(ent.lemma_)
        count+=1
        if count == 40:
            break

print( find_actor_occurences( processed_text )[:20] )

find_actor_occurences( processed_text )[:20] 

find_actor_occurences( processed_text ) 

def get_actors_offsets( doc ):
    actor_offsets = defaultdict( list )
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            actor_offsets[ent.lemma_].append( ent.start )
    return dict( actor_offsets )

actors_occurences = get_actors_offsets( processed_text )

from matplotlib.pyplot import hist
from cycler import cycler

NUM_BINS = 10

def normalize(occurencies, normalization_constant):
    return [o / float(len(processed_text)) for o in occurencies]

def plot_actor_timeseries(actor_offsets, actor_labels, normalization_constant=None):
    """
    Plot actors' personal names specified in `actor_labels` list as time series.
    
    :param actor_offsets: dict object in form {'elizabeth': [123, 543, 4534], 'darcy': [205, 2111]}
    :param actor_labels: list of strings that should match some of the keys in `actor_offsets`
    :param normalization_constant: int
    """
    x = [actor_offsets[actor_label] for actor_label in actor_labels] 
    
    if normalization_constant:
        x = [normalize(actor_offset, normalization_constant) for actor_offset in x]
        

    with plt.style.context('fivethirtyeight'):
        plt.figure()
        n, bins, patches = plt.hist(x, NUM_BINS, label=actor_labels)
        plt.clf()
        
        ax = plt.subplot(111)
        for i, a in enumerate(n):
            ax.plot([x / (NUM_BINS - 1) for x in range(len(a))], a, label=actor_labels[i])
            
        matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['r','k','c','b','y','m','g','#54a1FF'])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plot_actor_timeseries(actors_occurences, ['darcy', 'bingley'], normalization_constant=len(processed_text))

plt.figure()

plt.show()

# Find words (adjectives) that describe Mr Darcy.

def get_actor_adjectives(doc, actor_lemma):
    """
    Find all the adjectives related to `actor_lemma` in `doc`
    
    :param doc: Spacy NLP parsed document
    :param actor_lemma: string object
    :return: list of adjectives related to `actor_lemma`
    """
    
    adjectives = []
    for ent in processed_text.ents:
        if ent.lemma_ == actor_lemma:
            for token in ent.subtree:
                if token.pos_ == 'ADJ': # Replace with if token.dep_ == 'amod':
                    adjectives.append(token.lemma_)
    
    for ent in processed_text.ents:
        if ent.lemma_ == actor_lemma:
            if ent.root.dep_ == 'nsubj':
                for child in ent.root.head.children:
                    if child.dep_ == 'acomp':
                        adjectives.append(child.lemma_)
    
    return adjectives

print(get_actor_adjectives(processed_text, 'darcy'))




# Find actors that are 'talking', 'saying', 'doing' the most. Find the
# relationship between entities and corresponding root verbs.

actor_verb_counter = Counter()
VERB_LEMMA = 'marry'

for ent in processed_text.ents:
    if ent.label_ == 'PERSON' and ent.root.head.lemma_ == VERB_LEMMA:
        actor_verb_counter[ent.text] += 1

print(actor_verb_counter.most_common(10)) 
        
# Find all the actors that got married in the book
#
# Here is an example sentence from which this information could be extracted:
# 
# "her mother was talking to that one person (Lady Lucas) freely,
# openly, and of nothing else but her expectation that Jane would soon
# be married to Mr. Bingley."
#

# Extract Keywords using noun chunks from the news article (file
# 'article.txt').  Spacy will pick some noun chunks that are not informative at
# all (e.g. we, what, who).  Try to find a way to remove non informative
# keywords.

article = read_file('/tmp/article.txt')
doc = nlp(article)

keywords = Counter()
for chunk in doc.noun_chunks:
    if nlp.vocab[chunk.lemma_].prob < - 8: # probablity value -8 is arbitrarily selected threshold
        keywords[chunk.lemma_] += 1

keywords.most_common(20)



