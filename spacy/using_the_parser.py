# Using the dependency parse

import spacy
nlp = spacy.load('en')

doc = nlp(u'I like green eggs and ham.')
for np in doc.noun_chunks:
    print(np.text, np.root.text, np.root.dep_, np.root.head.text)
    # I I nsubj like
    # green eggs eggs dobj like
    # ham ham conj eggs

doc.is_parsed

from spacy.symbols import det

the, dog = nlp(u'the dog')
assert the.dep == det
assert the.dep_ == 'det'

doc.dep

from spacy.symbols import nsubj, VERB
# Finding a verb with a subject from below — good
verbs = set()
for possible_subject in doc:
    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
        verbs.add(possible_subject.head)

# Finding a verb with a subject from above — less good
verbs = []
for possible_verb in doc:
    if possible_verb.pos == VERB:
        for possible_subject in possible_verb.children:
            if possible_subject.dep == nsubj:
                verbs.append(possible_verb)
                break

for possible_subject in doc:
    print( possible_subject.dep )

apples = nlp(u'bright red apples on the tree')[2]
print([w.text for w in apples.lefts])
# ['bright', 'red']
print([w.text for w in apples.rights])
# ['on']
assert apples.n_lefts == 2
assert apples.n_rights == 1

from spacy.symbols import nsubj
doc = nlp(u'Credit and mortgage account holders must submit their requests within 30 days.')
root = [w for w in doc if w.head is w][0]
subject = list(root.lefts)[0]
for descendant in subject.subtree:
    assert subject.is_ancestor_of(descendant)

from spacy.symbols import nsubj
doc = nlp(u'Credit and mortgage account holders must submit their requests.')
holders = doc[4]
span = doc[holders.left_edge.i : holders.right_edge.i + 1]
span.merge()
for word in doc:
    print(word.text, word.pos_, word.dep_, word.head.text)
    # Credit and mortgage account holders nsubj NOUN submit
    # must VERB aux submit
    # submit VERB ROOT submit
    # their DET det requests
    # requests NOUN dobj submit

import spacy

nlp = spacy.load('en', parser=False)


