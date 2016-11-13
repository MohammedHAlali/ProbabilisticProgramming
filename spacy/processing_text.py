import spacy

nlp = spacy.load('en')

doc = nlp(u'Hello, spacy!')

print( (w.text, w.pos_) for w in doc )

for w in doc:
    print(w.text,'\t' ,w.pos_)

text = u'Hello, world! A three sentence document.\nWith new lines...'
doc = nlp(text)

doc = nlp.make_doc(text)
for proc in nlp.pipeline:
    proc(doc)

doc = nlp.tokenizer(text)
nlp.tagger(doc)
nlp.parser(doc)
nlp.entity(doc)

for doc in nlp.pipe(texts, batch_size=10000, n_threads=3):
   pass

doc = Doc(nlp.vocab
        , words=[u'Hello', u',', u'world', u'!']
        , spaces=[False, True, False, False])
