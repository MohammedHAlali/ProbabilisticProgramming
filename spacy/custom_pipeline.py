
def arbitrary_fixup_rules(doc):
    for token in doc:
        if token.text == u'bill' and token.tag_ == u'NNP':
            token.tag_ = u'NN'

def custom_pipeline(nlp):
    return (nlp.tagger, arbitrary_fixup_rules, nlp.parser, nlp.entity)

nlp = spacy.load('en', create_pipeline=custom_pipeline)

nlp.pipeline

nlp.pipeline = [nlp.tagger]
