from spacy.matcher import *
from spacy.attrs import *
from spacy.tokens import Doc

matcher = Matcher(nlp.vocab)

matcher.add_pattern("HelloWorld", [{LOWER: "hello"}, {IS_PUNCT: True}, {LOWER: "world"}])

doc = nlp(u'Hello, world!')

matches = matcher(doc)



matcher.add_entity(
    "GoogleNow", # Entity ID -- Helps you act on the match.
    {"ent_type": "PRODUCT", "wiki_en": "Google_Now"}, # Arbitrary attributes (optional)
)

matcher.add_pattern(
    "GoogleNow", # Entity ID -- Created if doesn't exist.
    [ # The pattern is a list of *Token Specifiers*.
        { # This Token Specifier matches tokens whose orth field is "Google"
          ORTH: "Google"
        },
        { # This Token Specifier matches tokens whose orth field is "Now"
          ORTH: "Now"
        }
    ],
    label=None # Can associate a label to the pattern-match, to handle it better.
)

def trim_title(doc, ent_id, label, start, end):
    if doc[start].check_flag(IS_TITLE_TERM):
        return (ent_id, label, start+1, end)
    else:
        return (ent_id, label, start, end)

titles = set(title.lower() for title in [u'Mr.', 'Dr.', 'Ms.', u'Admiral'])
IS_TITLE_TERM = matcher.vocab.add_flag(lambda string: string.lower() in titles)
matcher.add_entity('PersonName', acceptor=trim_title)
matcher.add_pattern('PersonName', [{LOWER: 'mr.'}, {LOWER: 'cruise'}])
matcher.add_pattern('PersonName', [{LOWER: 'dr.'}, {LOWER: 'seuss'}])
doc = Doc(matcher.vocab, words=[u'Mr.', u'Cruise', u'likes', 'Dr.', u'Seuss'])
for ent_id, label, start, end in matcher(doc):
    print(doc[start:end].text)
    # Cruise
    # Seuss

def merge_phrases(matcher, doc, i, matches):
    '''
    Merge a phrase. We have to be careful here because we'll change the token indices.
    To avoid problems, merge all the phrases once we're called on the last match.
    '''
    if i != len(matches)-1:
        return None
    # Get Span objects
    spans = [(ent_id, label, doc[start : end]) for ent_id, label, start, end in matches]
    for ent_id, label, span in spans:
        span.merge(label=label, tag='NNP' if label else span.root.tag_)

matcher.add_entity('GoogleNow', on_match=merge_phrases)
matcher.add_pattern('GoogleNow', [{ORTH: 'Google'}, {ORTH: 'Now'}])
doc = Doc(matcher.vocab, words=[u'Google', u'Now', u'is', u'being', u'rebranded'])
matcher(doc)
print([w.text for w in doc])
