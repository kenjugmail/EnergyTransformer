"""Synthetic sentence tasks, exact-text-disjoint by stable hash.

The semantics are part of the benchmark specification: declarative supported
location updates write; explicit forgetting deletes; hypothetical/quoted/report
sentences are distractors. No real-person information or external corpus is used.
"""
import hashlib
import random
from .bridge import NAMES, VALUES, HOLD, SET, DELETE, ASK

TRAIN_TEMPLATES={
 SET:[
 '{name} is in the {value}.','{name} moved to the {value}.',
 '{name} went to the {value}.','put {name} in the {value}.',
 'update the location of {name} to the {value}.','{name} now stays in the {value}.',
 'correction {name} is now in the {value}.','remember that {name} is in the {value}.',
 'the current location of {name} is the {value}.','set {name} location to the {value}.'],
 DELETE:[
 'forget the location of {name}.','delete the location of {name}.',
 '{name} location is unknown.','clear the location of {name}.',
 'no location is recorded for {name}.','remove the location of {name}.',
 'forget where {name} is.','the current location of {name} is no longer known.'],
 ASK:[
 'where is {name}?','tell me the location of {name}.',
 'what is the location of {name}?','where can i find {name}?',
 'please recall the location of {name}.','which room is {name} in?',
 'what place is {name} in?','do you know where {name} is?'],
 HOLD:[
 '{name} might go to the {value}.','someone said {name} is in the {value}.',
 'do not update {name} to the {value}.','pretend {name} is in the {value}.',
 'if {name} were in the {value} it would be quiet.',
 '{name} is not in the {value}.','the rumor says {name} is in the {value}.',
 'do not forget the location of {name}.','do not delete the location of {name}.',
 'a story says {name} moved to the {value}.',
 'the {value} is next to the road.','today the weather is clear.']}

HELDOUT_TEMPLATES={
 SET:['the location of {name} is the {value} now.',
      'please remember the current location of {name} is the {value}.',
      '{name} is currently inside the {value}.'],
 DELETE:['please clear where {name} is.',
         'remove what you know about the location of {name}.',
         'erase the stored whereabouts of {name}.'],
 ASK:['please tell me where {name} is now.',
      'recall what place {name} is in.',
      'what are the whereabouts of {name}?'],
 HOLD:['do not change the location of {name} to the {value}.',
       'maybe {name} went to the {value}.',
       'the story about {name} in the {value} is only imaginary.']}

def bucket(text):
 return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)%10

def record(rng,split='train',op=None,key=None,value=None):
 templates=HELDOUT_TEMPLATES if split=='wording' else TRAIN_TEMPLATES
 for _ in range(20000):
  o=rng.randrange(4) if op is None else op
  k=rng.randrange(16) if key is None else key
  v=rng.randrange(16) if value is None else value
  template=rng.choice(templates[o]);text=template.format(name=NAMES[k],value=VALUES[v])
  b=bucket(text)
  if split in ('wording','composition') or (split=='train' and b<8) or (split=='validation' and b==8) or (split=='iid' and b==9):
   return dict(text=text,op=o,key=k,value=v,template=template)
 raise RuntimeError('Unable to generate split-constrained record')

def dataset(count,seed,split):
 rng=random.Random(seed)
 return [record(rng,split) for _ in range(count)]

def vocabulary_texts():
 # Train grammar and train names/values only. Held-out wording never sets vocabulary.
 return [t.format(name=n,value=v) for ts in TRAIN_TEMPLATES.values() for t in ts
         for n in NAMES for v in VALUES]
