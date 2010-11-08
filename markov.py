#!/usr/bin/python
from itertools import *
from random import random, randint, seed
from cPickle import dump,load,loads, HIGHEST_PROTOCOL
import os, re, bz2

def moving_window(seq, size=2, prefix=0, suffix=0, prefixitem=None, suffixitem=None):
    it = chain([prefixitem]*prefix, seq, [suffixitem]*suffix)
    window = tuple(islice(it, size))
    while True:
        yield window
        window = window[1:]+(it.next(),)

def cumulative_distribution(values, probabilities):
    cumulative_probability, distribution = 0, []
    for value,probability in sorted(zip(values, probabilities), key=lambda (v,p):p, reverse=True):
        cumulative_probability += probability
        distribution.append((cumulative_probability, value))
    return distribution

class MarkovModel(object):
    def __init__(self, length, combiner=tuple, probabilities={}):
        self.transition_counts = {}
        self.length = length
        self.combiner = combiner
        self.probabilities = probabilities

    def train(self, datasets):
        for item in datasets:
            for window in moving_window(item, self.length+1, prefix=self.length, suffix=1):
                state, transition = window[:-1], window[-1]
                self.transition_counts[state][transition] = self.transition_counts.setdefault(state,{}).get(transition,0) + 1
        
        self._update_probabilities()

    def _update_probabilities(self):
        self.probabilities = {}
        
        for state, transition_counts in self.transition_counts.iteritems():
            total_transitions = sum(transition_counts.values())
            
            transitions = transition_counts.keys()
            probabilities = [float(amount)/total_transitions for amount in transition_counts.values()]
            
            self.probabilities[state] = cumulative_distribution(transitions, probabilities)

    def generate(self):
        def fetch_transition(probability_distribution, position):
            for probability, transition in probability_distribution:
                if probability > position:
                    return transition
        
        def iterate_random_transitions():
            state = (None,)*self.length
            while True:
                transition = fetch_transition(self.probabilities[state], position=random())
                if not transition:
                    break
                yield transition
                state = state[1:] + (transition,)
        
        return self.combiner(iterate_random_transitions())

    def save(self, file, *args):
        compressed_file = bz2.BZ2File(file,'w', buffering=2*10**5)
        dump((self.probabilities, self.length) + tuple(args), compressed_file, HIGHEST_PROTOCOL)

    def load(cls, file, **kwargs):
        data = loads(bz2.BZ2File(file).read())
        return (cls(data[1], probabilities=data[0], **kwargs),) + tuple(data[2:])
    load = classmethod(load)

if __name__ == "__main__":
    from optparse import OptionParser, Option
    def parse_option(option):
        m = re.match('(?P<type>[^ ]+) (.) ([^=:]*)(?:=(?P<default>[^:]+))?(?:: (?P<help>.*))?', option)
        return Option('-'+m.group(2), '--'+m.group(3), dest=m.group(3), **m.groupdict())

    options,args = OptionParser(description="Generate sentences and words using a markov model", option_list=map(parse_option, [
        'string d datadir=data: Directory to read example files from. [default: %default]',
        'string e extension=.txt: Filename extension of example files.  [default: %default]',
        'int l length=2: Length of state of the markov model.  [default: %default]',
        'int n amount=1: Amount of sentences to generate.  [default: %default]',
        'string c charset=utf8: Charset of input files.  [default: %default]',
        'string i inputfile: Load model from file',
        'string o outputfile: Store model to file',
        'string r seed: Use seed for generation']) + [
        Option('-s', '--sentence', action='store_true', dest="sentence", default=True, help="Do sentence generation. [default]"),
        Option('-w', '--word', action='store_false', dest="sentence", help="Do word generation."),
    ]).parse_args()
    
    
    def split_to_words_and_punctuation(sentence):
        return re.findall('(?:\\w|-)+|\\S+', sentence, re.U)
    
    def split_to_words(data):
        return re.findall('(?:\\w|-)+', data, re.U)
    
    def split_to_sentences(data):
        return re.split('(?<=[.?!]) ', data)
    
    def join_words_and_punctuation(words):
        return re.sub(' ([,.?!])','\\1', " ".join(words))
    
    model, settings = options.inputfile and MarkovModel.load(options.inputfile) or (MarkovModel(options.length),options)
    
    model.combiner = settings.sentence and join_words_and_punctuation or "".join

    if not options.inputfile:
        def read_file(filename):
            return open(os.path.join(options.datadir, filename)).read().decode(options.charset)
        
        def files():
            for filename in os.listdir(options.datadir):
                if filename.endswith(options.extension):
                    yield read_file(filename)
        def wordgenerator():
            for file in files():
                for sentence in split_to_sentences(file):
                    yield split_to_words_and_punctuation(sentence)
        
        def chargenerator():
            for file in files():
                for word in split_to_words(file):
                    yield word.lower()
        
        if settings.sentence:
            tokens = wordgenerator()
        else:
            tokens = chargenerator()
        
        model.train(tokens)
    
    if options.outputfile:
        model.save(options.outputfile, options)
    else:
        seed(options.seed)
        print "\n".join([model.generate().encode('utf8') for i in range(int(options.amount))])
