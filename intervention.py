import os

from rpy2.rinterface import NA_Real, NA_Integer, NA_Character, NA_Logical, NULL
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects.vectors import IntVector, FloatVector

def load_r_module(name):
    path = os.path.join(os.path.dirname(__file__), name + '.R')
    with open(path, 'r') as f:
        string = f.read()
    return SignatureTranslatedAnonymousPackage(string, name)

INTERVENTION = load_r_module('intervention')

def is_na(x):
    return x is NA_Real or x is NA_Integer or x is NA_Character or x is NA_Logical

def is_null(x):
    return x is NULL

def list_to_vector(xs, type=FloatVector):
    return NULL if len(xs) == 0 else type([NA_Real if x is None else x for x in xs])

def vector_to_list(xs, type=float):
    return None if is_null(xs) else [None if is_na(x) else type(x) for x in xs]

def vector_to_single(xs, type=float):
    return None if is_null(xs) else vector_to_list(xs, type)[0]

class InterventionAtTimeResult(object):
    def __init__(self, result, pvalue_complement=False):
        d = dict(result.items())
        self.order = vector_to_list(d['order'], int)
        if self.order is not None:
            self.order = tuple(self.order)
        self.aos = vector_to_list(d['aos'], int)
        if self.aos is not None:
            self.aos = [x - 1 for x in self.aos]
        self.intervention = vector_to_single(d['intervention'])
        self.interventions = vector_to_list(d['interventions'])
        self.null_interventions = vector_to_list(d['null.interventions'])
        self.pvalues = vector_to_list(d['test.p.values'])
        if pvalue_complement:
            self.pvalues = [1.0 - f for f in self.pvalues]
    def str_lines(self):
        return [
            'order: %s' % str(self.order),
            'removed additive outliers: %s' % self.aos,
            'intervention: %f' % self.intervention,
            'interventions: [%s]' % ', '.join('%f' % f for f in self.interventions),
            'null-interventions: %s' % self.null_interventions,
            'p-values: [%s]' % ', '.join('%f' % f for f in self.pvalues),
        ]
    def __str__(self):
        return '\n'.join(self.str_lines())

def intervention_at_time(xs, ts, null_interventions, ao_alpha=None, stationary=False, pvalue_complement=False):
    xs_vector = list_to_vector(xs)
    ts_vector = list_to_vector([t + 1 for t in ts], IntVector)
    null_interventions_vector = list_to_vector(null_interventions)
    ao_alpha = NULL if ao_alpha is None else ao_alpha
    return InterventionAtTimeResult(INTERVENTION.intervention_at_time(xs_vector, ts_vector, null_interventions_vector, ao_alpha, stationary), pvalue_complement=pvalue_complement)
