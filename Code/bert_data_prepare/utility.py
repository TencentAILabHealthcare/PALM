# -*- coding: utf-8 -*-

import re
from enum import Enum

class IupacAminoAcid(Enum):
    A = ('A', 'Ala', 'Alanine')
    C = ('C', 'Cys', 'Cysteine')
    D = ('D', 'Asp', 'Aspartic acid')
    E = ('E', 'Glu', 'Glutamic acid')
    F = ('F', 'Phe', 'Phenylalanine')
    G = ('G', 'Gly', 'Glycine')
    H = ('H', 'His', 'Histidine')
    I = ('I', 'Ile', 'Isoleucine')
    K = ('K', 'Lys', 'Lysine')
    L = ('L', 'Leu', 'Leucine')
    M = ('M', 'Met', 'Methionine')
    N = ('N', 'Asn', 'Asparagine')
    P = ('P', 'Pro', 'Proline')
    Q = ('Q', 'Gln', 'Glutamine')
    R = ('R', 'Arg', 'Arginine')
    S = ('S', 'Ser', 'Serine')
    T = ('T', 'Thr', 'Threonine')
    V = ('V', 'Val', 'Valine')
    W = ('W', 'Trp', 'Tryptophan')
    Y = ('Y', 'Tyr', 'Tyrosine')

    # O = ('O', 'Pyl', 'Pyrrolysine')
    # U = ('U', 'Sec', 'Selenocysteine')
    # B = ('B', 'Asx', 'Aspartic acid or Asparagine')
    # Z = ('Z', 'Glx', 'Glutamic acid or Glutamine')
    # X = ('X', 'Xaa', 'Any amino acid')
    # J = ('J', 'Xle', 'Leucine or Isoleucine')

    @property
    def code(self):
        return self.value[0]

    @property
    def abbr(self):
        return self.value[1]

    @property
    def name(self):
        return self.value[2]

    @classmethod
    def codes(cls):
        return [c.value[0] for c in cls]

    @classmethod
    def abbrs(cls):
        return [c.value[1] for c in cls]

    @classmethod
    def names(cls):
        return [c.value[2] for c in cls]

AMINO_ACID = IupacAminoAcid
GAP = '-'

def is_valid_aaseq(seq, allow_gap=False):
    aas = ''.join(AMINO_ACID.codes())
    if allow_gap:
        aas = aas + GAP
    pattern = '^[%s]+$' % aas
    found = re.match(pattern, str(seq))
    return found is not None