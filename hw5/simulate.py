#!/usr/bin/env python3

import bisect
import itertools as it
import random

import numpy as np
from hmmlearn import hmm


class Model:
    def __init__(self, cpg_to_cpg, normal_to_normal, cpg_to_normal, normal_to_cpg):
        self.alphabet = "ACGT"
        self.cpg_to_cpg = cpg_to_cpg
        self.normal_to_normal = normal_to_normal
        self.cpg_to_normal = cpg_to_normal
        self.normal_to_cpg = normal_to_cpg

    def _cgp_probs(self):
        joint = np.concatenate(
            (
                (1 - self.cpg_to_normal) * self.cpg_to_cpg,
                self.cpg_to_normal * self.normal_to_normal,
            ),
            axis=1,
        )

        return joint

    def _normal_probs(self):
        joint = np.concatenate(
            (
                self.normal_to_cpg * self.cpg_to_cpg,
                (1 - self.normal_to_cpg) * self.normal_to_normal,
            ),
            axis=1,
        )

        return joint

    def transition_probs(self):
        return np.concatenate((self._cgp_probs(), self._normal_probs()))

    def generate(self, length):
        transition_cumsum = self.transition_probs().cumsum(axis=1)
        bases = []
        cpg_sites = []
        cpg_start = None
        in_cpg = False

        start = random.randrange(len(transition_cumsum))
        bases.append(self.alphabet[start % len(self.alphabet)])

        if start < 4:
            in_cpg = True
            cpg_start = start

        prev_index = start
        for i in range(1, length):
            p = random.random()
            next_index = bisect.bisect_left(transition_cumsum[prev_index], p)
            bases.append(self.alphabet[next_index % len(self.alphabet)])

            if next_index < 4:
                if not in_cpg:
                    cpg_start = i
                    in_cpg = True
            else:
                if in_cpg:
                    cpg_sites.append((cpg_start, i - 1))
                    in_cpg = False

            prev_index = next_index

        if in_cpg:
            cpg_sites.append((cpg_start, length - 1))

        return "".join(bases), cpg_sites


# Rows are ACGT, columns are ACGT
cpg_to_cpg_probs = np.array(
    [
        [0.1, 0.4, 0.4, 0.1],
        [0.05, 0.45, 0.45, 0.05],
        [0.05, 0.45, 0.45, 0.05],
        [0.1, 0.4, 0.4, 0.1],
    ]
)

normal_to_normal_probs = np.array(
    [
        [0.25, 0.25, 0.25, 0.25],
        [0.15, 0.35, 0.35, 0.15],
        [0.15, 0.35, 0.35, 0.15],
        [0.25, 0.25, 0.25, 0.25],
    ]
)

cpg_to_normal_prob = 0.005
normal_to_cpg_prob = 0.0025
