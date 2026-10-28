"""Checkpointable parameter-block CMA-ES with independent interaction weights.

Each block is an ordinary pycma solve in one physical coordinate. Cycling
these solves preserves the four different objectives; this is not a joint
four-dimensional covariance search.
"""
from pathlib import Path
import pickle
import warnings

import numpy as np

from .parameters import PARAMETERS


def weighted_loss(losses, confidence):
    losses, confidence = np.asarray(losses, float), np.asarray(confidence, float)
    if losses.shape != confidence.shape or not np.isfinite(confidence).all() or (confidence < 0).any():
        raise ValueError('Expected a finite nonnegative confidence per support episode')
    if not np.isfinite(losses).all():
        return np.inf
    # Column normalization preserves candidate rankings and gives all methods
    # comparable numerical magnitudes. A zero column falls back to equal weights.
    weights = confidence / confidence.sum() if confidence.sum() > 0 else np.ones_like(confidence)/len(confidence)
    return float(weights @ losses)


class BlockCMA:
    def __init__(self, start, confidence, seed=9101, cycles=2, block_generations=3, population=4, sigma=.2):
        self.current = np.asarray(start, float).copy()
        self.confidence = np.asarray(confidence, float).copy()
        if self.current.shape != (4,) or np.any(abs(self.current) > .025):
            raise ValueError('Start outside the frozen local bounds')
        if self.confidence.ndim != 2 or self.confidence.shape[1] != 4 or not np.isfinite(self.confidence).all() or (self.confidence < 0).any():
            raise ValueError('Expected confidence[episode,4]')
        if min(cycles, block_generations) < 1 or population < 4 or sigma <= 0:
            raise ValueError('Invalid CMA budget')
        self.seed, self.cycles = seed, cycles
        self.block_generations, self.population, self.sigma = block_generations, population, sigma
        self.block = self.generation = self.logical_candidates = 0
        self.trace, self.seen, self.pending = [], {tuple(self.current)}, None
        self.es = None

    @property
    def done(self):
        return self.block >= self.cycles * 4

    def ask(self):
        import cma
        if self.done:
            return None
        if self.pending is not None:
            return self.pending
        j = self.block % 4
        if self.es is None:
            self.anchor = self.current.copy()
            self.best_q, self.best_cost = self.anchor.copy(), np.inf
            rng = np.random.RandomState(self.seed + self.block * 1009)
            # pycma 4.5 warns about N=1, and its automatic maxstd setter has
            # an N=1 initialization bug. Disable that optional std cap; bounded
            # phenotype transformation still enforces all physical bounds.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self.es = cma.CMAEvolutionStrategy(
                    [(self.current[j]+.025)/.05], self.sigma,
                    dict(bounds=[0.,1.], maxstd=np.inf, popsize=self.population,
                         randn=rng.randn, seed=self.seed+self.block*1009,
                         CMA_active=False, verbose=-9))
        x = self.es.ask()
        q = np.tile(self.anchor, (len(x),1))
        q[:,j] = np.clip(np.array(x)[:,0],0.,1.)*.05-.025
        self.pending = dict(block=self.block, generation=self.generation,
                            parameter=PARAMETERS[j], x=x, q=q, incumbent=self.best_q.copy())
        return self.pending

    def tell(self, candidate_losses, incumbent_losses):
        if self.pending is None:
            raise ValueError('ask() must precede tell()')
        p, j = self.pending, self.block % 4
        losses = np.asarray(candidate_losses, float)
        if losses.shape != (self.population,len(self.confidence)):
            raise ValueError('One complete support loss vector per CMA candidate is required')
        costs = np.array([weighted_loss(row,self.confidence[:,j]) for row in losses])
        old_cost = weighted_loss(incumbent_losses,self.confidence[:,j])
        if old_cost < self.best_cost:
            self.best_cost = old_cost
        for q,cost in zip(p['q'],costs):
            self.seen.add(tuple(q))
            if cost < self.best_cost-1e-12:
                self.best_q, self.best_cost = q.copy(), float(cost)
        # Same finite failure penalty in both arms; failed candidates cannot
        # become selected incumbents. Raw costs/validity are retained in trace.
        self.es.tell(p['x'],np.where(np.isfinite(costs),costs,1e12).tolist())
        self.logical_candidates += self.population
        self.generation += 1
        block_end = self.generation == self.block_generations
        if block_end:
            self.current = self.best_q.copy()
        self.trace.append(dict(block=self.block,cycle=self.block//4,parameter=PARAMETERS[j],
            generation=self.generation, q=p['q'], costs_mm=costs, support_losses_mm=losses,
            incumbent_q=p['incumbent'],incumbent_cost_mm=old_cost, best_q=self.best_q.copy(),
            best_cost_mm=self.best_cost,current_q=self.current.copy(),block_complete=block_end,
            candidate_vectors=self.logical_candidates,
            logical_episode_evaluations=self.logical_candidates*len(self.confidence),
            episode_evaluations_including_start=(self.logical_candidates+1)*len(self.confidence),
            unique_candidate_vectors=len(self.seen),sigma=float(self.es.sigma)))
        if block_end:
            self.block += 1
            self.generation, self.es = 0, None
        self.pending = None
        return self.trace[-1]

    def save(self, path):
        path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_bytes(pickle.dumps(self,pickle.HIGHEST_PROTOCOL));temp.replace(path)

    @classmethod
    def load(cls, path):
        # Only load our own experiment checkpoint, never data-provided pickles.
        state = pickle.loads(Path(path).read_bytes())
        if not isinstance(state,cls):raise ValueError('Unexpected search checkpoint type')
        return state
