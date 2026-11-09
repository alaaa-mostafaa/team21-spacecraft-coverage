"""Genetic Algorithm
Generic: works on any ``Problem`` (the fleet pipeline, or a benchmark function).
The report chapter "Genetic Algorithm Implementation" (report/LaTeX/chapters/ga.tex) explains
every choice; the section names below match it.

    P <- pop_size or auto (1.5 dim rounded up to a multiple of 10, kept in [30, 60])
    population <- P random designs, stratified over the fleet size N (each passes the cheap checks)
    evaluate them                                              Algorithm -> Feasibility -> Engine
    while the evaluation budget is not used:
        n_e, n_c, n_m <- round(eps P), round(chi (P - n_e)), the rest              (population split)
        elites    <- the n_e best distinct designs (feasible first, then lower J), copied unchanged
        n_c children: 2 tournament parents -> align slots -> swap or blend each slot pair (crossover)
        n_m children: 1 tournament parent  -> change 1-2 active genes with step sigma_g   (mutation)
        repair, re-mutate duplicates, evaluate the children; best <- feasible first, then lower J
        population <- elites + children; sigma_g shrinks geometrically from step0 to step_min

Problem-specific parts come from an OPTIONAL part of the Problem interface:
``problem.slot_groups()`` (gene indices of each spacecraft slot) and ``integer_mask`` (the on/off
switch inside each slot).  Without slot groups (benchmark functions) every gene is its own group:
alignment is skipped, the slot swap is uniform crossover and the blend is plain BLX-alpha.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import Algorithm, OptimResult, Problem, Recorder, Step, better, local_move

# how a design was made (history column "operator")
OP_INIT, OP_SWAP, OP_BLEND, OP_MUTATION = 0, 1, 2, 3
OP_NAMES = {OP_INIT: "initial", OP_SWAP: "crossover (swap)", OP_BLEND: "crossover (blend)",
            OP_MUTATION: "mutation"}


@dataclass
class GAParams:
    max_evals: int = 1500              # evaluation budget (the initial population included)
    pop_size: int | None = None        # P; None -> auto_pop_size(dim): 40 for N_max = 5, 60 for N_max = 10
    stratified: bool = True            # start: equal share per fleet size N (False: every switch a coin flip)
    elite_frac: float = 0.05           # eps: n_e = round(eps P) best designs survive unchanged
    crossover_frac: float = 0.8        # chi: n_c = round(chi (P - n_e)) crossover children, the rest mutation
    tournament_k: int = 2              # k: designs per tournament (2 = binary tournament)
    align: bool = True                 # match the slots of the two parents before crossover
    p_blend: float = 0.3               # p_b: chance a pair of flying slots (or T) is blended instead of swapped
    alpha_blend: float = 0.3           # BLX-alpha: blend weight lambda ~ U(-alpha, 1 + alpha) per gene
    blend_weight: float | None = None  # fixed lambda (0.5 = midpoint of the parents); None -> random BLX-alpha
    step0: float = 0.25                # initial mutation step, fraction of each variable's range
    step_min: float = 0.01             # final mutation step
    max_vars_per_move: int = 2         # a mutation changes 1 .. this many active genes (as SA's move)
    seed: int = 0


def auto_pop_size(dim: int) -> int:
    """P = 1.5 dim rounded up to a multiple of 10, kept in [30, 60]  (report Eq. popsize)."""
    return int(min(60, max(30, 10 * math.ceil(1.5 * dim / 10))))


def split(P: int, elite_frac: float, crossover_frac: float) -> tuple[int, int, int]:
    """(n_e, n_c, n_m): elites, crossover children, mutation children  (report Eq. split).
    At least one child per generation, so every generation uses evaluations."""
    n_e = min(int(math.floor(elite_frac * P + 0.5)), P - 1)
    n_c = int(math.floor(crossover_frac * (P - n_e) + 0.5))
    return n_e, n_c, P - n_e - n_c


def n_generations(budget: int, P: int, n_e: int) -> int:
    """G = 1 + ceil((B - P) / (P - n_e)): the initial generation plus the bred ones (report Eq. generations)."""
    return 1 + max(0, math.ceil((budget - P) / (P - n_e)))


def _key(x: np.ndarray) -> bytes:
    """Identity of a design for duplicate checks."""
    return np.round(np.asarray(x, float), 9).tobytes()


class _Genome:
    """Gene groups of a problem: one group per spacecraft slot, every other gene on its own."""

    def __init__(self, problem: Problem):
        lo, hi = problem.bounds()
        self.span = np.asarray(hi - lo, float)
        self.integer = np.asarray(problem.integer_mask, bool)
        self.periodic = np.asarray(problem.periodic_mask, bool)
        get = getattr(problem, "slot_groups", None)
        self.slots = [np.asarray(g, int) for g in get()] if get is not None else []
        in_slot = {int(j) for g in self.slots for j in g}
        self.singles = [np.array([j]) for j in range(problem.dim) if j not in in_slot]
        self.switch = [g[self.integer[g]] for g in self.slots]              # the on/off gene of each slot
        sw = np.concatenate(self.switch) if self.slots else np.array([], int)
        # the fleet size can only be chosen if the switches can move (not with a forced N)
        self.can_set_n = bool(len(sw)) and bool(np.all(hi[sw] > lo[sw]))

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    def flying(self, x: np.ndarray, i: int) -> bool:
        sw = self.switch[i]
        return True if len(sw) == 0 else bool(np.rint(x[sw[0]]) >= 1)

    def n_of(self, x: np.ndarray) -> int:
        return sum(self.flying(x, i) for i in range(self.n_slots))

    # ---------------------------------------------------------------- slot alignment
    def distance(self, xa: np.ndarray, i: int, xb: np.ndarray, j: int) -> float:
        """Distance between slot i of A and slot j of B (report Eq. slotdist): real genes only,
        each normalised by its range; angles by the shorter arc, normalised by half a turn."""
        ga, gb = self.slots[i], self.slots[j]
        real = ~self.integer[ga]
        ga, gb = ga[real], gb[real]
        diff = xb[gb] - xa[ga]
        span = self.span[ga]
        safe = np.where(span > 0, span, 1.0)
        per = self.periodic[ga]
        d = np.where(per, np.abs(np.mod(diff + safe / 2, safe) - safe / 2) / (safe / 2), np.abs(diff) / safe)
        return float(np.sum(np.where(span > 0, d, 0.0)))

    def align(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        """B with its slots reordered so that every slot of A faces the most similar slot of B.
        Greedy matching, closest pair first: flying-flying pairs, then flying-off, then off-off."""
        n = self.n_slots
        if n < 2:
            return xb.copy()
        fa = [self.flying(xa, i) for i in range(n)]
        fb = [self.flying(xb, j) for j in range(n)]
        cost = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = (self.distance(xa, i, xb, j) + 10.0 * (fa[i] != fb[j])
                              + 20.0 * (not fa[i] and not fb[j]))
        perm = np.empty(n, int)
        for _ in range(n):
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            perm[i] = j
            cost[i, :] = np.inf
            cost[:, j] = np.inf
        y = xb.copy()
        for i in range(n):
            y[self.slots[i]] = xb[self.slots[perm[i]]]
        return y


class GeneticAlgorithm(Algorithm):
    name = "ga"
    Params = GAParams

    # ------------------------------------------------------------------ selection
    def tournament(self, costs: np.ndarray, feas: np.ndarray, rng: np.random.Generator, exclude: int = -1) -> int:
        """Index of the best of k random individuals (feasible first, then lower cost).
        ``exclude``: redraw if the winner is this individual (the second crossover parent)."""
        k = max(1, int(self.params.tournament_k))
        w = -1
        for _ in range(20):
            idx = rng.integers(0, len(costs), size=k)
            w = int(idx[0])
            for i in idx[1:]:
                if better(costs[i], feas[i], costs[w], feas[w]):
                    w = int(i)
            if w != exclude or len(costs) == 1:
                return w
        return w

    # ------------------------------------------------------------------ crossover
    def crossover(self, problem: Problem, genome: _Genome, xa: np.ndarray, xb: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, bool]:
        """Two children of A and B (report Algorithm crossover).  Returns (child1, child2, blended)."""
        p = self.params
        if p.align and genome.n_slots:
            xb = genome.align(xa, xb)
        c1, c2 = xa.copy(), xb.copy()
        blended = False
        n_single = len(genome.singles)
        for gi, g in enumerate(genome.singles + genome.slots):
            slot = gi - n_single                                  # < 0: a gene of its own (e.g. T)
            can_blend = slot < 0 or (genome.flying(xa, slot) and genome.flying(xb, slot))
            real = g[~genome.integer[g]]
            if can_blend and len(real) and rng.random() < p.p_blend:
                for j in real:                                    # blend (BLX-alpha, angles on the short arc)
                    lam = p.blend_weight if p.blend_weight is not None else \
                        rng.uniform(-p.alpha_blend, 1.0 + p.alpha_blend)
                    a, b = xa[j], xb[j]
                    if genome.periodic[j]:
                        s = genome.span[j]
                        d = np.mod(b - a + s / 2, s) - s / 2      # signed shorter arc from a to b
                        c1[j], c2[j] = a + lam * d, b - lam * d
                    else:
                        c1[j], c2[j] = a + lam * (b - a), b + lam * (a - b)
                blended = True
            elif rng.random() < 0.5:                              # swap the whole group
                c1[g], c2[g] = xb[g], xa[g]
        return problem.repair(c1), problem.repair(c2), blended

    # ------------------------------------------------------------------ helpers
    def _unique(self, problem: Problem, x: np.ndarray, seen: set, step: float, rng) -> np.ndarray:
        """Mutate x again while it duplicates a design of the population or of this generation."""
        for _ in range(10):
            if _key(x) not in seen:
                break
            x = local_move(problem, x, step, rng, self.params.max_vars_per_move)
        seen.add(_key(x))
        return x

    def _initial(self, problem: Problem, genome: _Genome, P: int, x0, rng) -> list[np.ndarray]:
        """P starting designs: stratified over the fleet size (or uniform), each passing the cheap
        checks if possible; x0 (start from a given design) is the first of them."""
        draw = getattr(problem, "random_feasible", None)
        xs = [problem.repair(np.asarray(x0, float))] if x0 is not None else []
        m = P - len(xs)
        sizes: list[int | None] = [None] * m
        if self.params.stratified and genome.can_set_n:
            n_max = genome.n_slots
            sizes = [n for n in range(1, n_max + 1) for _ in range(m // n_max)]
            sizes += [int(n) + 1 for n in rng.choice(n_max, size=m - len(sizes), replace=False)]
            sizes = [sizes[i] for i in rng.permutation(len(sizes))]
        for n in sizes:
            try:
                if draw is None:
                    x = problem.random_solution(rng)
                elif n is None:
                    x = draw(rng)
                else:
                    x = draw(rng, n=n)
            except RuntimeError:                                  # no feasible draw: keep a penalised one
                x = problem.random_solution(rng, n) if n is not None else problem.random_solution(rng)
            xs.append(np.asarray(x, float))
        return xs

    @staticmethod
    def diversity(X: np.ndarray, genome: _Genome) -> float:
        """Mean spread of the genes, as a fraction of their range (uniform random population ~ 0.29).
        Angles: 0.2887 (1 - R), R = length of the mean unit vector (0 when spread evenly)."""
        ok = genome.span > 0
        if len(X) < 2 or not np.any(ok):
            return 0.0
        span = np.where(ok, genome.span, 1.0)
        lin = X.std(axis=0) / span
        ang = 2 * np.pi * X / span
        R = np.hypot(np.cos(ang).mean(axis=0), np.sin(ang).mean(axis=0))
        d = np.where(genome.periodic, 0.2887 * (1.0 - R), lin)
        return float(d[ok].mean())

    # ------------------------------------------------------------------ run
    def run(self, problem: Problem, x0: np.ndarray | None = None,
            callback: Callable[[Step], bool | None] | None = None) -> OptimResult:
        t_start = time.perf_counter()
        p = self.params
        rng = np.random.default_rng(p.seed)
        rec = Recorder(problem, p.max_evals, callback)
        genome = _Genome(problem)
        P = max(2, int(p.pop_size or auto_pop_size(problem.dim)))
        n_e, n_c, n_m = split(P, p.elite_frac, p.crossover_frac)
        G = n_generations(p.max_evals, P, n_e)
        credit = {op: 0 for op in OP_NAMES}
        gen_stats: dict[str, list] = {k: [] for k in ("evals", "best", "mean", "feasible", "diversity",
                                                      "N_mean", "N_min", "N_max", "step")}

        def evaluate(x, op, g, step, parent):
            before = rec.best
            ev = rec.evaluate(x)
            improved = rec.best is ev and ev is not before
            credit[op] += int(improved)
            rec.report(x, ev, improved, None, {"generation": g, "operator": op, "step": step,
                                               "x_prev": parent if parent is not None else x})
            return ev

        def record(g_x, g_ev, step):
            costs = np.array([e.cost for e in g_ev])
            feas = np.array([e.feasible for e in g_ev])
            gen_stats["evals"].append(rec.n_evals)
            gen_stats["best"].append(rec.best.cost)
            gen_stats["mean"].append(float(costs[feas].mean()) if feas.any() else float("nan"))
            gen_stats["feasible"].append(float(feas.mean()))
            gen_stats["diversity"].append(self.diversity(np.array(g_x), genome))
            ns = [genome.n_of(x) for x in g_x] if genome.n_slots else [0]
            gen_stats["N_mean"].append(float(np.mean(ns)))
            gen_stats["N_min"].append(float(min(ns)))
            gen_stats["N_max"].append(float(max(ns)))
            gen_stats["step"].append(step)

        # ---- generation 0
        pop_x, pop_ev = [], []
        for x in self._initial(problem, genome, P, x0, rng):
            if rec.exhausted:
                break
            pop_ev.append(evaluate(x, OP_INIT, 0, p.step0, None))
            pop_x.append(x)

        # ---- generations 1, 2, ...
        g = 0
        while not rec.exhausted:
            step = p.step0 * (p.step_min / p.step0) ** (g / max(G - 2, 1))      # report Eq. sigma
            record(pop_x, pop_ev, step)
            costs = np.array([e.cost for e in pop_ev])
            feas = np.array([e.feasible for e in pop_ev])
            order = sorted(range(len(pop_x)), key=lambda i: (not feas[i], costs[i]))
            elites, kept = [], set()
            for i in order:                                      # elitism: n_e best DISTINCT designs
                if len(elites) == n_e:
                    break
                k = _key(pop_x[i])
                if k not in kept:
                    kept.add(k)
                    elites.append(i)

            seen = {_key(x) for x in pop_x}
            children: list[tuple[np.ndarray, int, np.ndarray]] = []
            while len(children) < n_c:                           # crossover children
                a = self.tournament(costs, feas, rng)
                b = self.tournament(costs, feas, rng, exclude=a)
                c1, c2, blended = self.crossover(problem, genome, pop_x[a], pop_x[b], rng)
                for c in (c1, c2):
                    if len(children) < n_c:
                        children.append((self._unique(problem, c, seen, step, rng),
                                         OP_BLEND if blended else OP_SWAP, pop_x[a]))
            for _ in range(n_m):                                 # mutation children
                a = self.tournament(costs, feas, rng)
                c = local_move(problem, pop_x[a], step, rng, p.max_vars_per_move)
                children.append((self._unique(problem, c, seen, step, rng), OP_MUTATION, pop_x[a]))

            new_x = [pop_x[i] for i in elites]
            new_ev = [pop_ev[i] for i in elites]
            for c, op, parent in children:
                if rec.exhausted:
                    break
                new_ev.append(evaluate(c, op, g + 1, step, parent))
                new_x.append(c)
            pop_x, pop_ev = new_x, new_ev
            g += 1
        if pop_x:
            record(pop_x, pop_ev, float("nan"))                  # the last (possibly incomplete) generation

        return self._result(rec, t_start, "budget used", {
            "pop_size": P, "n_elite": n_e, "n_crossover": n_c, "n_mutation": n_m,
            "generations_planned": G, "generations_run": g + 1,
            "improvements_by_operator": {OP_NAMES[k]: v for k, v in credit.items()},
            "generations": {k: np.array(v, float) for k, v in gen_stats.items()}})


__all__ = ["GAParams", "GeneticAlgorithm", "auto_pop_size", "split", "n_generations", "OP_NAMES"]
