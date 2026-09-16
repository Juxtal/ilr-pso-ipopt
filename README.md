# ilr-pso-ipopt
This package generalizes the algorithm from M. T. Tillmann, M. Ibañez, K. Dittmer, J. Lu, et al. “Modeling and Optimising Complex Enzymatic Reaction
Processes: A Practical Guide for Biotechnologists and Bioprocess Engineers.” . On top of the original method, it adds ILR to solve the allocation problem.

Extracted and generalized from an enzyme-allocation optimization algorithm
originally developed for the author's master's thesis (a fixed-total
enzyme budget distributed across a kinetic reactor model). This package
strips away everything specific to that model and keeps only the
optimization method itself, so it can be reused on any Pyomo model with a
fixed-sum allocation constraint — not just the thesis's own.

A generic optimizer for **fixed-sum, non-negative allocation** problems
built with [Pyomo](https://www.pyomo.org/): any set of variables that must
satisfy

```
x_i >= 0          for every i
sum(x_i) == total
```

can be optimized with this package, regardless of what the variables mean
(enzyme amounts, budget shares, mixture ratios, ...). The optimizer never
looks at your model beyond that constraint — you tell it how to build the
model and, with a plain variable name, which Var forms the allocation
group. Pointing this at a different model (or a renamed/re-indexed
variable in the same model) is a one-line change, not new adapter code.

## How it works

1. Candidate allocations on the fixed-sum simplex are generated with an
   isometric log-ratio (ILR) transform, so a swarm optimizer ([MEALPY](https://github.com/thieu1995/mealpy)'s
   PSO) can search in an unconstrained space while every point it visits
   maps back to a feasible allocation.
2. Each PSO-proposed allocation is used as the initial guess for a real
   NLP solve of your Pyomo model (IPOPT by default), and scored by the
   solved objective.
3. The best solutions found so far are perturbed in ILR space to search
   for additional, distinct local optima across several rounds.

## Install

```bash
pip install -e /path/to/ilr_pso_ipopt
```

You also need the `ipopt` solver binary available to Pyomo (e.g. via
`conda install -c conda-forge ipopt`, or any other IPOPT installation on
your `PATH`) — it is not a pip package.

## Usage

```python
from ilr_pso_ipopt import optimize_allocation

def build_model():
    ...  # return a fresh pyomo.environ.ConcreteModel with an active Objective
    # model.x = pyo.Var(model.P, domain=pyo.NonNegativeReals, ...)

result = optimize_allocation(
    build_model=build_model,
    allocation_vars="x",  # <- the Var's name; every VarData under it is used
    total=10.0,            # <- the sum your allocation must add up to
    maximize=True,
)

print(result.best_objective)
print(result.best_allocation)
```

`allocation_vars` also accepts a list of names (an allocation group split
across several Var components) or a callable `(model) -> Sequence[Var]`
for anything a name can't express, e.g. a filtered subset of an indexed
Var:

```python
allocation_vars=lambda model: [model.Enz_amount_rr[r, e] for r in model.F for e in model.ENZ]
```

If your model defines more than one active `Objective`, pass
`objective_name="..."` to say which one to read after each solve
(otherwise the single active one is used automatically, and having more
than one without specifying a name raises an error rather than silently
picking one).

See `examples/` for a complete, runnable example.
