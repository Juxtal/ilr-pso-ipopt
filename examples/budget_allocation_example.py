"""
Example: allocate a fixed budget across N projects to maximize return.

This is a stand-in for *any* "allocate a fixed total across several
things" model - the optimizer in ilr_pso_ipopt never looks at what the
allocation represents. To point it at a different model you only change
two things: build_model() (your own Pyomo model) and allocation_vars
(the *name* of the Var component that must sum to `total` - see main()
below). No adapter function to rewrite, so a variable rename, a
re-indexed set, or swapping in an entirely different model all just mean
changing that one string.

The objective here is intentionally non-convex (diminishing returns per
project, plus pairwise synergy/cannibalization terms between projects),
so a single IPOPT solve from one starting point can get stuck in
different local optima depending on where it starts - which is exactly
why this package restarts from several ILR-PSO-proposed allocations.

Run with:
    python examples/budget_allocation_example.py
"""

import numpy as np
import pyomo.environ as pyo

from ilr_pso_ipopt import optimize_allocation

# ------------------------------------------------------------------
# Problem data: 6 projects sharing one fixed total budget.
# ------------------------------------------------------------------

N_PROJECTS = 6
TOTAL_BUDGET = 10.0

rng = np.random.default_rng(0)
LINEAR_RETURN = rng.uniform(2.0, 5.0, N_PROJECTS)          # a_i
DIMINISHING_RETURN = rng.uniform(0.2, 0.6, N_PROJECTS)     # b_i (x_i^2 penalty)

# Symmetric pairwise synergy (+) / cannibalization (-) matrix, zero diagonal.
SYNERGY = rng.uniform(-0.3, 0.3, (N_PROJECTS, N_PROJECTS))
SYNERGY = (SYNERGY + SYNERGY.T) / 2.0
np.fill_diagonal(SYNERGY, 0.0)


def build_model():
    """Return a fresh Pyomo model for this problem (called once per solve)."""
    model = pyo.ConcreteModel()
    model.P = pyo.RangeSet(0, N_PROJECTS - 1)

    # The allocation variables: how much budget goes to each project.
    model.x = pyo.Var(model.P, domain=pyo.NonNegativeReals, bounds=(0.0, TOTAL_BUDGET))

    def objective_rule(m):
        direct_return = sum(
            LINEAR_RETURN[i] * m.x[i] - DIMINISHING_RETURN[i] * m.x[i] ** 2
            for i in m.P
        )
        synergy = sum(
            SYNERGY[i, j] * m.x[i] * m.x[j]
            for i in m.P for j in m.P if i < j
        )
        return direct_return + synergy

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    return model


def main():
    result = optimize_allocation(
        build_model=build_model,
        allocation_vars="x",      # <- just the Var's name; no per-model adapter code
        total=TOTAL_BUDGET,       # <- the "总和" (total) you can set yourself
        maximize=True,
        first_pop_size=8,
        first_epochs=5,
        second_pop_size=8,
        second_epochs=5,
        max_search_rounds=5,
    )

    print("\n=== RESULT ===")
    print("Best objective :", result.best_objective)
    print("Best allocation:", np.round(result.best_allocation, 3))
    print("Allocation sum :", round(float(np.sum(result.best_allocation)), 6))
    print("Local optima found:", len(result.stored_objectives))
    print("Solver calls:", result.run_stats["solver_calls"])


if __name__ == "__main__":
    main()
