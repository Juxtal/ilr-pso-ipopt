"""
Second demo model, with every name changed from budget_allocation_example.py,
to show that swapping models only means writing a new build_model() /
picking a new allocation_vars string - nothing in ilr_pso_ipopt itself
changes.

This one also uses Pyomo's AbstractModel + create_instance(data) style
instead of ConcreteModel, to show that ilr_pso_ipopt doesn't care how you
build the model either - build_model() just has to return something
usable (i.e. a constructed/concrete instance) each time it's called.

Story: mix N pigments into a fixed total mass of paint to maximize a
"vividness" score. Vividness has diminishing returns per pigment plus
pairwise blend effects (some pigment pairs boost each other, some dull
each other) - non-convex, so several restarts can land on different
mixes.

Run with:
    python examples/paint_mixing_example.py
"""

import numpy as np
import pyomo.environ as pyo

from ilr_pso_ipopt import optimize_allocation

# ------------------------------------------------------------------
# Problem data: 5 pigments sharing one fixed total paint mass.
# ------------------------------------------------------------------

N_PIGMENTS = 5
TOTAL_PAINT_MASS = 8.0

rng = np.random.default_rng(7)
BASE_VIVIDNESS = rng.uniform(1.5, 4.0, N_PIGMENTS)          # reward per unit of pigment
SATURATION_PENALTY = rng.uniform(0.15, 0.5, N_PIGMENTS)     # diminishing returns

# Symmetric pairwise blend effect (+ boosts, - dulls), zero diagonal.
BLEND_EFFECT = rng.uniform(-0.25, 0.25, (N_PIGMENTS, N_PIGMENTS))
BLEND_EFFECT = (BLEND_EFFECT + BLEND_EFFECT.T) / 2.0
np.fill_diagonal(BLEND_EFFECT, 0.0)

PAINT_DATA = {
    None: {
        "Pigment": {None: list(range(N_PIGMENTS))},
        "base_vividness": {i: float(BASE_VIVIDNESS[i]) for i in range(N_PIGMENTS)},
        "saturation_penalty": {i: float(SATURATION_PENALTY[i]) for i in range(N_PIGMENTS)},
        "blend_effect": {
            (i, j): float(BLEND_EFFECT[i, j])
            for i in range(N_PIGMENTS) for j in range(N_PIGMENTS)
        },
        "total_mass": {None: TOTAL_PAINT_MASS},
    }
}


def _declare_paint_model() -> pyo.AbstractModel:
    """Declare the (data-free) abstract structure of the pigment-mixing model."""
    model = pyo.AbstractModel()

    model.Pigment = pyo.Set()
    model.base_vividness = pyo.Param(model.Pigment)
    model.saturation_penalty = pyo.Param(model.Pigment)
    model.blend_effect = pyo.Param(model.Pigment, model.Pigment)
    model.total_mass = pyo.Param()

    def dose_bounds(m, _i):
        return (0.0, pyo.value(m.total_mass))

    # The allocation variables: how much mass of each pigment goes into the mix.
    model.dose = pyo.Var(model.Pigment, domain=pyo.NonNegativeReals, bounds=dose_bounds)

    def vividness_rule(m):
        direct = sum(
            m.base_vividness[i] * m.dose[i] - m.saturation_penalty[i] * m.dose[i] ** 2
            for i in m.Pigment
        )
        blend = sum(
            m.blend_effect[i, j] * m.dose[i] * m.dose[j]
            for i in m.Pigment for j in m.Pigment if i < j
        )
        return direct + blend

    model.vividness = pyo.Objective(rule=vividness_rule, sense=pyo.maximize)
    return model


_PAINT_MODEL_TEMPLATE = _declare_paint_model()


def make_paint_model():
    """Instantiate a fresh concrete model from the abstract template + data."""
    return _PAINT_MODEL_TEMPLATE.create_instance(PAINT_DATA)


def main():
    result = optimize_allocation(
        build_model=make_paint_model,
        allocation_vars="dose",       # <- only this + build_model change between demos
        total=TOTAL_PAINT_MASS,
        maximize=True,
        first_pop_size=8,
        first_epochs=5,
        second_pop_size=8,
        second_epochs=5,
        max_search_rounds=5,
    )

    print("\n=== RESULT ===")
    print("Best vividness :", result.best_objective)
    print("Best pigment mix:", np.round(result.best_allocation, 3))
    print("Mix mass sum   :", round(float(np.sum(result.best_allocation)), 6))
    print("Local optima found:", len(result.stored_objectives))
    print("Solver calls:", result.run_stats["solver_calls"])


if __name__ == "__main__":
    main()
