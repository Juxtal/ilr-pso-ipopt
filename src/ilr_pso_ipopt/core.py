"""
Generic ILR-PSO + IPOPT optimizer for fixed-sum ("simplex") allocation
problems expressed as Pyomo models.

The algorithm does not know or care what the allocation variables
represent (enzyme amounts, budget shares, mixture ratios, ...). It only
assumes:

    x_i >= 0  for every allocation variable
    sum(x_i) == total

and repeatedly:

  1. Samples/perturbs candidate allocations on that simplex using an
     isometric log-ratio (ILR) transform, so PSO can search in an
     unconstrained (D-1)-dimensional space while every point it visits
     maps back to a feasible allocation.
  2. Uses each PSO-proposed allocation as the *initial guess* for a real
     NLP solve (IPOPT by default) of the caller's Pyomo model, and scores
     the candidate by the solved objective.
  3. Restarts from the best solutions found so far, perturbed in ILR
     space, to search for additional distinct local optima.

The caller supplies the Pyomo model (``build_model``) and tells the
optimizer which variables form the fixed-sum group (``allocation_vars``) -
usually just the Pyomo component name as a string, so pointing this at a
different model (or a renamed/re-indexed variable in the same model) is a
one-line change, not new adapter code; a callable is also accepted for
cases that need custom selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pyomo.environ as pyo
from mealpy import FloatVar
from mealpy.swarm_based import PSO


# ============================================================
# ILR transform (pure math; no model dependency)
# ============================================================


@lru_cache(maxsize=None)
def build_helmert_basis(dimension: int) -> np.ndarray:
    """Orthogonal Helmert basis used by the ILR transform: shape (D, D-1)."""
    if dimension < 2:
        raise ValueError("ILR requires at least two allocation components.")

    basis = np.zeros((dimension, dimension - 1), dtype=float)
    for column in range(1, dimension):
        basis[:column, column - 1] = 1.0 / np.sqrt(column * (column + 1))
        basis[column, column - 1] = -column / np.sqrt(column * (column + 1))
    return basis


def ilr_transform(z, epsilon: float = 1.0e-12) -> np.ndarray:
    """D-dimensional non-negative allocation -> (D-1)-dimensional ILR coordinates."""
    z = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(z)) or np.sum(z) <= 0.0:
        raise ValueError("z must be finite and have a positive sum.")

    positive_z = np.maximum(z, epsilon)
    composition = positive_z / np.sum(positive_z)
    basis = build_helmert_basis(len(z))
    return basis.T @ np.log(composition)


def ilr_inverse(y, total: float) -> np.ndarray:
    """(D-1)-dimensional ILR coordinates -> D-dimensional allocation summing to ``total``."""
    y = np.asarray(y, dtype=float)
    dimension = len(y) + 1
    basis = build_helmert_basis(dimension)

    log_composition = basis @ y
    composition = np.exp(log_composition - np.max(log_composition))
    composition /= np.sum(composition)
    return composition * total


# ============================================================
# Minimal, model-agnostic Pyomo solve helper
# ============================================================


def _solve(model, solver_name="ipopt", tee=False, options=None):
    solver = pyo.SolverFactory(solver_name)

    if solver_name == "ipopt":
        solver.options["max_iter"] = 5000
        solver.options["tol"] = 1e-6

    if options:
        for key, value in options.items():
            solver.options[key] = value

    return solver.solve(model, tee=tee, load_solutions=False)


def _solver_succeeded(results) -> bool:
    if results is None:
        return False
    return (
        results.solver.status == pyo.SolverStatus.ok
        and results.solver.termination_condition
        in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal}
    )


def _get_objective_value(model, objective_name: Optional[str] = None) -> float:
    if objective_name is not None:
        component = getattr(model, objective_name, None)
        if not isinstance(component, pyo.Objective):
            raise AttributeError(f"Model has no Objective named {objective_name!r}.")
        return float(pyo.value(component))

    objectives = list(model.component_data_objects(pyo.Objective, active=True))
    if not objectives:
        raise RuntimeError(
            "The model has no active Objective. Pass objective_name= if it is "
            "defined but not active by default."
        )
    if len(objectives) > 1:
        names = sorted({obj.parent_component().name for obj in objectives})
        raise RuntimeError(
            f"Model has multiple active Objectives ({', '.join(names)}); "
            "pass objective_name= to optimize_allocation to pick one."
        )
    return float(pyo.value(objectives[0]))


# ============================================================
# Generic allocation-vector helpers
# ============================================================


# What the caller may pass as `allocation_vars` to point at the group of
# fixed-sum variables:
#   - a Pyomo component name (str), e.g. "x" or "Enz_amount_rr" - every
#     VarData under it is used, in the component's own iteration order;
#   - a sequence of such names, concatenated in order (for a group spread
#     across several Var components);
#   - a callable(model) -> Sequence[Var] for anything more custom.
AllocationVarsSpec = Union[str, Sequence[str], Callable[["pyo.ConcreteModel"], Sequence["pyo.Var"]]]


def _resolve_allocation_vars(model, spec: AllocationVarsSpec):
    if callable(spec):
        return list(spec(model))

    names = [spec] if isinstance(spec, str) else list(spec)

    variables = []
    for name in names:
        component = getattr(model, name, None)
        if component is None:
            raise AttributeError(f"Model has no component named {name!r}.")
        if not isinstance(component, pyo.Var):
            raise TypeError(f"Component {name!r} is not a Pyomo Var (got {type(component).__name__}).")

        if component.is_indexed():
            variables.extend(component[index] for index in component.keys())
        else:
            variables.append(component)

    return variables


def _get_bounds(variables, total):
    lower_bounds, upper_bounds = [], []
    for variable in variables:
        lower_bounds.append(float(pyo.value(variable.lb)) if variable.has_lb() else 0.0)
        upper_bounds.append(float(pyo.value(variable.ub)) if variable.has_ub() else float(total))

    lb = np.asarray(lower_bounds, dtype=float)
    ub = np.asarray(upper_bounds, dtype=float)

    if np.any(lb > ub):
        raise ValueError("At least one allocation variable has lb > ub.")
    if total < np.sum(lb) - 1.0e-8:
        raise ValueError(f"total={total} is smaller than sum(lb)={np.sum(lb)}.")
    if total > np.sum(ub) + 1.0e-8:
        raise ValueError(f"total={total} is larger than sum(ub)={np.sum(ub)}.")

    return lb, ub


def _extract_vector(variables) -> np.ndarray:
    return np.asarray([pyo.value(v) for v in variables], dtype=float)


def _allocation_is_feasible(z, total, lb, ub, tolerance=1.0e-8) -> bool:
    z = np.asarray(z, dtype=float)
    return (
        len(z) == len(lb)
        and np.all(np.isfinite(z))
        and abs(np.sum(z) - total) <= tolerance
        and np.all(z >= lb - tolerance)
        and np.all(z <= ub + tolerance)
    )


def _set_initial_values(variables, z, total):
    z = np.asarray(z, dtype=float)
    if len(z) != len(variables):
        raise ValueError(f"len(z)={len(z)} does not match len(variables)={len(variables)}")
    if not np.isclose(np.sum(z), total, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"Allocation sums to {np.sum(z)}, expected {total}.")

    for value, variable in zip(z, variables):
        variable.set_value(float(value))
        if variable.fixed:
            variable.unfix()


# ============================================================
# Initial / local populations on the simplex
# ============================================================


def _sample_initial_allocations(total, population_size, ilr_bound, lb, ub, seed, alpha, max_attempts=100_000):
    dimension = len(lb)
    remaining = total - np.sum(lb)
    rng = np.random.default_rng(seed)
    samples = []

    if abs(remaining) <= 1.0e-12:
        candidate = lb.copy()
        y = ilr_transform(candidate)
        if np.any(np.abs(y) > ilr_bound + 1.0e-10):
            raise ValueError("The only feasible allocation lies outside the ILR bound.")
        return [candidate.copy() for _ in range(population_size)]

    attempts = 0
    while len(samples) < population_size and attempts < max_attempts:
        attempts += 1
        proportions = rng.dirichlet(alpha * np.ones(dimension))
        candidate = lb + remaining * proportions

        if not _allocation_is_feasible(candidate, total, lb, ub):
            continue

        y = ilr_transform(candidate)
        if np.any(np.abs(y) > ilr_bound + 1.0e-10):
            continue

        samples.append(candidate)

    if len(samples) < population_size:
        raise RuntimeError(
            "Unable to generate enough feasible ILR starting points. "
            "Try increasing ilr_bound, changing dirichlet_alpha, or checking the variable bounds."
        )
    return samples


def _build_local_starting_allocations(
    seed_solutions, total, population_size, ilr_bound, lb, ub, perturb_scale, seed, dirichlet_alpha,
):
    rng = np.random.default_rng(seed)
    base_coordinates = []
    starting_allocations = []

    for z_seed in seed_solutions:
        if not _allocation_is_feasible(z_seed, total, lb, ub, tolerance=1.0e-6):
            continue

        y_seed = np.clip(ilr_transform(z_seed), -ilr_bound, ilr_bound)
        z_seed_in_bounds = ilr_inverse(y_seed, total)

        if _allocation_is_feasible(z_seed_in_bounds, total, lb, ub):
            base_coordinates.append(y_seed)
            starting_allocations.append(z_seed_in_bounds)

        if len(starting_allocations) >= population_size:
            return starting_allocations[:population_size]

    if not base_coordinates:
        return _sample_initial_allocations(
            total, population_size, ilr_bound, lb, ub, seed, alpha=dirichlet_alpha,
        )

    attempts = 0
    while len(starting_allocations) < population_size and attempts < 10_000:
        attempts += 1
        base_y = base_coordinates[rng.integers(0, len(base_coordinates))]
        candidate_y = np.clip(
            base_y + rng.normal(0.0, perturb_scale, base_y.shape), -ilr_bound, ilr_bound,
        )
        candidate_z = ilr_inverse(candidate_y, total)

        if _allocation_is_feasible(candidate_z, total, lb, ub):
            starting_allocations.append(candidate_z)

    if len(starting_allocations) < population_size:
        missing = population_size - len(starting_allocations)
        starting_allocations.extend(
            _sample_initial_allocations(
                total, missing, ilr_bound, lb, ub, seed + 1, alpha=dirichlet_alpha,
            )
        )

    return starting_allocations[:population_size]


def _is_new_local_solution(z_new, objective_new, stored_z, stored_objectives, ilr_tolerance, objective_tolerance):
    y_new = ilr_transform(z_new)
    for z_old, objective_old in zip(stored_z, stored_objectives):
        y_old = ilr_transform(z_old)
        if (
            np.linalg.norm(y_new - y_old) < ilr_tolerance
            or abs(objective_new - objective_old) < objective_tolerance
        ):
            return False
    return True


# ============================================================
# One IPOPT solve from a given initial allocation
# ============================================================


def _solve_from_allocation(
    build_model, allocation_vars, z, total, solver_name, solver_options, stats,
    enforce_total_constraint=True, objective_name=None,
):
    model = build_model()
    variables = _resolve_allocation_vars(model, allocation_vars)
    lb, ub = _get_bounds(variables, total)
    z = np.asarray(z, dtype=float)

    if not _allocation_is_feasible(z, total, lb, ub):
        return None, None, None

    _set_initial_values(variables, z, total)

    if enforce_total_constraint:
        # The initial value alone does not keep IPOPT from drifting away
        # from `total` during the solve (the allocation variables are not
        # fixed) - add an explicit constraint so the polished solution
        # still honors the fixed sum, not just the starting point.
        model._ilr_pso_ipopt_total_constraint = pyo.Constraint(
            expr=sum(variables) == total
        )

    stats["solver_calls"] += 1
    results = _solve(model, solver_name=solver_name, tee=False, options=solver_options)

    if not _solver_succeeded(results):
        return None, None, None

    try:
        model.solutions.load_from(results)
        objective = _get_objective_value(model, objective_name)
        solved_z = _extract_vector(variables)
    except Exception:
        return None, None, None

    return model, objective, solved_z


# ============================================================
# ILR-PSO round
# ============================================================


def _make_objective_function(
    build_model, allocation_vars, total, lb, ub, maximize, penalty, solver_name, solver_options, stats,
    enforce_total_constraint, objective_name,
):
    sign = -1.0 if maximize else 1.0

    def objective_function(y):
        stats["ef_count"] += 1
        try:
            z = ilr_inverse(y, total)

            if not _allocation_is_feasible(z, total, lb, ub):
                lower_violation = np.maximum(lb - z, 0.0)
                upper_violation = np.maximum(z - ub, 0.0)
                violation = float(np.sum(lower_violation**2 + upper_violation**2))
                return penalty * (1.0 + violation)

            _, objective, _ = _solve_from_allocation(
                build_model, allocation_vars, z, total, solver_name, solver_options, stats,
                enforce_total_constraint, objective_name,
            )
            if objective is None:
                return penalty

            return sign * objective

        except Exception:
            return penalty

    return objective_function


def _run_ilr_pso(
    starting_allocations, total, epochs, population_size, ilr_bound, lb, ub,
    maximize, penalty, objective_function, pso_seed, round_id,
):
    if len(starting_allocations) != population_size:
        raise ValueError(
            f"Expected {population_size} starting solutions, received {len(starting_allocations)}."
        )

    dimension = len(lb)
    sign = -1.0 if maximize else 1.0

    starting_coordinates = []
    for z in starting_allocations:
        y = ilr_transform(z)
        if np.any(np.abs(y) > ilr_bound + 1.0e-10):
            raise ValueError(
                "A starting allocation lies outside the ILR bound. "
                "Increase ilr_bound or regenerate the population."
            )
        starting_coordinates.append(y)

    problem = {
        "bounds": FloatVar(lb=[-ilr_bound] * (dimension - 1), ub=[ilr_bound] * (dimension - 1)),
        "minmax": "min",
        "obj_func": objective_function,
    }

    optimizer = PSO.OriginalPSO(epoch=epochs, pop_size=population_size, verbose=False)
    best_agent = optimizer.solve(
        problem, starting_solutions=starting_coordinates, seed=pso_seed + round_id,
    )

    best_fitness = float(best_agent.target.fitness)
    if not np.isfinite(best_fitness) or best_fitness >= penalty:
        return None, None

    best_y = np.asarray(best_agent.solution, dtype=float)
    best_z = ilr_inverse(best_y, total)
    best_objective = sign * best_fitness

    return best_z, best_objective


# ============================================================
# Public API
# ============================================================


@dataclass
class OptimizationResult:
    """Result of :func:`optimize_allocation`."""

    best_model: object
    best_allocation: np.ndarray
    best_objective: float
    stored_allocations: list = field(default_factory=list)
    stored_objectives: list = field(default_factory=list)
    run_stats: dict = field(default_factory=dict)


def optimize_allocation(
    build_model: Callable[[], "pyo.ConcreteModel"],
    allocation_vars: AllocationVarsSpec,
    total: float,
    *,
    maximize: bool = True,
    ilr_bound: float = 6.0,
    penalty: float = 1.0e12,
    first_epochs: int = 10,
    first_pop_size: int = 20,
    first_sampling_seed: int = 1,
    dirichlet_alpha: float = 1.0,
    second_epochs: int = 10,
    second_pop_size: int = 20,
    local_ilr_perturb_scale: float = 0.25,
    max_search_rounds: int = 10,
    pso_seed: int = 42,
    ilr_solution_tol: float = 1.0e-4,
    objective_tol: float = 1.0e-4,
    solver_name: str = "ipopt",
    solver_options: Optional[dict] = None,
    enforce_total_constraint: bool = True,
    objective_name: Optional[str] = None,
    on_local_solution: Optional[Callable[[int, "pyo.ConcreteModel", float, np.ndarray], None]] = None,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize any "fixed-sum, non-negative" allocation in a Pyomo model.

    This works for *any* allocation-style model: the variables named by
    ``allocation_vars`` just need to satisfy ``x_i >= 0`` and
    ``sum(x_i) == total`` (enzyme amounts, budget shares, mixture ratios,
    ... — the algorithm has no notion of what they mean).

    Parameters
    ----------
    build_model:
        Called with no arguments to build a fresh Pyomo ``ConcreteModel``.
        Called once per candidate evaluation, so it must return an
        independent model instance each time.
    allocation_vars:
        Which variables in the model form the fixed-sum group. Usually
        just the Pyomo component's name, e.g. ``"x"`` or
        ``"Enz_amount_rr"`` — every ``VarData`` under it is used
        (indexed or scalar). This is the one piece of glue code a new
        model needs, and with a plain name it is a one-line change: point
        it at a different model, or a renamed/re-indexed variable in the
        same model, without touching any other code. Also accepts:

        * a sequence of names, concatenated in order (an allocation group
          spread across several Var components); or
        * a callable ``(model) -> Sequence[Var]`` for anything that a
          name can't express (e.g. a filtered subset of an indexed Var).
    enforce_total_constraint:
        If True (default), an explicit ``sum(allocation_vars) == total``
        constraint is added to every model before solving, so the
        IPOPT-polished solution still honors ``total`` (the allocation
        variables are only *seeded* with the ILR-proposed values, not
        fixed, so without this constraint the solver is free to drift
        away from the target sum). Set to False only if your own model
        already enforces this sum itself.
    total:
        The fixed sum the allocation variables must add up to (e.g. total
        enzyme budget, total investment, ...).
    maximize:
        Whether the model's active ``Objective`` should be maximized
        (default) or minimized.
    objective_name:
        Name of the ``Objective`` component to read after each solve.
        Only needed if the model defines more than one active Objective
        (otherwise the single active one is used automatically) or one
        that is not active by default.
    on_local_solution:
        Optional callback ``(round_id, model, objective, allocation)``
        invoked every time a new distinct local optimum is found — use it
        to export/plot results however your application needs; the
        optimizer itself does not write any files.

    Returns
    -------
    OptimizationResult
    """

    def log(*parts):
        if verbose:
            print(*parts)

    run_stats = {"solver_calls": 0, "ef_count": 0}

    template_model = build_model()
    variables = _resolve_allocation_vars(template_model, allocation_vars)
    lb, ub = _get_bounds(variables, total)

    objective_function = _make_objective_function(
        build_model, allocation_vars, total, lb, ub, maximize, penalty,
        solver_name, solver_options, run_stats, enforce_total_constraint, objective_name,
    )

    # --------------------------------------------------------
    # Round 1: global Dirichlet initialization + ILR-PSO
    # --------------------------------------------------------
    log("=== STEP 1: first global ILR-PSO round ===")

    first_population = _sample_initial_allocations(
        total, first_pop_size, ilr_bound, lb, ub, first_sampling_seed, alpha=dirichlet_alpha,
    )

    candidate_z, _ = _run_ilr_pso(
        first_population, total, first_epochs, first_pop_size, ilr_bound, lb, ub,
        maximize, penalty, objective_function, pso_seed, round_id=1,
    )
    if candidate_z is None:
        raise RuntimeError("The first ILR-PSO round did not find a valid solution.")

    model, objective, solution = _solve_from_allocation(
        build_model, allocation_vars, candidate_z, total, solver_name, solver_options, run_stats,
        enforce_total_constraint, objective_name,
    )
    if model is None:
        raise RuntimeError("First-round IPOPT polish failed. Cannot continue.")

    stored_allocations = [solution.copy()]
    stored_objectives = [objective]
    best_model, best_allocation, best_objective = model, solution.copy(), objective

    if on_local_solution is not None:
        on_local_solution(1, model, objective, solution)

    # --------------------------------------------------------
    # Later rounds: perturb stored local solutions in ILR space
    # --------------------------------------------------------
    log("=== STEP 2: local ILR-PSO rounds ===")

    for round_id in range(2, max_search_rounds + 1):
        order = np.argsort(stored_objectives)
        order = order[::-1] if maximize else order
        ordered_solutions = [stored_allocations[i] for i in order]

        local_population = _build_local_starting_allocations(
            ordered_solutions, total, second_pop_size, ilr_bound, lb, ub,
            local_ilr_perturb_scale, seed=1000 + round_id, dirichlet_alpha=dirichlet_alpha,
        )

        candidate_z, _ = _run_ilr_pso(
            local_population, total, second_epochs, second_pop_size, ilr_bound, lb, ub,
            maximize, penalty, objective_function, pso_seed, round_id=round_id,
        )
        if candidate_z is None:
            log(f"Round {round_id}: no valid PSO candidate found.")
            continue

        model, objective, solution = _solve_from_allocation(
            build_model, allocation_vars, candidate_z, total, solver_name, solver_options, run_stats,
            enforce_total_constraint, objective_name,
        )
        if model is None:
            log(f"Round {round_id}: IPOPT polish failed.")
            continue

        if not _is_new_local_solution(
            solution, objective, stored_allocations, stored_objectives, ilr_solution_tol, objective_tol,
        ):
            log(f"Round {round_id}: no new local solution found, stopping.")
            break

        stored_allocations.append(solution.copy())
        stored_objectives.append(objective)

        if on_local_solution is not None:
            on_local_solution(round_id, model, objective, solution)

        improved = objective > best_objective if maximize else objective < best_objective
        if improved:
            best_model, best_allocation, best_objective = model, solution.copy(), objective

    log("=== DONE ===")
    log(f"Local optima found: {len(stored_allocations)}")
    log(f"Best objective: {best_objective}")

    return OptimizationResult(
        best_model=best_model,
        best_allocation=best_allocation,
        best_objective=best_objective,
        stored_allocations=stored_allocations,
        stored_objectives=stored_objectives,
        run_stats=run_stats,
    )
