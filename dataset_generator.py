
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File name: dataset_generator.py
# Author: AM 2914 D
# Date created: 2025-12-27
# Version = "1.0"
# License =  "CC0 1.0"
# Listening = "Piano Concerto No. 2 in C minor, Op. 18 - Sergei Rachmaninoff (1901)"
# =============================================================================
"""Run many experiments sweeping parameters of the segregation simulation"""
# =============================================================================

"""
Example usage:
    python dataset_generator.py --out results.csv

Notes:
    - Check that the simulation code (Agent, run_simulation helpers, etc.) from
      the provided script is available in the same module or imported here.
    - This script runs headless (no plotting) and uses the same logic for
      neighbor/cluster/zones computations as the simulation functions.
"""

from segregation_sim import init_agents, build_neighbor_edges, sensor_beep_states, count_groups, compute_influence_zones, compute_clusters, AREA_SIZE, MAX_STEPS, STEP_SIZE, RANDOM_TURN_SD, ZONES_ALPHA_SCALE

import argparse
import csv
import json
import itertools
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy.spatial import cKDTree

# --- Helper: run a single experiment and collect per-step data ---


def run_experiment_and_collect(
    experiment_id: int,
    N: int,
    k: int,
    frac_orange: float,
    area_x: float,
    area_y: Optional[float],
    max_steps: int,
    step_size: float,
    random_turn_sd: float,
    seed: int,
    raster_zones_res: int = 150,
    zones_alpha_scale: float = 0.2,
    cluster_dist: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Run a single headless simulation and collect per-step, per-agent data rows.

    Parameters:
        experiment_id: Unique integer id for this experiment.
        N: Number of agents.
        k: Number of nearest neighbors used for sensing/links.
        frac_orange: Fraction of agents assigned the "orange" color (rest are "blue").
        area_x: Width of the rectangular area.
        area_y: Height of the rectangular area. If None, assumed equal to area_x.
        max_steps: Maximum number of timesteps to execute.
        step_size: Distance agents move per step (passed to Agent.step).
        random_turn_sd: Standard deviation for random heading changes (passed to Agent.step).
        seed: RNG seed for deterministic initialization and progression.
        raster_zones_res: Resolution (pixels per side) for rasterizing influence zones.
        zones_alpha_scale: Alpha scaling factor passed to compute_influence_zones.
        cluster_dist: Distance threshold used by compute_clusters to merge nearby clusters.

    Returns:
        A list of dictionaries, each dictionary representing one agent at one step.
        Each dict contains experiment metadata, agent state (x, y, color, moving, homophilic),
        neighbor links, cluster id and members, optional cluster radius, zone id and zone members,
        and the executed experiment_length.
    """

    if area_y is None:
        area_y = area_x

    np.random.seed(seed)

    # Initialize agents with provided parameters (use init_agents but allow
    # square area only; if non-square, we adjust by scaling positions after init)
    agents = init_agents(N=N, frac_orange=frac_orange,
                         area_size=max(area_x, area_y))
    # If area_x != area_y, remap y coordinate to [0, area_y]
    if area_x != area_y:
        # init_agents used area = max(area_x, area_y), so rescale to requested area
        max_area = max(area_x, area_y)
        for a in agents:
            a.pos[0] = np.clip(a.pos[0], 0.0, max_area) * (area_x / max_area)
            a.pos[1] = np.clip(a.pos[1], 0.0, max_area) * (area_y / max_area)

    # override global step size and random turn sd per-agent step call by passing args
    rows: List[Dict[str, Any]] = []
    step = 0
    actual_steps = 0

    while step < max_steps:
        step += 1
        # compute beeps and set moving
        beeps = sensor_beep_states(agents, k)
        for i, a in enumerate(agents):
            a.moving = bool(beeps[i])

        # record pre-step state (position, moving, homophilic, links, clusters, zones)
        pts = np.array([a.pos for a in agents])
        colors_arr = np.array([a.color for a in agents], dtype=int)

        # Build neighbor edges for this step (use k-NN on current positions)
        edges = build_neighbor_edges(agents, k=k, max_dist=None)

        # For each agent, compute its neighbor list (from edges)
        neighbor_map = {i: [] for i in range(len(agents))}
        for i, j in edges:
            neighbor_map[i].append(j)
            neighbor_map[j].append(i)

        # compute homophilic flag per agent: true if all k neighbors (up to k) same color
        homophilic = np.zeros(len(agents), dtype=bool)
        if len(agents) > 0:
            tree = cKDTree(pts)
            kk = min(k + 1, len(agents))
            _, idxs = tree.query(pts, kk)
            for i in range(len(agents)):
                nbrs = [j for j in idxs[i] if j != i][:k]
                if len(nbrs) == 0:
                    homophilic[i] = False
                else:
                    homophilic[i] = bool(
                        np.all(colors_arr[nbrs] == colors_arr[i]))

        # clusters (connected components) and cluster descriptors
        _, _, clusters_map, homophilic_clusters = count_groups(
            agents, k=k, max_dist=None)
        # compute compute_clusters descriptors for visualization (gives radii and merged groups)
        n_merged, merged_descr = compute_clusters(
            agents, cluster_dist=cluster_dist)

        # build cluster_id mapping: map each agent to its cluster root id (or -1)
        cluster_id_map = {i: -1 for i in range(len(agents))}
        cluster_members_map: Dict[int, List[int]] = {}
        for root, members in clusters_map.items():
            for m in members:
                cluster_id_map[m] = int(root)
            cluster_members_map[int(root)] = list(members)

        # zones: compute influence zones raster and connected components (we want zone labels per pixel,
        # and map pixels to nearest agent; then produce zone membership: for each zone id, which agent ids)
        rgba, total_zones, zone_label_map = compute_influence_zones(
            agents, area_size=max(area_x, area_y), res=raster_zones_res, alpha_scale=zones_alpha_scale
        )
        # zone_label_map contains labels starting at 1 per-color accumulation; zero = no zone
        # compute for each zone label the set of agent ids that own cells in that zone.
        zone_members_map: Dict[int, List[int]] = {}
        if total_zones > 0:
            # For each raster cell, determine assigned agent id (nearest)
            res = zone_label_map.shape[0]
            ys = np.linspace(max(area_y, area_x)/(2*res),
                             max(area_y, area_x) - max(area_y, area_x)/(2*res), res)
            xs = np.linspace(max(area_x, area_y)/(2*res),
                             max(area_x, area_y) - max(area_x, area_y)/(2*res), res)
            gx, gy = np.meshgrid(xs, ys)
            grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
            tree = cKDTree(pts)
            _, idxs = tree.query(grid_pts, k=1)
            assigned = idxs.reshape((res, res))
            for lbl in np.unique(zone_label_map):
                if lbl == 0:
                    continue
                # cells with this label
                mask = (zone_label_map == lbl)
                if not mask.any():
                    continue
                agents_in_zone = np.unique(assigned[mask])
                zone_members_map[int(lbl)] = [int(aid)
                                              for aid in agents_in_zone]

        # prepare rows for each agent at this step
        for i, a in enumerate(agents):
            row = {
                "experiment_id": experiment_id,
                "step": step,
                "N": N,
                "k": k,
                "frac_orange": frac_orange,
                "area_x": area_x,
                "area_y": area_y,
                "max_steps": max_steps,
                "step_size": step_size,
                "random_turn_sd": random_turn_sd,
                "seed": seed,
                "dot_id": int(i),
                "x": float(a.pos[0]),
                "y": float(a.pos[1]),
                "color": int(a.color),
                "is_moving": bool(a.moving),
                "is_homophilic": bool(homophilic[i]),
                "links": json.dumps(neighbor_map.get(i, [])),
                "cluster_id": int(cluster_id_map.get(i, -1)),
                "cluster_members": json.dumps(cluster_members_map.get(cluster_id_map.get(i, -1), [])),
                # cluster radius: if we can find merged circle describing this agent's cluster, else None
                "cluster_radius": None,
                "zone_id": None,
                "zone_members": json.dumps([]),
            }
            # try to get cluster radius from merged_descr: merged_descr keys are (root,color)
            # merged_descr values are lists of circle descriptors having 'members','center','radius'
            try:
                cid = cluster_id_map.get(i, -1)
                if cid != -1:
                    # look in merged_descr for any key whose group contains this agent
                    found_radius = None
                    for key, sublist in merged_descr.items():
                        for circ in sublist:
                            if i in circ.get("members", []):
                                found_radius = float(circ.get("radius", 0.0))
                                break
                        if found_radius is not None:
                            break
                    row["cluster_radius"] = found_radius
            except Exception:
                row["cluster_radius"] = None

            # zone membership: find any zone label that contains this agent (via zone_members_map)
            for zlbl, members in zone_members_map.items():
                if i in members:
                    row["zone_id"] = int(zlbl)
                    row["zone_members"] = json.dumps(members)
                    break

            rows.append(row)

        # Advance agents by one step using provided step_size and random_turn_sd
        for a in agents:
            # use Agent.step signature
            a.step(step_size=step_size, random_turn_sd=random_turn_sd,
                   area=max(area_x, area_y))

        # If all agents silent (not moving) break
        if all(not a.moving for a in agents):
            actual_steps = step
            break

    if actual_steps == 0:
        actual_steps = step

    # Add experiment length to each row (number of steps executed)
    for r in rows:
        r["experiment_length"] = actual_steps

    return rows


# --- Parameter sweep utilities ---
def param_product(params: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """
    Produce the Cartesian product of sweep parameters.

    Parameters:
        params: Mapping from parameter name to an iterable of candidate values.

    Returns:
        A list of dictionaries; each dictionary is one combination mapping parameter
        names to scalar values.
    """

    keys = list(params.keys())
    vals = [list(params[k]) for k in keys]
    combos = []
    for prod in itertools.product(*vals):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


# --- Main entrypoint for dataset generation ---
def generate_dataset(
    out_csv: str,
    sweep_params: Dict[str, Iterable[Any]],
    seeds: Iterable[int] = (1,),
    raster_zones_res: int = 150,
    zones_alpha_scale: float = 0.2,
    cluster_dist: float = 1.0,
    step_size_override: Optional[float] = None,
    random_turn_sd_override: Optional[float] = None,
) -> None:
    """
    Sweep parameters, run simulations, and write a per-agent-per-step CSV.

    Parameters:
        out_csv: Path to the output CSV file to write.
        sweep_params: Dict specifying parameter names and iterables of values to sweep.
            Supported keys: N, k, frac_orange, area_x, area_y, max_steps, step_size, random_turn_sd.
            Each value may be a scalar or an iterable.
        seeds: Iterable of integer RNG seeds to run for each parameter combination.
        raster_zones_res: Resolution used when rasterizing influence zones.
        zones_alpha_scale: Alpha scaling factor passed to compute_influence_zones.
        cluster_dist: Cluster merging distance passed to compute_clusters.
        step_size_override: If provided, used as the default step_size when not present in sweep_params.
        random_turn_sd_override: If provided, used as default random_turn_sd when not present.

    Side effects:
        Writes the CSV specified by out_csv. Prints a completion message on success.
    """

    # Normalize sweep params values to iterables
    norm_params = {}
    for k, v in sweep_params.items():
        if isinstance(v, (list, tuple, set, np.ndarray)):
            norm_params[k] = list(v)
        else:
            norm_params[k] = [v]

    combos = param_product(norm_params)
    fieldnames = [
        "experiment_id",
        "step",
        "experiment_length",
        "N",
        "k",
        "frac_orange",
        "area_x",
        "area_y",
        "max_steps",
        "step_size",
        "random_turn_sd",
        "seed",
        "dot_id",
        "x",
        "y",
        "color",
        "is_moving",
        "is_homophilic",
        "links",
        "cluster_id",
        "cluster_members",
        "cluster_radius",
        "zone_id",
        "zone_members",
    ]

    exp_id = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for combo in combos:
            for seed in seeds:
                exp_id += 1

                def _scalar_from_val(v, default):
                    # if v is an iterable (range, list) but not a str, take first element;
                    # if v is None, return default
                    if v is None:
                        return default
                    if isinstance(v, str):
                        return v
                    try:
                        iter(v)
                    except TypeError:
                        return v
                    # it's iterable — take first element
                    try:
                        return list(v)[0]
                    except Exception:
                        return default

                N = int(_scalar_from_val(combo.get("N"), 22))
                k = int(_scalar_from_val(combo.get("k"), 8))
                frac_orange = float(_scalar_from_val(
                    combo.get("frac_orange"), 0.5))
                area_x = float(_scalar_from_val(
                    combo.get("area_x"), AREA_SIZE))
                area_y = float(_scalar_from_val(combo.get("area_y"), area_x))
                max_steps = int(_scalar_from_val(
                    combo.get("max_steps"), MAX_STEPS))
                step_size = float(_scalar_from_val(
                    combo.get("step_size"), step_size_override or STEP_SIZE))
                random_turn_sd = float(_scalar_from_val(
                    combo.get("random_turn_sd"), random_turn_sd_override or RANDOM_TURN_SD))

                # assign parameters, using defaults if missing
                rows = run_experiment_and_collect(
                    experiment_id=exp_id,
                    N=N,
                    k=k,
                    frac_orange=frac_orange,
                    area_x=area_x,
                    area_y=area_y,
                    max_steps=max_steps,
                    step_size=step_size,
                    random_turn_sd=random_turn_sd,
                    seed=int(seed),
                    raster_zones_res=raster_zones_res,
                    zones_alpha_scale=zones_alpha_scale,
                    cluster_dist=cluster_dist,
                )

                # write rows
                for r in rows:
                    # ensure all fieldnames exist (fill missing with empty)
                    out_row = {fn: r.get(fn, "") for fn in fieldnames}
                    writer.writerow(out_row)

    print(f"Finished writing dataset to {out_csv}")


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CSV dataset by sweeping simulation params.")
    parser.add_argument("--out", type=str,
                        default="dataset.csv", help="output CSV file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1], help="list of RNG seeds to run")
    parser.add_argument("--raster-res", type=int,
                        default=150, help="raster zones resolution")
    parser.add_argument("--cluster-dist", type=float,
                        default=1.0, help="cluster dist for merged circles")
    parser.add_argument("--max-rows", type=int,
                        default=None, help="(unused) stop early")
    args = parser.parse_args()

    # Example sweep: you can edit this block to specify the parameter grid to sweep.
    sweep = {
        "N": [10, 22],
        "k": [3, 8],
        "frac_orange": [0.25, 0.5, 0.75],
        "area_x": range(1, 4),
        # "area_y" can be included to run rectangular areas
        "max_steps": [100],
        "step_size": [0.25],
        "random_turn_sd": [0.7],
    }

    generate_dataset(
        out_csv=args.out,
        sweep_params=sweep,
        seeds=args.seeds,
        raster_zones_res=args.raster_res,
        zones_alpha_scale=ZONES_ALPHA_SCALE,
        cluster_dist=args.cluster_dist,
    )
