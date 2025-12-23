#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File name: segregation_sim.py
# Author: INV 802 / JJW 50
# Date created: 2025-12-21
# Version = "1.0"
# License =  "CC0 1.0"
# Listening = "Trond Kallevåg - Twins of Træna (2025)"
# =============================================================================
""" Simple agent-based spatial simulation of group dynamics with hidden feedback rules"""
# =============================================================================


from __future__ import annotations
import argparse
from typing import List, Tuple, Dict, Set, Any, Optional
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mcoll
from matplotlib.animation import FuncAnimation
import mplcyberpunk

# -------------------------
# Defaults / parameters
# -------------------------
N: int = 22
FRAC_ORANGE: float = 0.5
AREA_SIZE: float = 10.0
K: int = 8
MAX_STEPS: int = 300
STEP_SIZE: float = 0.25
RANDOM_TURN_SD: float = 0.7
STOP_WHEN_SILENT: bool = True
SEED: int = 1

DETAIL_MODE: bool = False
NEIGHBOR_LINE_DIST: float = 2.0
GLOW_LINE_WIDTH: float = 2.2
NONMATCH_LINE_WIDTH: float = 1.0

RASTER_ZONES: bool = True
RASTER_ZONES_RES: int = 150
ZONES_ALPHA_SCALE: float = 0.20

CLUSTER_CIRCLES: bool = False
CLUSTER_DIST: float = 1.0
CLUSTER_CIRCLE_ALPHA: float = 0.08

np.random.seed(SEED)

# color map constants
ORANGE_RGB: np.ndarray = np.array([1.0, 0.549, 0.0])
BLUE_RGB: np.ndarray = np.array([0.0, 0.753, 1.0])
COLOR_HEX: Dict[int, str] = {1: "#FF8C00", 0: "#00C0FF"}


# -------------------------
# Agent and motion
# -------------------------
class Agent:
    """A simple 2D agent with position, color, heading, and moving state.

    Attributes:
        pos: 2-element numpy array giving x,y position.
        color: int flag for color (1 = orange, 0 = blue).
        moving: whether agent is currently moving (bool).
        heading: unit 2D numpy array indicating direction of travel.
    """

    pos: np.ndarray
    color: int
    moving: bool
    heading: np.ndarray

    def __init__(self, pos: Tuple[float, float], color: int) -> None:
        """Initialize an Agent.

        Args:
            pos: (x, y) initial position.
            color: 1 for orange, 0 for blue.
        """
        self.pos = np.array(pos, dtype=float)
        self.color = int(color)  # 1=orange,0=blue
        self.moving = True
        theta = np.random.uniform(0, 2 * np.pi)
        self.heading = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    def step(self,
             step_size: float = STEP_SIZE,
             random_turn_sd: float = RANDOM_TURN_SD,
             area: float = AREA_SIZE) -> None:
        """Advance the agent one time step.

        The agent applies a small random rotation to its heading, normalizes it,
        moves forward by step_size, and reflects off square area boundaries.

        Args:
            step_size: distance moved along heading.
            random_turn_sd: standard deviation of small random rotation (radians).
            area: side length of the square domain [0, area] x [0, area].
        """
        if not self.moving:
            return
        # small random rotation
        ang = np.random.normal(0.0, random_turn_sd)
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]])
        self.heading = R.dot(self.heading)
        self.heading /= (np.linalg.norm(self.heading) + 1e-12)
        self.pos += self.heading * step_size
        # reflect at boundaries
        for i in (0, 1):
            if self.pos[i] < 0:
                self.pos[i] = -self.pos[i]
                self.heading[i] = -self.heading[i]
            elif self.pos[i] > area:
                self.pos[i] = 2 * area - self.pos[i]
                self.heading[i] = -self.heading[i]


# -------------------------
# Initialization
# -------------------------
def init_agents(N: int = N,
                frac_orange: float = FRAC_ORANGE,
                area_size: float = AREA_SIZE) -> List[Agent]:
    """Create a list of Agents with random positions and colors.

    Args:
        N: number of agents.
        frac_orange: fraction assigned color 1 (orange).
        area_size: side length of sampling area.

    Returns:
        List[Agent]: initialized agents.
    """
    colors = np.zeros(N, dtype=int)
    n_orange = int(round(N * frac_orange))
    colors[:n_orange] = 1
    np.random.shuffle(colors)
    positions = np.random.uniform(0.0, area_size, size=(N, 2))
    return [Agent(positions[i], int(colors[i])) for i in range(N)]


# -------------------------
# Neighbor & sensor logic
# -------------------------
def build_neighbor_edges(agents: List[Agent],
                         k: Optional[int] = None,
                         max_dist: Optional[float] = None) -> List[Tuple[int, int]]:
    """Build neighbor edges for agents using either k-NN or distance threshold.

    Args:
        agents: list of Agent objects.
        k: number of nearest neighbors to connect (if provided).
        max_dist: maximum distance to connect two agents (used when k is None).

    Returns:
        Sorted list of unique undirected edges as (i, j) index tuples.
    """
    pts = np.array([a.pos for a in agents])
    N = len(agents)
    tree = cKDTree(pts)
    edges: Set[Tuple[int, int]] = set()
    if k is not None:
        kk = min(k + 1, N)
        _, idxs = tree.query(pts, kk)
        for i in range(N):
            nbrs = [j for j in idxs[i] if j != i][:k]
            for j in nbrs:
                edges.add((min(i, int(j)), max(i, int(j))))
    else:
        if max_dist is None:
            raise ValueError("max_dist required when k is None")
        for i, j in tree.query_pairs(float(max_dist)):
            edges.add((i, j))
    return sorted(edges)


def sensor_beep_states(agents: List[Agent], k: int) -> np.ndarray:
    """Compute boolean 'beep' states: agent beeps if a strict majority of its k neighbors are opposite color.

    For each agent, consider up to k nearest neighbors (excluding self). An agent
    beeps (returns True) if the number of neighbors with a different color is
    >= (floor(len(nbrs)/2) + 1).

    Args:
        agents: list of Agent objects.
        k: number of neighbors to consider per agent.

    Returns:
        N-length boolean numpy array indicating beep state per agent.
    """
    N = len(agents)
    if N == 0:
        return np.array([], dtype=bool)
    pts = np.array([a.pos for a in agents])
    tree = cKDTree(pts)
    kk = min(k + 1, N)
    dists, idxs = tree.query(pts, kk)
    beeps = np.zeros(N, dtype=bool)
    for i in range(N):
        nbrs = [j for j in idxs[i] if j != i][:k]
        if len(nbrs) == 0:
            beeps[i] = False
            continue
        opp = sum(1 for j in nbrs if agents[j].color != agents[i].color)
        threshold = (len(nbrs) // 2) + 1
        beeps[i] = (opp >= threshold)
    return beeps


# -------------------------
# Clustering / connectivity
# -------------------------
def _union_find_connect_components(N: int,
                                   edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Union-find to compute connected components from undirected edges.

    Args:
        N: number of nodes (0..N-1).
        edges: list of undirected edges (i, j).

    Returns:
        Dict mapping root representative -> list of member indices.
    """
    parent = list(range(N))
    rank = [0] * N

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in edges:
        union(a, b)
    clusters: Dict[int, List[int]] = {}
    for i in range(N):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return clusters


def count_groups(agents: List[Agent],
                 k: Optional[int] = None,
                 max_dist: Optional[float] = None) -> Tuple[int, int, Dict[int, List[int]], Dict[int, List[int]]]:
    """Count connected groups and homophilic groups among agents.

    Args:
        agents: list of Agent objects.
        k: if provided, build k-NN graph; otherwise use max_dist threshold.
        max_dist: distance threshold used when k is None.

    Returns:
        Tuple (n_clusters, n_homophilic, clusters, homophilic)
          - n_clusters: total number of connected components
          - n_homophilic: number of components where all members share the same color
          - clusters: mapping root -> list of member indices for all components
          - homophilic: mapping root -> members for homophilic components only
    """
    N = len(agents)
    if N == 0:
        return 0, 0, {}, {}
    if k is None and max_dist is None:
        max_dist = NEIGHBOR_LINE_DIST
    edges = build_neighbor_edges(agents, k=k, max_dist=max_dist)
    clusters = _union_find_connect_components(N, edges)
    colors = np.array([a.color for a in agents])
    homophilic: Dict[int, List[int]] = {}
    for root, members in clusters.items():
        if np.all(colors[members] == colors[members][0]):
            homophilic[root] = members
    return len(clusters), len(homophilic), clusters, homophilic


# -------------------------
# Influence zone / raster computations
# -------------------------
def compute_influence_zones(agents: List[Agent],
                            area_size: float = AREA_SIZE,
                            res: int = RASTER_ZONES_RES,
                            alpha_scale: float = ZONES_ALPHA_SCALE) -> Tuple[np.ndarray, int, np.ndarray]:
    """Rasterize agent 'influence' across a grid by nearest-agent Voronoi assignment.

    Each grid cell is assigned the color of its nearest agent; the alpha channel
    encodes inverse distance to that nearest agent scaled by alpha_scale.

    Args:
        agents: list of Agent objects.
        area_size: side length of domain.
        res: output raster resolution (res x res).
        alpha_scale: scale factor for alpha channel.

    Returns:
        A tuple (rgba, total_zones, zone_label_map) where:
          - rgba: (res, res, 4) float array with RGB set to agent color and A set per-cell
          - total_zones: total number of connected color zones across both colors
          - zone_label_map: (res, res) integer labels for connected components
    """
    N = len(agents)
    if N == 0:
        return np.zeros((res, res, 4)), 0, np.zeros((res, res), dtype=int)
    pts = np.array([a.pos for a in agents])
    colors = np.array([a.color for a in agents])
    ys = np.linspace(area_size/(2*res), area_size - area_size/(2*res), res)
    xs = np.linspace(area_size/(2*res), area_size - area_size/(2*res), res)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    tree = cKDTree(pts)
    dists, idxs = tree.query(grid_pts, k=1)
    assigned = idxs.reshape((res, res))
    assigned_color = colors[assigned]
    maxd = dists.max() if dists.size else 1.0
    alpha = (1.0 - (dists / (maxd + 1e-9))) * alpha_scale
    alpha = alpha.reshape((res, res))
    rgba = np.zeros((res, res, 4), dtype=float)
    rgba[..., :3] = np.where(assigned_color[..., None]
                             == 1, ORANGE_RGB, BLUE_RGB)
    rgba[..., 3] = alpha
    # connected components per color (4-connectivity)
    zone_label_map = np.zeros((res, res), dtype=int)
    total_zones = 0
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    for col in (0, 1):
        mask = (assigned_color == col)
        if mask.any():
            lbl, ncomp = label(mask, structure=struct)
            if ncomp > 0:
                zone_label_map[mask] = lbl[mask] + total_zones
                total_zones += ncomp
    return rgba, total_zones, zone_label_map


def compute_clusters(agents: List[Agent],
                     cluster_dist: float = CLUSTER_DIST) -> Tuple[int, Dict[Tuple[int, int], List[Dict]]]:
    """Compute merged same-color cluster circles for visualization.

    Steps:
      - Find same-color pairs within cluster_dist and union them into small clusters.
      - For each cluster, compute a circle center (mean) and radius (max distance to center + cluster_dist).
      - Merge circles of the same color that intersect and produce merged descriptors.

    Args:
        agents: list of Agent objects.
        cluster_dist: distance threshold to consider agents in same proximate cluster.

    Returns:
        Tuple (n_merged, merged) where:
          - n_merged: number of merged circle groups
          - merged: dict keyed by (root_index, color) mapping to list of circle descriptors
    """
    N = len(agents)
    if N == 0:
        return 0, {}
    pts = np.array([a.pos for a in agents])
    colors = np.array([a.color for a in agents])
    tree = cKDTree(pts)
    pairs = tree.query_pairs(float(cluster_dist))
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        if colors[i] == colors[j]:
            union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(N):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    circle_list: List[Dict] = []
    for members in clusters.values():
        col = int(colors[members[0]])
        member_pts = pts[members]
        center = member_pts.mean(axis=0)
        maxd = np.max(np.linalg.norm(member_pts - center, axis=1))
        radius = float(maxd + cluster_dist)
        circle_list.append({'members': members, 'color': col,
                           'center': center, 'radius': radius})

    M = len(circle_list)
    if M == 0:
        return 0, {}

    cparent = list(range(M))

    def cfind(x: int) -> int:
        while cparent[x] != x:
            cparent[x] = cparent[cparent[x]]
            x = cparent[x]
        return x

    def cunion(a: int, b: int) -> None:
        ra, rb = cfind(a), cfind(b)
        if ra != rb:
            cparent[rb] = ra

    for i in range(M):
        for j in range(i+1, M):
            if circle_list[i]['color'] != circle_list[j]['color']:
                continue
            if np.linalg.norm(circle_list[i]['center'] - circle_list[j]['center']) <= (circle_list[i]['radius'] + circle_list[j]['radius']):
                cunion(i, j)

    merged: Dict[Tuple[int, int], List[Dict]] = {}
    for idx in range(M):
        root = cfind(idx)
        key = (root, circle_list[idx]['color'])
        merged.setdefault(key, []).append(circle_list[idx])

    return len(merged), merged


# -------------------------
# Drawing helpers
# -------------------------
def draw_cluster_circles_ax(ax: plt.Axes,
                            merged_clusters: Dict[Tuple[int, int], List[Dict[str, Any]]],
                            alpha: float = CLUSTER_CIRCLE_ALPHA) -> None:
    """Draw merged cluster influence circles onto an Axes.

    Existing stored patches on the axes (attribute _infl_circles) are removed
    before drawing new ones.

    Args:
        ax: Matplotlib Axes to draw on.
        merged_clusters: dict mapping keys to lists of cluster descriptors having
                         'center', 'radius', and 'members'.
        alpha: fill alpha for the circles.
    """
    for c in getattr(ax, "_infl_circles", []):
        try:
            c.remove()
        except Exception:
            pass
    ax._infl_circles = []
    for key, subclusters in merged_clusters.items():
        centers = np.array([sc['center'] for sc in subclusters])
        sizes = np.array([len(sc['members']) for sc in subclusters])
        wcenter = (centers * sizes[:, None]).sum(axis=0) / sizes.sum()
        maxr = 0.0
        for sc in subclusters:
            dist = np.linalg.norm(sc['center'] - wcenter) + sc['radius']
            if dist > maxr:
                maxr = dist
        circ = patches.Circle(
            wcenter, maxr, color=COLOR_HEX[key[1]], alpha=alpha, linewidth=1, fill=True, zorder=1)
        ax.add_patch(circ)
        ax._infl_circles.append(circ)


def make_neighbor_line_collections(ax: plt.Axes) -> Tuple[mcoll.LineCollection, mcoll.LineCollection]:
    """Create and add LineCollections for same-color and different-color neighbor edges.

    Args:
        ax: Matplotlib Axes.

    Returns:
        Tuple (same_color_collection, diff_color_collection).
    """
    same = mcoll.LineCollection(
        [], linewidths=GLOW_LINE_WIDTH, colors=[], zorder=3, alpha=0.45)
    diff = mcoll.LineCollection([], linewidths=NONMATCH_LINE_WIDTH,
                                colors='#BBBBBB', zorder=3, alpha=0.6, linestyles='dashed')
    ax.add_collection(same)
    ax.add_collection(diff)
    return same, diff


def update_neighbor_lines_from_agents(same_col: mcoll.LineCollection,
                                      diff_col: mcoll.LineCollection,
                                      agents: List[Agent],
                                      k: int = K,
                                      max_dist: float = NEIGHBOR_LINE_DIST) -> None:
    """Update LineCollections based on current agent positions and colors.

    Args:
        same_col: LineCollection for same-color neighbor edges.
        diff_col: LineCollection for different-color neighbor edges.
        agents: list of Agent objects.
        k: number of neighbors to use for k-NN graph.
        max_dist: fallback maximum distance if k is None.
    """
    pts = np.array([a.pos for a in agents])
    edges = build_neighbor_edges(agents, k=k, max_dist=max_dist)
    same_segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    same_cols: List[str] = []
    diff_segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i, j in edges:
        seg = (pts[i].tolist(), pts[j].tolist())
        if agents[i].color == agents[j].color:
            same_segs.append(seg)
            same_cols.append(COLOR_HEX[agents[i].color])
        else:
            diff_segs.append(seg)
    same_col.set_segments(same_segs)
    if same_cols:
        same_col.set_color(same_cols)
    diff_col.set_segments(diff_segs)


# -------------------------
# Visualization & simulation runner
# -------------------------
def run_simulation(
    N: int = N,
    k: int = K,
    animate: bool = None,
    raster_zones_res: int = RASTER_ZONES_RES,
    zones_alpha_scale: float = ZONES_ALPHA_SCALE,
    raster_zones: bool = RASTER_ZONES,
    cluster_circles: bool = CLUSTER_CIRCLES,
    cluster_dist: float = CLUSTER_DIST,
    cluster_circle_alpha: float = CLUSTER_CIRCLE_ALPHA,
    area_size: float = AREA_SIZE,
    max_steps: int = MAX_STEPS,
    detail_mode: bool = DETAIL_MODE
) -> List[Agent]:
    """Run the simulation and optionally animate it.

    This function sets up matplotlib, initializes agents, and either animates
    the simulation using FuncAnimation or runs it in a headless loop.

    Args:
        N: number of agents.
        k: neighbor count for sensor/beep logic.
        animate: if True show interactive animation, otherwise run headless.
        raster_zones_res: raster resolution for influence visualization.
        zones_alpha_scale: alpha scaling for influence raster.
        raster_zones: whether to display influence raster.
        cluster_circles: whether to draw merged influence circles.
        cluster_dist: cluster merge distance for circles.
        cluster_circle_alpha: alpha for influence circles.
        area_size: side length of simulation square.
        max_steps: maximum steps to run (when animate=False or forced stop).
        detail_mode: when True, draw neighbor lines and grids.

    Returns:
        List[Agent]: final agent states after simulation ends or animation stops.
    """
    agents = init_agents(N=N, frac_orange=0.5, area_size=area_size)

    plt.style.use("cyberpunk")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)

    if detail_mode:
        ax.grid(linestyle='-', color='#888888', alpha=0.4)
    else:
        ax.grid(linestyle='')
        ax.axis('off')

    pts = np.array([a.pos for a in agents])
    base_colors = [COLOR_HEX[a.color] for a in agents]
    sizes = [120 if a.moving else 50 for a in agents]

    same_lines, diff_lines = make_neighbor_line_collections(ax)

    zones_im = None
    if raster_zones:
        rgba, n_zones, _ = compute_influence_zones(
            agents, area_size=area_size, res=RASTER_ZONES_RES, alpha_scale=zones_alpha_scale)
        zones_im = ax.imshow(rgba, extent=[
            0, area_size, 0, area_size], origin='lower', zorder=0, interpolation='bilinear')
        zone_count = n_zones
    else:
        zone_count = 0

    if detail_mode and cluster_circles:
        cluster_count, merged = compute_clusters(
            agents, cluster_dist=cluster_dist)
        draw_cluster_circles_ax(ax, merged, alpha=cluster_circle_alpha)

    glow0 = ax.scatter(pts[:, 0], pts[:, 1], s=np.array(
        sizes)*3.0, c=base_colors, alpha=0.06, linewidths=0, zorder=3)
    glow1 = ax.scatter(pts[:, 0], pts[:, 1], s=np.array(
        sizes) * 2.8, c=base_colors, alpha=0.12, linewidths=0, zorder=4)
    glow2 = ax.scatter(pts[:, 0], pts[:, 1], s=np.array(
        sizes) * 1.8, c=base_colors, alpha=0.16, linewidths=0, zorder=5)
    glow3 = ax.scatter(pts[:, 0], pts[:, 1], s=np.array(
        sizes) * 1.2, c=base_colors, alpha=0.28, linewidths=0, zorder=6)
    scat = ax.scatter(pts[:, 0], pts[:, 1], c=base_colors,
                      s=sizes, edgecolors='k', linewidths=0.6, zorder=7)

    title = ax.text(0.02, 1.02, "", transform=ax.transAxes)
    step = 0

    def update_layer(scatter_obj: plt.PathCollection, points: Any, colors: Any, sizes_arr: Any) -> None:
        """Helper to update scatter PathCollection data safely."""
        if len(points) == 0:
            scatter_obj.set_offsets(np.zeros((0, 2)))
            scatter_obj.set_facecolor([])
            scatter_obj.set_sizes(np.array([], dtype=float))
            return
        scatter_obj.set_offsets(points)
        scatter_obj.set_facecolor(colors)
        scatter_obj.set_sizes(sizes_arr)

    def compute_main_sizes(agents: List[Agent], k: int) -> np.ndarray:
        """Compute visual sizes: enlarge agents whose k neighbors are all same color."""
        pts_arr = np.array([a.pos for a in agents])
        N_agents = len(agents)
        default_sizes = np.array(
            [120 if a.moving else 50 for a in agents], dtype=float)
        if N_agents == 0:
            return default_sizes
        tree = cKDTree(pts_arr)
        kk = min(k + 1, N_agents)
        dists, idxs = tree.query(pts_arr, kk)
        colors_arr = np.array([a.color for a in agents], dtype=int)
        main_sizes = default_sizes.copy()
        for i in range(N_agents):
            nbrs = [j for j in idxs[i] if j != i][:k]
            if len(nbrs) == 0:
                continue
            if np.all(colors_arr[nbrs] == colors_arr[i]):
                main_sizes[i] = 240.0
        return main_sizes

    def init() -> Tuple[plt.PathCollection, Any]:
        title.set_text(f"step 0 k={k}")
        return scat, title

    def update(frame: int) -> Tuple[plt.PathCollection, Any]:
        nonlocal step, cluster_count, zone_count
        step += 1
        beeps = sensor_beep_states(agents, k)
        for i, a in enumerate(agents):
            a.moving = bool(beeps[i])
        for a in agents:
            a.step()

        pts = np.array([a.pos for a in agents])
        colors_plot = [COLOR_HEX[a.color] for a in agents]
        sizes_arr = np.array([120 if a.moving else 50 for a in agents])

        moving_mask = np.array([a.moving for a in agents])
        if moving_mask.any():
            pts_m = pts[moving_mask]
            cols_m = [colors_plot[i] for i, m in enumerate(moving_mask) if m]
            sizes_m = sizes_arr[moving_mask]
            update_layer(glow0, pts_m, cols_m, sizes_m * 4.8)
            update_layer(glow1, pts_m, cols_m, sizes_m * 2.8)
            update_layer(glow2, pts_m, cols_m, sizes_m * 1.8)
            update_layer(glow3, pts_m, cols_m, sizes_m * 1.2)
        else:
            update_layer(glow0, [], [], [])
            update_layer(glow1, [], [], [])
            update_layer(glow2, [], [], [])
            update_layer(glow3, [], [], [])

        main_sizes = compute_main_sizes(agents, k)
        update_layer(scat, pts, colors_plot, main_sizes)
        scat.set_edgecolors(['k'] * len(agents))

        if detail_mode:
            update_neighbor_lines_from_agents(
                same_lines, diff_lines, agents, k=k)

        if raster_zones and zones_im is not None:
            rgba, n_zones, _ = compute_influence_zones(
                agents, area_size=area_size, res=raster_zones_res, alpha_scale=zones_alpha_scale)
            zones_im.set_data(rgba)
            zone_count = n_zones

        if cluster_circles:
            cluster_count, merged = compute_clusters(
                agents, cluster_dist=cluster_dist)
            draw_cluster_circles_ax(ax, merged, alpha=cluster_circle_alpha)

        homophilic_count = int((main_sizes == 240.0).sum())
        parts = [f"step {step}", f"k={k}",
                 f"moving={sum(a.moving for a in agents)}", f"homophilic_agents={homophilic_count}"]
        if cluster_circles and (cluster_count > 0):
            parts.append(f"clusters={cluster_count}")
        if raster_zones and (zone_count > 0):
            parts.append(f"zones={zone_count}")
        title.set_text(" ".join(parts))

        if STOP_WHEN_SILENT and (sum(a.moving for a in agents) == 0 or step >= max_steps):
            try:
                ani.event_source.stop()
            except Exception:
                pass
        return scat, title

    if animate:
        ani = FuncAnimation(fig, update, init_func=init,
                            interval=120, blit=False, cache_frame_data=False)
        plt.show()
    else:
        step = 0
        n_zones = 'zone mode not enabled'
        n_clusters = 'cluster mode not enabled'
        while step < max_steps:
            beeps = sensor_beep_states(agents, k)
            for i, a in enumerate(agents):
                a.moving = bool(beeps[i])
            for a in agents:
                a.step()
            step += 1
            if STOP_WHEN_SILENT and (sum(a.moving for a in agents) == 0):
                break
        main_sizes = compute_main_sizes(agents, k)
        homophilic_count = int((main_sizes == 240.0).sum())

        zones_count = None
        if raster_zones:
            rgba, zones_count, _ = compute_influence_zones(
                agents, area_size=area_size, res=raster_zones_res, alpha_scale=zones_alpha_scale)

        clusters_count = None
        if cluster_circles:
            zone_count, merged = compute_clusters(
                agents, cluster_dist=cluster_dist)
            clusters_count = len(merged)  # count, not full details
        metrics = {"steps": step, "homophilic_agents": homophilic_count}
        if detail_mode:
            metrics["zones"] = zones_count
            metrics["clusters"] = clusters_count
        else:
            if raster_zones:
                metrics["zones"] = zones_count
            if cluster_circles:
                metrics["clusters"] = clusters_count
        parts = [f"steps={metrics['steps']}",
                 f"homophilic_agents={metrics['homophilic_agents']}"]
        if "clusters" in metrics:
            parts.append(f"clusters={metrics['clusters']}")
        if "zones" in metrics:
            parts.append(f"zones={metrics['zones']}")
        print("Finished " + " | ".join(parts))

    return agents

# -------------------------
# CLI
# -------------------------
def parse_and_run() -> None:
    """CLI parsing and run entrypoint.

    CLI options:
      - zones : enable/disable the influence raster
      - zones-res : raster resolution for zones
      - zones-alpha : alpha scale for zones
      - cluster-circles : enable merged cluster circles
      - cluster-dist : distance threshold for cluster circles
      - cluster-alpha : alpha for cluster circles
      - detail: enable detail mode (neighbor lines, grid)
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--headless", dest="animate", action="store_false",
                        help="headless mode")
    parser.set_defaults(animate=True)
    parser.add_argument("--detail", dest="detail_mode", action="store_true",
                        help="enable detail mode (draw neighbor lines, grid)")
    parser.add_argument("--no-detail", dest="detail_mode", action="store_false",
                        help="disable detail mode")
    parser.set_defaults(detail_mode=DETAIL_MODE)

    parser.add_argument("--N", type=int, default=22, help="number of agents")
    parser.add_argument("--k", type=int, default=8,
                        help="neighbors for sensor logic")
    parser.add_argument("--area", type=float,
                        default=AREA_SIZE, help="simulation area size")
    parser.add_argument("--steps", type=int,
                        default=MAX_STEPS, help="maximum steps to run")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-zones", dest="zones", action="store_const", const=False,
                       help="disable zones raster")
    group.add_argument("--raster-zones", dest="zones", action="store_const", const=True,
                       help="enable zones raster")
    parser.set_defaults(zones=RASTER_ZONES)

    parser.add_argument("--zones-res", type=int, default=RASTER_ZONES_RES,
                        help="zones raster resolution")
    parser.add_argument("--zones-alpha", type=float, default=ZONES_ALPHA_SCALE,
                        help="zones alpha scale")

    cc_group = parser.add_mutually_exclusive_group()
    cc_group.add_argument("--cluster-circles", dest="cluster_circles", action="store_true",
                          help="draw merged same-color cluster circles")
    cc_group.add_argument("--no-cluster-circles", dest="cluster_circles", action="store_false",
                          help="disable cluster circles")
    parser.set_defaults(cluster_circles=CLUSTER_CIRCLES)

    parser.add_argument("--cluster-dist", type=float, default=CLUSTER_DIST,
                        help="distance threshold for forming cluster circles")
    parser.add_argument("--cluster-alpha", type=float, default=CLUSTER_CIRCLE_ALPHA,
                        help="alpha for cluster circle fill")

    args = parser.parse_args()

    show_zones = args.zones

    run_simulation(
        N=args.N,
        k=args.k,
        animate=args.animate,
        raster_zones_res=args.zones_res,
        zones_alpha_scale=args.zones_alpha,
        raster_zones=show_zones,
        cluster_circles=args.cluster_circles,
        cluster_dist=args.cluster_dist,
        cluster_circle_alpha=args.cluster_alpha,
        area_size=args.area,
        max_steps=args.steps,
        detail_mode=args.detail_mode
    )

if __name__ == "__main__":
    parse_and_run()  # 🂱
