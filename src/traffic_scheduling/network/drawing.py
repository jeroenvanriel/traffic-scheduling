# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: traffic
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Preamble

# %% [markdown]
# > ⚠️ **Warning:** This notebook file is paired to a .py file with the same name, such that we can cleanly import the functionality from other notebooks. This is done using the facilities of the jupytext package. The cells in this notebook that are only meant as "demonstration" are marked with the cell tag "active-ipynb", which causes the jupytext synchronization command to ignore these when syncing to the .py file. This is our current way of doing "literate programming" with jupyter notebooks.

# %%
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union


# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Drawing graphs and networks

# %% [markdown]
# Draw the connectivity graph with labeled nodes.

# %%
def draw_graph(G):
    """Draw connectivity graph with labeled nodes."""
    nx.draw_networkx(G, nx.get_node_attributes(G, 'pos'))
    plt.gca().set_aspect('equal')
    plt.gca().axis('off') # remove box
    fig = plt.gcf()
    return fig


# %% [markdown]
# Introduce a helper function to draw polygons with holes.

# %%
def polygon_to_pathpatch(poly, color='lightgrey'):
    """Convert Shapely Polygon/MultiPolygon to PathPatch, preserving holes."""

    def single_polygon_to_path(p):
        verts = []
        codes = []

        # Exterior
        x, y = p.exterior.coords.xy
        exterior_verts = list(zip(x, y))
        verts.extend(exterior_verts)
        codes.extend([Path.MOVETO] + [Path.LINETO]*(len(exterior_verts)-2) + [Path.CLOSEPOLY])

        # Interiors (holes)
        for interior in p.interiors:
            xi, yi = interior.coords.xy
            interior_verts = list(zip(xi, yi))
            verts.extend(interior_verts)
            codes.extend([Path.MOVETO] + [Path.LINETO]*(len(interior_verts)-2) + [Path.CLOSEPOLY])

        return PathPatch(Path(verts, codes), facecolor=color, edgecolor='none')

    if poly.geom_type == 'Polygon':
        return single_polygon_to_path(poly)
    elif poly.geom_type == 'MultiPolygon':
        return [single_polygon_to_path(p) for p in poly.geoms]


# %% [markdown]
# Either draw the road network using only road outlines, or as a lightgray filled polygon.

# %%
def draw_road_outline(G, road_width=1):
    """Draw stylized road network."""
    fig, ax = plt.subplots()
    lineargs = { 'color': 'k', 'linewidth': 0.8 }

    for u, v, a in G.edges(data=True):
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']

        # extend vehicle_w to both sides
        length = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        # compute the normal by rotating counterclockwise
        nx = (y1-y2) / length * road_width / 2
        ny = (x2-x1) / length * road_width / 2

        # sidelines
        plt.plot([x1 + nx, x2 + nx], [y1 + ny, y2 + ny], **lineargs)
        plt.plot([x1 - nx, x2 - nx], [y1 - ny, y2 - ny], **lineargs)

    ax.axis('off')
    ax.set_aspect('equal')
    fig.tight_layout(pad=0.05)
    return fig

def draw_road(G, road_width=1, ax=None):
    road_polygons = []
    for u, v, a in G.edges(data=True):
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        dx, dy = x2-x1, y2-y1
        length = np.hypot(dx, dy)
        nx, ny = -dy/length*road_width/2, dx/length*road_width/2

        # vertices of lane rectangle
        verts = [(x1+nx, y1+ny), (x2+nx, y2+ny), (x2-nx, y2-ny), (x1-nx, y1-ny)]
        road_polygons.append(Polygon(verts))

    network_outline = unary_union(road_polygons)
    patches = polygon_to_pathpatch(network_outline)

    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False

    if isinstance(patches, list):
        for p in patches:
            ax.add_patch(p)
    else: ax.add_patch(patches)

    ax.autoscale_view()
    ax.set_aspect('equal')
    if standalone:
        ax.axis('off')
        fig.tight_layout()
        return fig


# %% [markdown]
# ## Drawing network schedules
#

# %%
def plot_schedule(schedule, height=None):
    indices = schedule.instance.vehicle_indices
    nodes = list(schedule.instance.G.nodes)
    nr_nodes = len(nodes)

    if height is None:
        height = 0.4 * schedule.instance.rho
    end = schedule.makespan
    fig, ax = plt.subplots(figsize=(end, nr_nodes))
    cmap = colormaps["Paired"] # lane colors

    # crossing times from schedule
    for r, k in indices:
        for v in schedule.instance.routes[r]:
            row = nodes.index(v) + 1
            ax.add_patch(Rectangle((schedule.y[r,k,v], nr_nodes-row - height / 2), width=schedule.instance.rho, height=height,
                            linewidth=1, facecolor=cmap(r), edgecolor='k'))

    ticks = np.arange(nr_nodes)
    # reverse for top to bottom numbering
    labels = np.flip(nodes, axis=0)
    plt.yticks(ticks=ticks, labels=labels)

    plt.autoscale()
    plt.show()


# %% [markdown]
# ## Drawing vehicles

# %%
from traffic_scheduling.network.util import dist

def current_edge(G, route, pos):
    """Find the current edge (u,v) along the route of a vehicle, given its route
    position. Position x is assumed to be on edge (u,v) if x_u < x <= x_v. Also
    returns the route position relative to node u, so x - x_u. We assume that x
    is in the first edge (u,v) of the route whenever x_u <= x <= x_v."""
    if pos == 0: # first edge of the route is [x_u, x_v]
        return route[0], route[1], 0
    # all other routes are (x_u, x_v]
    cum_pos = 0
    v_prev = route[0]
    for v in route[1:]:
        d = dist(G, v_prev, v)
        if cum_pos < pos and pos <= cum_pos + d:
            return v_prev, v, pos - cum_pos
        cum_pos += d
        v_prev = v
    raise Exception("Position is beyond route.")

def get_pos(G, route, pos, vehicle_l, vehicle_w):
    """Get (x, y, angle) for drawing vehicle rectangle."""
    u, v, pos = current_edge(G, route, pos)
    x1, y1 = G.nodes[u]['pos']
    x2, y2 = G.nodes[v]['pos']

    # calculate angle
    v1 = np.array([x2-x1, y2-y1])
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = np.array([1, 0])
    angle = np.sign(y2-y1) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / math.pi * 180

    # go pos units in direction of edge
    pos -= vehicle_l # we want the front of the vehicle as reference
    # pos -= vehicle_w / 2 # route position to real position
    length = dist(G, u, v)
    x = x1 + (x2 - x1) / length * pos
    y = y1 + (y2 - y1) / length * pos

    # correct for vehicle width
    nx = (y1-y2) / length * vehicle_w / 2
    ny = (x2-x1) / length * vehicle_w / 2

    return x - nx, y - ny, angle

def draw_vehicles(G, routes, positions, vehicle_l, vehicle_w, ax, colors=None):
    rects = {}
    for vehicle, position in positions.items():
        route = routes[vehicle[0]]
        x, y, angle = get_pos(G, route, position, vehicle_l, vehicle_w)
        color = 'red' if colors is None else colors[vehicle]
        vehicleargs = { 'facecolor': color, 'edgecolor': color, 'linewidth': 0.0 }
        rect = Rectangle((x, y), angle=angle, width=vehicle_l, height=vehicle_w, **vehicleargs)
        rects[vehicle] = rect
        ax.add_patch(rect)

# %%
