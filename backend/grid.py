import math
import numpy as np

def grid(metadata, layout, params):
  """
  layout: numpy arrays x, y
  metadata: user-defined numpy arrays with metadata
  n_layer: number of cells in the layer (squared)
  n_tile: number of cells in the tile (squared)
  """
  x = layout["x"]
  y = layout["y"]
  x_min = np.min(x)
  x_max = np.max(x)
  y_min = np.min(y)
  y_max = np.max(y)

  # this creates the grid
  bins = np.linspace(x_min, x_max, params["n_layer"] - 1)
  xd = np.digitize(x, bins)
  bins = np.linspace(y_min, y_max, params["n_layer"] - 1)
  yd = np.digitize(y, bins)

  # the number of tiles is the number of cells divided by the number of cells in each tile
  # use ceil to ensure coverage when n_layer is not divisible by n_tile
  num_tiles = math.ceil(params["n_layer"]/params["n_tile"])
  # we will save the tiles in an array indexed by the tile coordinates
  tiles = {}
  for ti in range(num_tiles):
    for tj in range(num_tiles):
      tiles[(ti,tj)] = {
        "x": [],
        "y": [],
        "ci": [], # cell-space x coordinate
        "cj": [], # cell-space y coordinate
        "gi": [], # global index
      }

  for i,xi in enumerate(x):
    # layout-space coordinates
    yi = y[i]
    # grid-space cell coordinates
    ci = xd[i]
    cj = yd[i]
    # tile coordinate (clamped to available tiles)
    ti = min(math.floor(ci / params["n_tile"]), num_tiles - 1)
    tj = min(math.floor(cj / params["n_tile"]), num_tiles - 1)

    # TODO: don't append a point if it doesn't match a filter function provided in params
    filter = params.get("filter", lambda i,metadata: True)
    if(filter(i, metadata=metadata)):
      tiles[(ti,tj)]["x"].append(xi)
      tiles[(ti,tj)]["y"].append(yi)
      tiles[(ti,tj)]["ci"].append(ci)
      tiles[(ti,tj)]["cj"].append(cj)
      tiles[(ti,tj)]["gi"].append(i)
    
  return tiles

def write_grid_local(tiles, params):
  """
  Legacy utility to persist tiles locally (unused).
  Kept as a stub to preserve API surface; no external deps.
  """
  for ti, tj, tile in enumerate_tiles(tiles):
    filename = "{directory}/{name}/tile_{n_layer}_{n_tile}_{ti}_{tj}".format(ti=ti, tj=tj, **params)
    print("skipping save for", filename)

def enumerate_tiles(tiles):
  """
  Convenience
  """
  enumerated = []
  for key in tiles.keys():
    enumerated.append((key[0], key[1], tiles[key]))
  return enumerated


# TODO: use named parameters?
def tile_cells(tile):
  # we need to collect all the data points for a given set of cell indices
  # and then pass that to the render function
  cells = {}
  for i,gi in enumerate(tile["gi"]):
    ci = tile["ci"][i]
    cj = tile["cj"][i]
    c = cells.get((ci,cj), {"gi": [], "i": ci, "j": cj})
    c["gi"].append(gi)
    cells[(ci,cj)] = c

  return cells
