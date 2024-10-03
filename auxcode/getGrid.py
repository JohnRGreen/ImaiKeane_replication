# Function that sets up a grid for interpolation. "min" and "max" are both vectors of the same length, which will usually be the number of time periods in a problem. "points" then gives the number of points between the min and max (in each period). "Method" refers to the type of spacing between points: linear spacing, log spacing, or something in between denoted by the "power" parameter.


def GetGrid(min, max, points, method, power):

  # MMake sure min and max are arrays of the same length
  if len(min) != len(max):
    print("Error: min and max must be arrays of the same length.")
    exit()

  # get span in each period
  span = max - min

  # Get the grid using log steps
  if method == "log":
    loggrid = np.full((points, len(span)), np.nan)
    for i in range(len(min)):
      vec = np.linspace(np.log(1), np.log(1 + span[i]), points)
      loggrid[:, i] = vec

    # then exponentiate
    grid = np.exp(loggrid) - 1

    # add back in the min
    min_mat = np.full((points, len(span)), min)
    grid = grid + min_mat

    # grid[1,] == asset_lb
    # round(grid[points,], 2) == round(asset_ub, 2)
    # all good
  elif method == "linear":  # if we want to do linear interpolation
    grid = np.full((points, len(span)), np.nan)
    for i in range(len(min)):
      vec = np.linspace(min[i], max[i], points)
      grid[:, i] = vec
  elif method == "power":
      grid = np.full((points, len(span)), np.nan)
      for i in range(len(min)):
        vec = np.linspace(0, 1, points)  # Create a linear space between 0 and 1
        transformed_vec = min[i] + (max[i] - min[i]) * vec **power  # Apply power function transformation
        grid[:, i] = transformed_vec
  else:
    print("error: method invalid. Choose 'log' or 'linear'")

  return grid
 

print("Defined the GetGrid function.")