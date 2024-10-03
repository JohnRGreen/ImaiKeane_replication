def discretize_log_distribution(points, mean, sd):
  # Function that returns quadrature points of number "points" from a log-normal distribution with mean "mean" and standard deviation "sd"
  var = sd ** 2
  Z = np.full(points + 1, np.nan)
  EVbetweenZ = np.full(points, np.nan)

  # manually set upper bounds at +/- 3
  Z[0] = -3
  Z[points] = 3

  # Get the cutoff points
  prob = 1 / points
  for i in range(points - 1, 0, -1):
    Z[i] = scipy.stats.norm.ppf(prob * i)

  # get mean value in each interval (see Adda & Cooper page 58)
  for i in range(points):
    ev = (scipy.stats.norm.pdf(Z[i]) - scipy.stats.norm.pdf(Z[i + 1])) / (scipy.stats.norm.cdf(Z[i + 1]) - scipy.stats.norm.cdf(Z[i]))
    pt = sd * ev + mean
    EVbetweenZ[i] = pt

  quad2 = np.exp(EVbetweenZ)
  return quad2

print("Defined the DiscretizeLogDistribution function.")