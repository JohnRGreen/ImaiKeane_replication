def getSimMoments_step6(simdf):
    
    # bins = [21, 25, 30, 35, 40, 45, 50, 55, 60, 65]

    # # Use pd.cut to categorize the ages into the defined bins
    # simdf['age_group'] = pd.cut(simdf['age'], bins, right=True)

    # # Group by age_group and calculate the mean assets in each group
    # mean_assets_by_age_group = simdf.groupby('age_group', observed=True)['a'].mean()

    # mean_assets_moments = mean_assets_by_age_group.tolist()
    # first_four = mean_assets_moments[:3]

    # # now get retirement assets
    # simdf_age_66 = simdf[simdf['age'] == 65]

    # # Calculate the mean assets when age is 65
    # mean_assets_age_66 = simdf_age_66['a'].mean()

    # # If you want to combine them into a single list
    # moments = first_four + [mean_assets_age_66]

    bins = [20, 23, 26, 29, 32, 64, 65]

    # Use pd.cut to categorize the ages into the defined bins
    simdf['age_group'] = pd.cut(simdf['age'], bins, right=True)

    # Group by age_group and calculate the mean assets in each group
    mean_assets_by_age_group = simdf.groupby('age_group', observed=True)['a'].mean()

    mean_assets_moments = mean_assets_by_age_group.tolist()
    first_four = mean_assets_moments[:4]

    # Calculate the mean assets when age is 65
    mean_assets_age_66 = mean_assets_moments[5]

    # Calculate hours moments
    mean_hours_by_age_group = simdf.groupby('age_group', observed=True)['h'].mean()

    mean_hours_moments = mean_hours_by_age_group.tolist()
    hours_moments = mean_hours_moments[:4]

    # If you want to combine them into a single list
    moments = first_four + [mean_assets_age_66] + hours_moments

    return moments

# Compare this snippet from solveProblem.py: