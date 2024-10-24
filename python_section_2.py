import pandas as pd
import numpy as np


# Question 9: Calculate Distance Matrix
def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize matrix with zeros
    ids = df['id'].unique()
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
    np.fill_diagonal(distance_matrix.values, 0)

    # Fill known distances from the dataframe
    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Symmetric property

    # Calculate cumulative distances
    for k in ids:
        for i in ids:
            for j in ids:
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j],
                                                distance_matrix.loc[i, k] + distance_matrix.loc[k, j])

    return distance_matrix


# Question 10: Unroll Distance Matrix
def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    unrolled_data = []
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                unrolled_data.append([id_start, id_end, df.loc[id_start, id_end]])

    return pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])


# Question 11: Find IDs within Percentage Threshold
def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> list:
    avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    threshold_min = avg_distance * 0.9
    threshold_max = avg_distance * 1.1

    ids_within_threshold = df.groupby('id_start')['distance'].mean().loc[
        lambda x: (x >= threshold_min) & (x <= threshold_max)].index.tolist()
    return sorted(ids_within_threshold)


# Question 12: Calculate Toll Rate
def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df


# Question 13: Calculate Time-Based Toll Rates
def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    df['start_day'] = 'Monday'
    df['start_time'] = pd.to_datetime('00:00:00').time()
    df['end_day'] = 'Sunday'
    df['end_time'] = pd.to_datetime('23:59:59').time()

    # Time-based discount factors
    for i, row in df.iterrows():
        start_day = row['start_day']
        start_time = row['start_time']

        if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if start_time <= pd.to_datetime('10:00:00').time():
                discount_factor = 0.8
            elif start_time <= pd.to_datetime('18:00:00').time():
                discount_factor = 1.2
            else:
                discount_factor = 0.8
        else:  # Weekends
            discount_factor = 0.7

        # Apply discount factor to all vehicle columns
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            df.at[i, vehicle] *= discount_factor

    return df
