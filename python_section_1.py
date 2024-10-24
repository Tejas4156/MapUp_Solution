from typing import Dict, List
import pandas as pd
import itertools
import re
import polyline
from geopy.distance import geodesic


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)):
            group.insert(0, lst[i + j])
        result.extend(group)
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    result = {}
    for word in lst:
        length = len(word)
        if length not in result:
            result[length] = []
        result[length].append(word)
    return dict(sorted(result.items()))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    def flatten(current_dict, parent_key=''):
        items = {}
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.update(flatten({f"{k}[{i}]": item}, parent_key))
            else:
                items[new_key] = v
        return items

    return flatten(nested_dict)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    return list(map(list, set(itertools.permutations(nums))))


def find_all_dates(text: str) -> List[str]:
    pattern = r'\b\d{2}-\d{2}-\d{4}|\b\d{2}/\d{2}/\d{4}|\b\d{4}\.\d{2}\.\d{2}'
    return re.findall(pattern, text)


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    df['distance'] = [0] + [
        geodesic(df.iloc[i], df.iloc[i - 1]).meters for i in range(1, len(df))
    ]
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            row.append(row_sum + col_sum)
        result.append(row)

    return result



from datetime import timedelta

def time_check(df: pd.DataFrame) -> pd.Series:
    df['startDateTime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['endDateTime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    def generate_full_week():
        days = pd.date_range('2023-01-01', periods=7).normalize()
        full_time_range = pd.date_range(start='00:00:00', end='23:59:59', freq='1S').time
        return [(day + timedelta(seconds=i)).normalize() for day in days for i in range(86400)]

    full_week = generate_full_week()

    def is_incomplete(group):
        coverage = []
        for _, row in group.iterrows():
            coverage.extend(pd.date_range(row['startDateTime'], row['endDateTime'], freq='1S'))
        coverage_set = set(coverage)
        return len(coverage_set) != len(full_week)

    completeness = df.groupby(['id', 'id_2']).apply(is_incomplete)

    return completeness

