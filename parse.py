import pandas as pd
from typing import Dict, Callable, List


def parse_sol(sol_path: str) -> Dict[str, float]:
    """
    Generates var -> val map given a lp solution.
    """
    d = dict()
    with open(sol_path, "r") as f:
        for line in f:
            line = line.strip()
            if (
                not line
                or line.startswith("objective value")
                or not line.startswith("x_")
            ):
                continue
            var, val, _ = line.split()
            d[var] = float(val)
    return d


def to_df(
    sol: Dict[str, float],
    index_of: Dict[str, int],
    country_of: Dict[str, str],
) -> pd.DataFrame:
    """
            Generates a dataframe representation of the obtained fixture.

    Inputs:
                - sol (Dict[str, float]): Obtained via `parse_sol`
                - index_of (Dict[str, int]): Map from country name to index.
                - country_of (Dict[int, str]): Map from str(index) to country names.

    """
    countries = index_of.keys()
    n = len(countries)
    data = dict()
    data["Team"] = countries
    for i in range(2 * (n - 1)):
        data[str(i)] = ["" for _ in range(n)]
    for var, val in sol.items():
        if not var.startswith("x_") or val == 0:
            continue
        _, i, j, k = var.split("_")
        c_i = country_of[i]
        c_i_idx = index_of[c_i]
        c_j = country_of[j]
        c_j_idx = index_of[c_j]
        data[k][c_i_idx] = c_j
        data[k][c_j_idx] = f"@{c_i}"
    return pd.DataFrame(data)
