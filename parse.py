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
    n: int,
    sol: Dict[str, float],
) -> pd.DataFrame:
    """
            Generates a dataframe representation of the obtained fixture.

    Inputs:
                - n (int): Number of teams.
                - sol (Dict[str, float]): Obtained via `parse_sol`

    """
    data = dict()
    data["Team"] = [x for x in range(n)]
    for i in range(2 * (n - 1)):
        data[str(i)] = ["" for _ in range(n)]
    for var, val in sol.items():
        if not var.startswith("x_") or val == 0:
            continue
        _, i, j, k = var.split("_")
        data[k][int(i)] = j
        data[k][int(j)] = f"@{i}"
    return pd.DataFrame(data)


def to_df_mapped(
    sol: Dict[str, float],
    country_of: Dict[str, str],
) -> pd.DataFrame:
    """
            Generates a dataframe representation of the obtained fixture including country names.

    Inputs:
                - sol (Dict[str, float]): Obtained via `parse_sol`
                - country_of (Dict[str, str]). Maps indexes to countries.

    """
    df = to_df(len(country_of), sol)
    # Replace numeric team indices with country names
    df["Team"] = [country_of[str(i)] for i in df["Team"]]
    for col in df.columns[1:]:
        df[col] = df[col].apply(
            lambda v: (
                f"@{country_of[v[1:]]}"
                if isinstance(v, str) and v.startswith("@")
                else (country_of[v] if v != "" else "")
            )
        )
    return df
