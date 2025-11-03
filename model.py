from pyscipopt import Model, quicksum
from enum import Enum
from typing import List, Dict
from pyscipopt.scip import Solution, Variable
import unittest


class SymetricScheme(Enum):
    MIRRORED = 0
    FRENCH = 1
    ENGLISH = 2
    INVERTED = 3
    BACK_TO_BACK = 4
    MIN_MAX = 5


class FootballSchedulerModel:
    """
    Football Scheduler model class. Initializes the model and decision variables, and handles modeling logic.

    The model is described at:
    G. Durán, E. Mijangos, and M. Frisk,
    “Scheduling the South American qualifiers to the 2018 FIFA World Cup by integer programming,”
    European Journal of Operational Research, vol. 262, no. 3, pp. 1035-1048, 2017.

    Attributes:
                N: Number of teams
                K: Number of rounds
                I_s: List of top teams
                scheme: SymetricScheme
                x: Decision variable x[i,j,k]
                y: Decision variable y[i,k]
                w: Decision variable w[i,k]
                model: SCIP model
                c: Parameter c for MIN_MAX scheme
                d: Parameter d for MIN_MAX scheme
    """

    def __init__(
        self,
        N: int,
        scheme: SymetricScheme,
        I_s: List[int] = [],
        c: int = 0,
        d: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize the FootballScheduler class.

        Args:
                        N (int): Number of teams
                        scheme (SymetricScheme): Symmetric scheme to be used
                        I_s (List[int], optional): List of top teams. Should be a subset of teams {0, 1, ..., N-1}. Defaults to [].
                        c (int, optional): Parameter c for MIN_MAX scheme. Defaults to 0.
                        d (int, optional): Parameter d for MIN_MAX scheme. Defaults to 0.
                        verbose (bool,optional): Wether to show model logs. Defaults to False.
        """
        if N % 2 != 0:
            raise ValueError("N is not even")
        if len(I_s) > 0 and (max(I_s)) > N:
            raise ValueError("I_s must be a subset of teams I")
        if scheme != SymetricScheme.MIN_MAX and (c != 0 or d != 0):
            raise ValueError(
                "c and d should not be provided for schemes other than MIN_MAX"
            )
        if scheme == SymetricScheme.MIN_MAX and (
            not (1 <= c <= N) or not (c <= d <= 2 * (N - 1))
        ):
            raise ValueError("Invalid values for c and d")

        self.N = N
        self.K = 2 * (N - 1)
        self.I_s = I_s
        self.scheme = scheme

        self.x = {}
        self.y = {}
        self.w = {}

        self.__model = Model("Football Scheduler")
        self.__model.setIntParam("misc/usesymmetry", 0)

        if not verbose:
            self.__model.hideOutput()
            self.__model.setIntParam("display/verblevel", 0)
            self.__model.setBoolParam("display/lpinfo", False)
            self.__model.setBoolParam("display/relevantstats", False)
            self.__model.redirectOutput()

        self.c = c
        self.d = d

        self.__instance_vars()
        self.__instance_constraints()
        self.__set_objective()

    def __ensure_status(self, status: str):
        current_status = self.__model.getStatus()
        if current_status != status:
            raise RuntimeError(
                f"Model is not {status}, current status is {current_status}"
            )

    def __instance_vars(self):
        """
        Define the decision variables

        Variables:
                        x[i,j,k] = 1 if team i plays against team j in round k.
                        y[i,k] = 1 if team i has an H-A sequence in the doubleround that start at round k.
                        w[i,k] = 1 if team i has an away break in the double round starting in round k.
        """
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.K):
                    self.x[i, j, k] = self.__model.addVar(
                        vtype="B", name=f"x_{i}_{j}_{k}"
                    )

        for i in range(self.N):
            for k in range(self.K):
                self.y[i, k] = self.__model.addVar(vtype="B", name=f"y_{i}_{k}")
                self.w[i, k] = self.__model.addVar(vtype="B", name=f"w_{i}_{k}")

    def __instance_constraints(self):
        """
        Define the model constraints

        Constraints:
                        - Double round robin
                        - Compactness
                        - Top-teams
                        - Balance
                        - Aux Vars
                        - Symmetric scheme
        """
        if self.scheme != SymetricScheme.BACK_TO_BACK:
            self.__instance_double_round_robin_constraints()
        self.__instance_compactness_constraints()
        if len(self.I_s) > 0:
            self.__instance_top_teams_constraints()
        self.__instance_balance_constraints()
        self.__instance_aux_var_constraints()
        self.__instance_symmetric_scheme_constraints()

    def __instance_double_round_robin_constraints(self):
        # Double round robin constraints.
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                # (1) - every team faces every other team once in the first half
                self.__model.addCons(
                    quicksum(
                        self.x[i, j, k] + self.x[j, i, k] for k in range(self.N - 1)
                    )
                    == 1,
                    name=f"match_first_half_{i}_{j}",
                )
                # (2) - every team faces every other team once in the second half
                self.__model.addCons(
                    quicksum(
                        self.x[i, j, k] + self.x[j, i, k]
                        for k in range(self.N - 1, self.K)
                    )
                    == 1,
                    name=f"match_second_half_{i}_{j}",
                )
                # (3) - exactly one of the two games is played at home while the other one is played away
                self.__model.addCons(
                    quicksum(self.x[i, j, k] for k in range(self.K)) == 1,
                    name=f"not_two_home_half_{i}_{j}",
                )

    def __instance_compactness_constraints(self):
        # Compactness constraints
        for j in range(self.N):
            for k in range(self.K):
                # (4) - all teams must play one match in each round.
                self.__model.addCons(
                    quicksum(
                        self.x[i, j, k] + self.x[j, i, k]
                        for i in range(self.N)
                        if i != j
                    )
                    == 1,
                    name=f"one_match_per_round_{j}_{k}",
                )

    def __instance_top_teams_constraints(self):
        # Top-teams constraints
        for i in [x for x in range(self.N) if x not in self.I_s]:
            for k in range(self.K - 1):
                for j in self.I_s:
                    # (5) - No non-top team be required to play against any of the top teams in consecutive matches.
                    self.__model.addCons(
                        self.x[i, j, k]
                        + self.x[j, i, k]
                        + self.x[i, j, k + 1]
                        + self.x[j, i, k + 1]
                        <= 1,
                        name=f"top_team_cons_{i}_{j}_{k}",
                    )

    def __instance_balance_constraints(self):
        # Balance constraints
        for i in range(self.N):
            # (6) - Each team has bewteen N/2-1 and N/2 H-A sequences in double rounds.
            self.__model.addCons(
                quicksum(self.y[i, k] for k in range(0, self.K, 2))
                >= (self.N // 2) - 1,
                name=f"bound_below_HA_seq_{i}",
            )
            self.__model.addCons(
                quicksum(self.y[i, k] for k in range(0, self.K, 2)) <= (self.N // 2),
                name=f"bound_above_HA_seq_{i}",
            )

            for k in range(0, self.K, 2):
                # (7) - Teams should not play consecutive home or away matches in double rounds.
                self.__model.addCons(
                    quicksum(
                        self.x[i, j, k] + self.x[j, i, k + 1]
                        for j in range(self.N)
                        if j != i
                    )
                    <= 1 + self.y[i, k],
                    name=f"HA_{i}_{k}",
                )
                # (8) - No more H-A sequences than played games in round k
                self.__model.addCons(
                    quicksum(self.x[i, j, k] for j in range(self.N) if j != i)
                    >= self.y[i, k],
                    name=f"c8_{i}_{k}",
                )
                # (9) - No more H-A sequences than played games in round k + 1
                self.__model.addCons(
                    quicksum(self.x[j, i, k + 1] for j in range(self.N) if j != i)
                    >= self.y[i, k],
                    name=f"c9_{i}_{k}",
                )

    def __instance_aux_var_constraints(self):
        # Aux constraints
        for i in range(self.N):
            for k in range(0, self.K, 2):
                # (10) - Teams should not have two consecutive away breaks
                self.__model.addCons(
                    quicksum(
                        self.x[j, i, k] + self.x[j, i, k + 1]
                        for j in range(self.N)
                        if j != i
                    )
                    <= 1 + self.w[i, k],
                    name=f"AB_{i}_{k}",
                )
                # (11) - No more away breaks sequences than played games in round k
                self.__model.addCons(
                    quicksum(self.x[j, i, k] for j in range(self.N) if j != i)
                    >= self.w[i, k],
                    name=f"c11_{i}_{k}",
                )
                # (12) - No more away breaks sequences than played games in round k + 1
                self.__model.addCons(
                    quicksum(self.x[j, i, k + 1] for j in range(self.N) if j != i)
                    >= self.w[i, k],
                    name=f"c12_{i}_{k}",
                )

    def __instance_symmetric_scheme_constraints(self):
        if self.scheme == SymetricScheme.MIRRORED:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    for k in range(self.N - 1):
                        # (14) - Mirror scheme constraint
                        self.__model.addCons(
                            self.x[i, j, k] == self.x[j, i, k + self.N - 1],
                            f"mirrored_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.FRENCH:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    # (15) - French scheme constraint 1
                    self.__model.addCons(
                        self.x[i, j, 0] == self.x[j, i, 2 * self.N - 3],
                        f"french_1_{i}_{j}",
                    )
                    for k in range(1, self.N - 1):
                        # (15) - French scheme constraint 2
                        self.__model.addCons(
                            self.x[i, j, k] == self.x[j, i, k + self.N - 2],
                            f"french_2_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.ENGLISH:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    # (16) - English scheme constraint 1
                    self.__model.addCons(
                        self.x[i, j, self.N - 2] == self.x[j, i, self.N - 1],
                        f"english_1_{i}_{j}",
                    )
                    for k in range(1, self.N - 2):
                        # (16) - English scheme constraint 2
                        self.__model.addCons(
                            self.x[i, j, k] == self.x[j, i, k + self.N],
                            f"english_2_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.INVERTED:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    for k in range(self.N - 2):
                        # (17) - Inverted scheme constraint
                        self.__model.addCons(
                            self.x[i, j, k] == self.x[j, i, 2 * (self.N - 1) - 1 - k],
                            f"inverted_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.BACK_TO_BACK:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    for k in range(0, self.K - 1, 2):
                        # (18) - Back to back scheme constraint
                        self.__model.addCons(
                            self.x[i, j, k] == self.x[j, i, k + 1],
                            f"back_to_back_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.MIN_MAX:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    for k in range(0, self.K - self.c):
                        # (19) - Min max scheme constraint 1
                        self.__model.addCons(
                            quicksum(
                                self.x[i, j, q] + self.x[j, i, q]
                                for q in range(k, k + self.c + 1)
                            )
                            <= 1,
                            f"min_max_1_{i}_{j}_{k}",
                        )
                    for k in range(0, self.K):
                        # (19) - Min max scheme constraint 2
                        self.__model.addCons(
                            quicksum(
                                self.x[i, j, q]
                                for q in range(
                                    max(k - self.d, 0),
                                    min(k + self.d + 1, 2 * (self.N - 1)),
                                )
                                if q != k
                            )
                            >= self.x[j, i, k],
                            f"min_max_2_{i}_{j}_{k}",
                        )

        else:
            pass

    def __set_objective(self):
        # (13) - Minimize the total number of away breaks within double rounds across all teams.
        self.__model.setObjective(
            quicksum(self.w[i, k] for k in range(0, self.K, 2) for i in range(self.N)),
            sense="minimize",
        )

    def get_vars(self) -> List[Variable]:
        return self.__model.getVars()

    def presolve(self):
        self.__model.presolve()
        if self.__model.getStatus == "infeasible":
            raise RuntimeError(f"Model is infeasible")

    def optimize(self):
        self.__model.optimize()
        self.__ensure_status("optimal")

    def get_obj_value(self) -> float:
        self.__ensure_status("optimal")
        return self.__model.getObjVal()

    def get_best_sol(self) -> Solution:
        self.__ensure_status("optimal")
        return self.__model.getBestSol()

    def get_value(self, var: str) -> float:
        self.__ensure_status("optimal")
        return self.__model.getVal(var)

    def get_solving_time(self) -> float:
        self.__ensure_status("optimal")
        return self.__model.getSolvingTime()

    def write_problem(self, path: str):
        self.__model.writeProblem(path)

    def write_sol(self, path: str):
        self.__ensure_status("optimal")
        sol = self.get_best_sol()
        self.__model.writeSol(sol, filename=path)


class TestFootballSchedulerModel(unittest.TestCase):

    def test_instance_mirrored(self):
        _ = FootballSchedulerModel(10, SymetricScheme.MIRRORED)

    def test_instance_top_teams(self):
        _ = FootballSchedulerModel(10, SymetricScheme.MIRRORED, [1, 2])

    def test_instance_french(self):
        _ = FootballSchedulerModel(10, SymetricScheme.FRENCH)

    def test_french_is_feasible(self):
        model = FootballSchedulerModel(10, SymetricScheme.FRENCH, [1, 2])
        model.optimize()
        self.assertEqual(model.get_obj_value(), 0)

    def test_instance_english(self):
        _ = FootballSchedulerModel(10, SymetricScheme.ENGLISH)

    def test_english_is_feasible(self):
        model = FootballSchedulerModel(10, SymetricScheme.ENGLISH, [1, 2])
        model.optimize()
        self.assertEqual(model.get_obj_value(), 0)

    def test_instance_inverted(self):
        _ = FootballSchedulerModel(10, SymetricScheme.INVERTED)

    def test_inverted_is_feasible(self):
        model = FootballSchedulerModel(10, SymetricScheme.INVERTED, [1, 2])
        model.optimize()
        self.assertEqual(model.get_obj_value(), 0)

    def test_instance_back_to_back(self):
        _ = FootballSchedulerModel(10, SymetricScheme.BACK_TO_BACK)

    def test_back_to_back_is_feasible(self):
        model = FootballSchedulerModel(10, SymetricScheme.BACK_TO_BACK, [1, 2])
        model.optimize()
        self.assertEqual(model.get_obj_value(), 0)

    def test_instance_min_max(self):
        _ = FootballSchedulerModel(10, SymetricScheme.MIN_MAX, c=6, d=12)

    def test_min_max_presolve_is_feasible(self):
        model = FootballSchedulerModel(10, SymetricScheme.MIN_MAX, [1, 2], c=6, d=12)
        model.presolve()


if __name__ == "__main__":
    unittest.main()
