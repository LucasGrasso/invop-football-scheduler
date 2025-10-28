from pyscipopt import Model, quicksum
from enum import Enum


class SymetricScheme(Enum):
    MIRRORED = 0
    FRENCH = 1
    ENGLISH = 2
    INVERTED = 3
    BACK_TO_BACK = 4
    MIN_MAX = 5


class FootballScheduler:
    """
    Football Scheduler class. Initializes the model and decision variables, and handles modeling logic.

    The model is described at:
    Durán, Guillermo, et al.
    “Scheduling the South American Qualifiers to the 2018 FIFA World Cup by Integer Programming.”
    European Journal of Operational Research, vol. 262, no. 3, 1 Nov. 2017, pp. 1109–1115,
    reader.elsevier.com/reader/sd/pii/S0377221717303909?token=ABD2EAA5C380716EBE2A40E7E9FD273BCD1E3E77BE2AA19DA677BACEBB6711BA08C8C68F8EF491779EF37FEA0268545F,
    https://doi.org/10.1016/j.ejor.2017.04.043.

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
        K: int,
        I_s: list[int],
        scheme: SymetricScheme,
        c: int = 0,
        d: int = 0,
    ):
        """
        Initialize the FootballScheduler class.

        Args:
                        N (int): Number of teams
                        K (int): Number of rounds
                        I_s (list[int]): List of top teams. Should be a subset of teams {0, 1, ..., N-1}
                        scheme (SymetricScheme): Symmetric scheme to be used
                        c (int, optional): Parameter c for MIN_MAX scheme. Defaults to None.
                        d (int, optional): Parameter d for MIN_MAX scheme. Defaults to None.
        """
        if N % 2 != 0:
            raise ValueError("N must be even")
        if K != 2 * N:
            raise ValueError("K must be equal to 2N")
        if (max(I_s)) > N:
            raise ValueError("I_s must be a subset of teams I")
        if scheme != SymetricScheme.MIN_MAX and (c is not 0 or d is not 0):
            raise ValueError(
                "c and d should not be provided for schemes other than MIN_MAX"
            )
        if scheme == SymetricScheme.MIN_MAX and (
            not (1 <= c <= N) or not (c <= d <= N)
        ):
            raise ValueError("Invalid values for c and d")

        self.N = N
        self.K = K
        self.I_s = I_s
        self.scheme = scheme

        self.x = {}
        self.y = {}
        self.w = {}
        self.model = Model("Football Scheduler")

        self.c = c
        self.d = d

        self.__instance_vars()
        self.__instance_constraints()
        self.__set_objective()

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
                    self.x[i, j, k] = self.model.addVar(
                        vtype="B", name=f"x_{i}_{j}_{k}"
                    )

        for i in range(self.N):
            for k in range(self.K):
                self.y[i, k] = self.model.addVar(vtype="B", name=f"y_{i}_{k}")
                self.w[i, k] = self.model.addVar(vtype="B", name=f"w_{i}_{k}")

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
        self.__instance_double_round_robin_constraints()
        self.__instance_compactness_constraints()
        self.__instance_top_teams_constraints()
        self.__instance_balance_constraints()
        self.__instance_aux_var_constraints()
        self.__instance_symmetric_scheme_constraints()

    def __instance_double_round_robin_constraints(self):
        # Double round robin constraints.
        for i in range(self.N):
            for j in range(self.K):
                if i == j:
                    continue
                # (1) - every team faces every other team once in the first half
                self.model.addCons(
                    quicksum(self.x[i, j, k] + self.x[j, i, k] for k in range(self.N))
                    == 1,
                    name=f"match_first_half_{i}_{j}",
                )
                # (2) - every team faces every other team once in the second half
                self.model.addCons(
                    quicksum(
                        self.x[i, j, k] + self.x[j, i, k]
                        for k in range(self.N, self.K, 1)
                    )
                    == 1,
                    name=f"match_second_half_{i}_{j}",
                )
                # (3) - exactly one of the two games is played at home while the other one is played away
                self.model.addCons(
                    quicksum(self.x[i, j, k] for k in range(self.K)) == 1,
                    name=f"match_second_half_{i}_{j}",
                )

    def __instance_compactness_constraints(self):
        # Compactness constraints
        for j in range(self.N):
            for k in range(self.K):
                # (4) - all teams must play one match in each round.
                self.model.addCons(
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
            for k in range(self.K):
                for j in self.I_s:
                    # (5) - No non-top team be required to play against any of the top teams in consecutive matches.
                    self.model.addCons(
                        quicksum(
                            self.x[i, j, k]
                            + self.x[j, i, k]
                            + self.x[i, j, k + 1]
                            + self.x[j, i, k + 1]
                        )
                        <= 1,
                        name=f"top_team_cons_{i}_{j}_{k}",
                    )

    def __instance_balance_constraints(self):
        # Balance constraints
        for i in range(self.N):
            # (6) - Each team has bewteen N/2-1 and N/2 H-A sequences in double rounds.
            self.model.addCons(
                quicksum(self.y[i, k] for k in range(1, self.K, 2))
                >= (self.N // 2) - 1,
                name=f"bound_below_HA_seq_{i}",
            )
            self.model.addCons(
                quicksum(self.y[i, k] for k in range(1, self.K, 2)) <= (self.N // 2),
                name=f"bound_above_HA_seq_{i}",
            )

            for k in range(1, self.K, 2):
                # (7) - Teams should not play consecutive home or away matches in double rounds.
                self.model.addCons(
                    quicksum(
                        self.x[i, j, k] + self.x[i, j, k + 1]
                        for j in range(self.N)
                        if j != i
                    )
                    <= 1 + self.y[i, k],
                    name=f"HA_{i}_{k}",
                )
                # (8) -
                self.model.addCons(
                    quicksum(self.x[i, j, k] for j in range(self.N) if j != i)
                    >= self.y[i, k],
                    name=f"c8_{i}_{k}",
                )
                # (9) -
                self.model.addCons(
                    quicksum(self.x[i, j, k + 1] for j in range(self.N) if j != i)
                    >= self.y[i, k],
                    name=f"c9_{i}_{k}",
                )

    def __instance_aux_var_constraints(self):
        # Aux constraints
        for i in range(self.N):
            for k in range(1, self.K, 2):
                # (10) - Teams should not have two consecutive away breaks
                self.model.addCons(
                    quicksum(
                        self.x[i, j, k] + self.x[i, j, k + 1]
                        for j in range(self.N)
                        if j != i
                    )
                    <= 1 + self.w[i, k],
                    name=f"AB_{i}_{k}",
                )
                # (11) -
                self.model.addCons(
                    quicksum(self.x[i, j, k] for j in range(self.N) if j != i)
                    >= self.w[i, k],
                    name=f"c11_{i}_{k}",
                )
                # (12) -
                self.model.addCons(
                    quicksum(self.x[i, j, k + 1] for j in range(self.N) if j != i)
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
                        self.model.addCons(
                            self.x[i, j, k] == self.x[j, i, k + self.N - 1],
                            f"mirrored_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.FRENCH:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    # (15) - French scheme constraint 1
                    self.model.addCons(
                        self.x[i, j, 0] == self.x[j, i, 2 * self.N - 3],
                        f"french_1_{i}_{j}_{k}",
                    )
                    for k in range(1, self.N - 1):
                        # (15) - French scheme constraint 2
                        self.model.addCons(
                            self.x[i, j, k] == self.x[j, i, k + self.N - 2],
                            f"french_2_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.ENGLISH:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    # (16) - English scheme constraint 1
                    self.model.addCons(
                        self.x[i, j, self.N - 2] == self.x[j, i, self.N - 1],
                        f"english_1_{i}_{j}_{k}",
                    )
                    for k in range(1, self.N - 2):
                        # (16) - English scheme constraint 2
                        self.model.addCons(
                            self.x[i, j, k] == self.x[j, i, k + self.N],
                            f"english_2_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.INVERTED:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    for k in range(self.N - 1):
                        # (17) - Inverted scheme constraint
                        self.model.addCons(
                            self.x[i, j, k] == self.x[j, i, 2 * self.N - 1 - k],
                            f"inverted_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.BACK_TO_BACK:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    for k in range(1, self.N - 1, 2):
                        # (18) - Back to back scheme constraint
                        self.model.addCons(
                            self.x[i, j, k] == self.x[j, i, k + 1],
                            f"back_to_back_{i}_{j}_{k}",
                        )

        elif self.scheme == SymetricScheme.MIN_MAX:
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    for k in range(0, self.K - self.c - 1):
                        # (19) - Min max scheme constraint 1
                        self.model.addCons(
                            quicksum(
                                self.x[i, j, q] + self.x[j, i, q]
                                for q in range(k, k + self.c + 1)
                            )
                            <= 1,
                            f"min_max_1_{i}_{j}_{k}",
                        )
                    for k in range(0, self.K):
                        # (19) - Min max scheme constraint 2
                        self.model.addCons(
                            quicksum(
                                self.x[i, j, q]
                                for q in range(
                                    max(k - self.d, 0),
                                    min(k + self.d, 2 * (self.N - 1)),
                                )
                                if q != k
                            )
                            >= self.x[j, i, k],
                            f"min_max_2_{i}_{j}_{k}",
                        )

        else:
            pass

    def __set_objective(self):
        """
        Sets the objective function.
        """
        # (13) - Minimize the total number ofaway breaks within double rounds across all teams.
        self.model.setObjective(
            quicksum(self.w[i, k] for k in range(1, self.K, 2) for i in range(self.N)),
            sense="minimize",
        )
