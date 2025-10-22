from pyscipopt import Model, quicksum
from enum import Enum

class SymetricScheme(Enum):
	MIRRORED = 0
	FRENCH = 1
	ENGLISH = 2
	INVERTED = 3
	B2B = 4
    
class FootballScheduler:
    '''
    Football Scheduler class. Initializes the model and decision variables, and handles modeling logic.
    
    Attributes:
		N: Number of teams
		K: Number of rounds
		I_s: List of top teams
		scheme: SymetricScheme
		x: Decision variable x[i,j,k]
		y: Decision variable y[i,k]
		w: Decision variable w[i,k]
		model: SCIP model
    '''
    def __init__(self, N: int, K:int, I_s: list[int], scheme: SymetricScheme):
        if(max(I_s)) > N:
            raise ValueError("I_s must be a subset of teams I")
        self.N = N
        self.K = K
        self.I_s = I_s
        self.scheme = scheme
        self.x = {}
        self.y = {}
        self.w = {}
        self.model = Model("Football Scheduler")

    def __instance_vars(self):
        '''
		Define the decision variables
		Variables:
			x[i,j,k] = 1 if team i plays against team j in round k
			y[i,k] = 1 if team i plays at home in round k
			w[i,k] = 1 if team i plays away in round k
		'''
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.K):
                    self.x[i,j,k] = self.model.addVar(vtype="B", name=f"x_{i}_{j}_{k}")

        for i in range(self.N):
            for k in range(self.K):
                self.y[i, k] = self.model.addVar(vtype="B", name=f"y_{i}_{k}")
                self.w[i, k] = self.model.addVar(vtype="B", name=f"w_{i}_{k}")

    def __instance_base_constraints(self):
        """
        Define the base constraints
        Constraints:
			- Double round robin constraints
			- Compactness constraints
			- Top-teams constraints
        """
        # Double round robin constraints.
        for i in range(self.N):
            for j in range(self.K):
                if i == j:
                    continue
                # C1- every team faces every other team once in the first half
                self.model.addCons(
                    quicksum(self.x[i, j, k] + self.x[j, i, k] for k in range(self.N))
                    == 1,
                    name=f"match_first_half_{i}_{j}",
                )
                # C2 - every team faces every other team once in the second half
                self.model.addCons(
					quicksum(self.x[i, j, k] + self.x[j, i, k] for k in range(self.N, self.K, 1)) == 1,
					name=f"match_second_half_{i}_{j}",
				)
                # C3 - exactly one of the two games is played at home while the other one is played away
                self.model.addCons(
					quicksum(self.x[i, j, k] for k in range(self.K)) == 1,
					name=f"match_second_half_{i}_{j}",
				)

        # Compactness constraints
        for j in range(self.N):
            for k in range(self.K):
                # C4 - all teams must play one match in each round.
                self.model.addCons(
					quicksum(self.x[i, j, k] + self.x[j, i, k] for i in range(self.N) if i != j) == 1,
					name=f"one_match_per_round_{j}_{k}",
				)

        # Top-teams constraints
        for i in [x for x in range(self.N) if x not in self.I_s]:
            for k in range(self.K):
                for j in self.I_s:
                    # C5 - No non-top team be required to play against any of the top teams in consecutive matches.
                    self.model.addCons(
						quicksum(self.x[i, j, k] + self.x[j, i, k] + self.x[i, j, k + 1] + self.x[j, i, k + 1])
						<= 1,
						name=f"top_team_cons_{i}_{j}_{k}",
					)