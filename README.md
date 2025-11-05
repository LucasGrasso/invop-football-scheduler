# invop-football-scheduler

Implements a scheduler using a integer progamming model as described at [[1]](#References).

We encapusalate all modeling logic in the `FootballSchedulerModel` class. This class handles the correct initialization of the model as per [[1]](#References) and exposes an API for interacting with the model. For more info see _model.py_.

The _output_ folder contains two subfolders per `SymmetricScheme` (See _model.py_), one containing a "top teams" constraint and one not containing it (suffixed with `-basic`). In each folder you may find the outputed pyscipopt model, the int -> country map used and the solution as a latex table and as a pyscipopt output. When in a none basic model, the top teams used where `"ARG"` and `"BRA"` as per the paper.

Solution recovery logic is implemented in _parse.py_.

## Install depenencies

```bash
# create a new conda venv with with the deps
conda env create -f conda.env.yml
# and activate the venv
conda activate invop
```

## References:
- [1] G. Durán, E. Mijangos, and M. Frisk, “Scheduling the South American qualifiers to the 2018 FIFA World Cup by integer programming,” European Journal of Operational Research, vol. 262, no. 3, pp. 1035–1048, 2017.
- [2] Pyscipopt Docs.