
# Repository for Running Label Propagation Algorithms on Author Graph Data
This repository contains code for running label propagation algorithms on
author graph data.

### Requirements

See `requirements/requirements.txt` for the required packages.

### Usage

#### SciServer

You need to convert the anonymized graph data from Elsevier into Dataframes
that contain mappings from `auid -> eids` and `eid -> auids`. These need to be put
into the `data` directory.

Then to run the algorithm

```bash
python -m src.run_algo --runtime sciserver
```
For other options, see the help message.

```bash
python -m src.run_algo --help
```

#### Elsevier

Then to run the algorithm

```bash
python -m src.run_algo --runtime elsevier
```
For other options, see the help message.

```bash
python -m src.run_algo --help
```

### Implementing a runtime

The algorithm works generall in three phases:

1. Get the prior data for that year and any number of prior years.
2. Run the label propagation algorithm
3. Update the posterior data for that year

A backend then needs to implement steps 1 and 3.

You need to implement the following functions

```python
MaybeSparseMatrix = Union[np.ndarray, sp.spmatrix]

get_data(
    year: int,
    logger: logging.Logger
) -> Tuple[MaybeSparseMatrix, np.ndarray, np.ndarray]:
```

This function accepts a year and a logger and returns a tuple of the following:
- The adjacency matrix
- The auids
- The prior for the auids

The second function you need to implement is

```python
def update_posterior(
    auids: np.ndarray,
    posterior_y_value: np.ndarray,
    year: int,
    logger: logging.Logger,
) -> None:
```

This function accepts the auids, the posterior_y_value, and the year and
updates the posterior values for that year. It's important to note that
if you parse the graph in pieces of disconnnected sets, this will update
the same file multiple times.

### TODO

- [ ] Finish tests for sciserver.py
- [ ] Add SocNL