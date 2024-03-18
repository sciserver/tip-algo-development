
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


### TODO

- [ ] Finish tests for sciserver.py
- [ ] Add SocNL