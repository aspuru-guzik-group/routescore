# subway_maps

Documentation for using the RouteScore/subway maps project code.


To reset and run _**everything**_ from scratch (including making figures):
- Run `run_everything.py`


To reset the files to a clean state:
- Run `reset_all.py`


To run all the _RouteScore_ calculations for the laser molecules:
- Run `full_routine.py`


To make all figures from scratch:
- Run `all_figs.py`

A few notes:
- `routescore.py` is used to calculate _RouteScore_, but must be called from another file and receive information about the route and molecules from that file.
- _RouteScores_ are calculated from a `RS_xxx.py` file.
- To understand how a _RouteScore_ is calculated, follow the comments in `RS_Base.py`. The process is very similar for all other route types.
- There is also an example that you can follow in the `Example_calc.ipynb` Jupyter notebook.
