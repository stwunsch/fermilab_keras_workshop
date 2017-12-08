# Fermilab Keras Workshop

All examples and slides of the Keras workshop held in the Fermilab machine learning group from December 8, 2017 can be found in this respository.

## Software Setup

To run the plain Keras examples, use the script `init_virtualenv.sh` to set up the software. Then, source the environment with `source setup_virtualenv.sh`. This has to be done in every new shell.

For running the examples with the TMVA Keras interface, a ROOT installation above version 6.08 is required. The most convenient way to set up this environment is using a CVMFS software release. Run `source setup_cvmfs_lcg91.sh` to set up the software stack LCG 91 in a CVMFS enabled environment, e.g., CERN's lxplus machines.
