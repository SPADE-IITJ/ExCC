# ExCC
ExCC: External Memory Connected Components on Large Graphs

This repo contains the artifacts for ExCC. The only file required to run is ``excc_v1.cu``. As the name suggest, it is in the initial state. However, still runs perfectly file. ExCC assumes the graph to be given in the form of ``.egr`` format. For converting your MTX graphs from the _sparse suite matrix_ (TAMU), you can use ECL's converter code https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/index.html

### To run the code:

#### Compile:
```shell
nvcc -O3 -std=c++11 -arch=sm_86 excc_v1.cu -o excc
```

> Change the value of `-arch` flag as per your GPU.

#### Run

```shell
./excc graph batch_size(mb)
```

> For example, `./excc Agatha-2015.egr 512`

#### Run via scripts
There are two scripts in the repo:
1. `run_cc_exps.sh`: You can directly run this script. This will compile and run the code. Make sure you provide the correct path to your graphs' directory
2. `run_cc_exps.sh`: To run ExCC on different batch sizes and graphs.
