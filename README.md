# Efficient Algorithms for Device Placement of DNN Graph Operators - code

This code package contains algorithms and input files (DNN workloads) from the paper "Efficient Algorithms for Device Placement of DNN Graph Operators" published at NeurIPS 2020.
It allows one to reproduce the results in the paper, as well as run the partitioning algorithms on other workloads.

Throughout this package (code and input files), "FPGA" is used is to refer to an arbitrary accelerator. In particular, for layer-granularity graphs we actually use profiled processing time (latency) data from a GPU deployment scenario.

## Input format

All our algorithms take as input a JSON file with the following format (all fields are mandatory unless indicated otherwise). This format closely follows our model (see Section "Computational Model" in the paper):
* `maxSizePerFPGA` (floating-point): a memory size limit of a single FPGA/accelerator, in bytes,
* `maxFPGAs` (integer): number of accelerators (`k` from the paper),
* `maxCPUs` (integer): number of CPU cores (`\ell` from the paper),
* `nodes` (array): for each node (layer or operator in the DNN):
    * `id` (integer): unique ID of node,
    * `supportedOnFpga` (boolean or 0/1 integer): whether the node can be placed on an accelerator,
    * `cpuLatency` (floating-point): time it takes to execute this node on a CPU,
    * `fpgaLatency` (floating-point): time it takes to execute this node on an accelerator,
    * `isBackwardNode` (boolean or 0/1 integer): whether this is a backward node (only used in training workloads),
    * `colorClass` (integer, optional): if two nodes share the same color class, then they need to be colocated on the same device,
    * `size` (floating-point): memory taken by this node on an accelerator, in bytes,
* `edges` (array): for each edge:
    * `sourceId` (integer): the ID of the tail of the edge (edge from `sourceId` to `destId`),
    * `destId` (integer): the ID of the head of the edge,
    * `cost` (floating-point): time it takes to transfer the output of node with ID `sourceId` between an accelerator and CPU DRAM. Should be the same for all edges going out of the same node.

Other debug information may be present in the input files, such as `name`s or `layerId`s on nodes and `size`s on edges.




## Throughput maximization (max-load minimization)

Here we give two algorithms: DP (dynamic programming) and IP (integer programming) based. We also give several baselines. Input files are placed in the directory `throughput-inputs`.

### Dynamic programming solution

The solution is implemented in `throughput-dp/dp.cpp`. It is a single C++ file (using one header-only library for JSON parsing) and can be compiled with a recent version of `gcc` by running e.g. `g++ -O3 throughput-dp/dp.cpp -o dp.exe`.

The compiled program takes the input network in the JSON format outlined above via standard input. It can be executed as follows: `./dp.exe < throughput-inputs/OperatorGraphs/bert_l-3_inference.json`. It produces an output JSON file on standard output and diagnostic messages on standard error.

### Integer programming solution

This solution requires the solver Gurobi to be installed and have an active license.
The solution is implemented in Python 3.7.

To run the solution, execute `python throughput-ip/ip.py contig` or `python throughput-ip/ip.py noncontig` depending on whether contiguous splits are desired.
It also takes the input on standard input, so an example run could be `python throughput-ip/ip.py contig < throughput-inputs/OperatorGraphs/bert_l-3_inference.json`. The Gurobi solver will print information about the progress of the optimization to file `gurobi.log`.

Further parameters than can be adjusted include:
* the line `model.setParam("MIPGap", 0.01)` controls the optimality gap at which optimization will stop (currently 1%),
* the line `model.setParam("TimeLimit", 1200)` controls the time limit after which optimization will stop (currently 20 minutes),
* the line `model.setParam("MIPFocus", 1)` controls the solver strategy (currently it should focus on finding as good solutions as possible rather than refining the lower bound),
* a flag `ALSO_FORCE_ON_CPUS` in `ip.py` toggles whether contiguity should also be forced on subgraphs that go onto CPU cores,
* a second command-line argument `diffmaxima` can be used to toggle optimization of the alternative objective function `max_i FW_i + max_i BW_i` (see Appendix C).

### Scotch baseline

First compile the DP solution. Then run it with a command-line parameter `-scotch`, e.g.: `./dp.exe -scotch < throughput-inputs/OperatorGraphs/bert_l-3_inference.json`. For this to work, Scotch needs to be installed and its `bin` directory should be added to the system PATH. To check if this is the case, run `gpart` in the console. Alternatively, you can change the constant `scotchExecutable` in `dp.cpp`.

### Human-expert baseline

We provide human-expert splits in the directory `human-experts`. The `dp` program can compute the max-load (Time-Per-Sample) for a given split file using the command-line parameter `-expert`. The second parameter should be the split file. E.g. `./dp.exe -expert human-experts/resnet50_inference_expert.json < throughput-inputs/LayerGraphs/resnet50_inference.json`. If there is no human expert split for a training workload, use the corresponding inference split - the `dp` program will infer the split for the backward pass from the split for the forward pass.

### Local search baseline

Add an option `-localSearch` and a second parameter specifying the number of restarts (we used 10).

### PipeDream baseline

Use option `-pipeDream`.

### Place everything on a single FPGA

Use option `-oneFpga`.





## Latency minimization

Here we give an IP (integer programming) based algorithm,
as well as two baselines. Input files are placed in the directory `latency-inputs`.

### Integer programming solution

This solution requires the solver `Gurobi` to be installed and have an active license.
The solution is implemented in Python 3.7.

To run the solution, execute `python latency-ip/lip.py`.
It also takes the input on standard input, so an example run could be `python latency-ip/lip.py < latency-inputs/OperatorGraphs/bert_l-3_inference.json`. The Gurobi solver will print information about the progress of the optimization to file `gurobi.log`.

Parameters that can be adjusted include:
* the same three Gurobi optimizer parameters as for throughput maximization above (currently the time limit is set to 1 hour),
* a parameter `MAX_SUBGRAPHS_PER_FPGA` that allows one to obtain non-contiguous splits. This is the `q` parameter from Appendix A.

### Greedy baseline

The solution, described in Appendix F, is implemented in `latency-greedy/greedy.cpp`. It is a single C++ file (using one header-only library for JSON parsing) and can be compiled with a recent version of `gcc` by running e.g. `g++ -O3 latency-greedy/greedy.cpp -o greedy.exe`.

The compiled program takes the input network in the JSON format outlined above via standard input. It can be executed as follows: `./greedy.exe < latency-inputs/OperatorGraphs/bert_l-3_inference.json`. It produces an output JSON file on standard output and diagnostic messages on standard error.

### Max-load DP baseline

The same program can alternatively take any split (in the output JSON format produced by our implementations) and compute the minimum single-sample latency that can be obtained with this split. To do this, run `greedy.exe` with two command-line parameters: `-computeLatency` and the path to the JSON file containing the split. The input workload should still be given via standard input.

Our second baseline for latency minimization, described in Appendix F, consists in running the throughput-maximization (max-load minimization) DP solution and computing the latency that it obtains. To do so, run e.g.:
* `./dp.exe < latency-inputs/OperatorGraphs/bert_l-12_inference.json > bert-l12-split.json`
* `./greedy.exe -computeLatency bert-l12-split.json < latency-inputs/OperatorGraphs/bert_l-12_inference.json`

### Human expert baseline

See "max-load DP baseline" above - the same program can compute the latency for a given split. Run e.g.:
* `./greedy.exe -computeLatency human-experts/bert24_inference_expert.json < latency-inputs/LayerGraphs/bert24_inference.json`

### Scotch baseline

First run `dp` with `-scotch` option (see above) to produce the split. Then use `greedy` to compute the latency of the split. E.g.:
* `./dp.exe -scotch < latency-inputs/OperatorGraphs/bert_l-12_inference.json > bert-l12-split-scotch.json`
* `./greedy.exe -computeLatency bert-l12-split-scotch.json < latency-inputs/OperatorGraphs/bert_l-12_inference.json`






## Legal notices

**Trademarks**
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

We use the [JSON for Modern C++](https://github.com/nlohmann/json) library, copyright (c) 2013-2020 Niels Lohmann, licensed under the MIT license.
