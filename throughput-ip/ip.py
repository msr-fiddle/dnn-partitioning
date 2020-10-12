# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
from gurobipy import *

if len(sys.argv) < 2:
    raise 'no argument given'
elif sys.argv[1] == 'contig':
    FORCE_CONTIGUOUS_FORWARD = True
elif sys.argv[1] == 'noncontig':
    FORCE_CONTIGUOUS_FORWARD = False
else:
    raise 'argument should be contig/noncontig'

DIFFERENT_MAXIMA = False
if len(sys.argv) >= 3 and sys.argv[2] == 'diffmaxima':
    DIFFERENT_MAXIMA = True
print('DIFFERENT_MAXIMA =', DIFFERENT_MAXIMA)

ALSO_FORCE_ON_CPUS = True  # only relevant if FORCE_CONTIGUOUS_FORWARD = True

graph = json.load(sys.stdin)

maxFpgas = graph['maxFPGAs']
maxCpus = graph['maxCPUs']
maxSizePerFpga = graph['maxSizePerFPGA']
nodes = {}
arbitraryNumber = -2000000000
for node in graph['nodes']:
    if 'size' not in node:
        node['size'] = 0.0
    if 'colorClass' not in node:
        node['colorClass'] = arbitraryNumber
        arbitraryNumber += 1
    if 'containedNodes' in node:
        raise "input should not have containedNodes"
    nodes[node['id']] = node

outgoingConnectionCost = {}
for edge in graph['edges']:
    u = edge['sourceId']
    v = edge['destId']
    if u in outgoingConnectionCost:
        if abs(outgoingConnectionCost[u] - edge['cost']) > 1e-6:
            raise ("node " + str(u) + " has two different outgoing connection costs")
    else:
        outgoingConnectionCost[u] = edge['cost']
# set 0 where a node had no outgoing edges
for node_id, node in nodes.items():
    if node_id not in outgoingConnectionCost:
        outgoingConnectionCost[node_id] = 0.0
        
from gurobipy import *

# IP to maximize throughput when pipelining = minimize max-load on any machine

model = Model("minimize_maxload")
model.setParam("LogToConsole", 0)
model.setParam("LogFile", "gurobi.log")
model.setParam("MIPGap", 0.01)
model.setParam("TimeLimit", 1200)
model.setParam("MIPFocus", 1)


# create variables
x = {} # map from (node_id, machine_id) to variable
# cpus are 0..maxCpus-1, fpgas are maxCpus..maxCpus+maxFpgas-1
# (different than the write-up in the paper)
for node_id, node in nodes.items():
    for machine_id in range(maxCpus + maxFpgas):
        x[node_id, machine_id] = model.addVar(vtype = GRB.BINARY)
        if (not node['supportedOnFpga']) and machine_id >= maxCpus:
            # make sure it's False and not 'False' in the input file...
            model.addConstr(x[node_id, machine_id] == 0)

            
# schedule every node on exactly one machine
for node_id, node in nodes.items():
    times_scheduled = LinExpr()
    for machine_id in range(maxCpus + maxFpgas):
        times_scheduled += x[node_id, machine_id]
    model.addConstr(times_scheduled == 1)

    
# size (knapsack) constraints (only on FPGAs)
for machine_id in range(maxCpus, maxCpus + maxFpgas):
    fpga_size = LinExpr()
    for node_id, node in nodes.items():
        fpga_size += node['size'] * x[node_id, machine_id]
    model.addConstr(fpga_size <= maxSizePerFpga)

    
# contiguity constraints
if FORCE_CONTIGUOUS_FORWARD:
    z = {}  # map from (node_id, machine_id) to variable. only for forward nodes
    for machine_id in range(maxCpus + maxFpgas):
        if machine_id < maxCpus and not ALSO_FORCE_ON_CPUS:
            continue
        for node_id, node in nodes.items():
            if not node['isBackwardNode']:
                z[node_id, machine_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
                model.addConstr(z[node_id, machine_id] >= x[node_id, machine_id])
        for edge in graph['edges']:
            u = edge['sourceId']
            v = edge['destId']
            if (not nodes[u]['isBackwardNode']) and (not nodes[v]['isBackwardNode']):
                model.addConstr(z[v, machine_id] <= z[u, machine_id])
                model.addConstr(z[v, machine_id] <= x[v, machine_id] - x[u, machine_id] + 1)

            
# CommIn, CommOut
comm_in = {}
comm_out = {}
for machine_id in range(maxCpus, maxCpus + maxFpgas):
    for node_id, node in nodes.items():
        comm_in[node_id, machine_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
        comm_out[node_id, machine_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
    for edge in graph['edges']:
        u = edge['sourceId']
        v = edge['destId']
        model.addConstr(comm_in[u, machine_id] >= x[v, machine_id] - x[u, machine_id])
        model.addConstr(comm_out[u, machine_id] >= x[u, machine_id] - x[v, machine_id])

        
# load    
fw_load = []
bw_load = []
for machine_id in range(maxCpus + maxFpgas):
    fw_load.append(model.addVar(vtype = GRB.CONTINUOUS, lb=0.0))
    bw_load.append(model.addVar(vtype = GRB.CONTINUOUS, lb=0.0))

# CPU load
for machine_id in range(maxCpus):
    fw_cpu_load = LinExpr()
    bw_cpu_load = LinExpr()
    for node_id, node in nodes.items():
        if not node['isBackwardNode']:
            fw_cpu_load += node['cpuLatency'] * x[node_id, machine_id]
        else:
            bw_cpu_load += node['cpuLatency'] * x[node_id, machine_id]
        # the CPUs don't pay for communication (data is already in RAM)
    model.addConstr(fw_load[machine_id] == fw_cpu_load)
    model.addConstr(bw_load[machine_id] == bw_cpu_load)

    
# FPGA load
for machine_id in range(maxCpus, maxCpus + maxFpgas):
    fw_fpga_load = LinExpr()
    bw_fpga_load = LinExpr()
    for node_id, node in nodes.items():
        if not node['isBackwardNode']:
            fw_fpga_load += node['fpgaLatency'] * x[node_id, machine_id]
            # model with "calls": communication NOT overlapped with compute
            # so we add communication here
            fw_fpga_load += outgoingConnectionCost[node_id] * comm_in[node_id, machine_id]
            # note: a forward outside node that has communication to this subgraph will contribute
            # to the forward cost, even if it only has an edge to the backward subgraph.
            # that probably rarely happens though
            fw_fpga_load += outgoingConnectionCost[node_id] * comm_out[node_id, machine_id]
            # can instead overlap compute and communication - easy modification (see Appendix G.1)
        else:
            bw_fpga_load += node['fpgaLatency'] * x[node_id, machine_id]
            # model with "calls": communication NOT overlapped with compute
            # so we add communication here
            bw_fpga_load += outgoingConnectionCost[node_id] * comm_in[node_id, machine_id]
            bw_fpga_load += outgoingConnectionCost[node_id] * comm_out[node_id, machine_id]
            # can instead overlap compute and communication - easy modification
    model.addConstr(fw_load[machine_id] == fw_fpga_load)
    model.addConstr(bw_load[machine_id] == bw_fpga_load)


# the max-load that we are minimizing
MaxLoad = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
if not DIFFERENT_MAXIMA:
    for machine_id in range(maxCpus + maxFpgas):
        model.addConstr(MaxLoad >= fw_load[machine_id] + bw_load[machine_id])
else:
    FwMaxLoad = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
    BwMaxLoad = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
    for machine_id in range(maxCpus + maxFpgas):
        model.addConstr(FwMaxLoad >= fw_load[machine_id])
        model.addConstr(BwMaxLoad >= bw_load[machine_id])
    model.addConstr(MaxLoad == FwMaxLoad + BwMaxLoad)
    
    

# colocation constraints (color classes)
color_class_to_ids = {}
for node_id, node in nodes.items():
    cc = node["colorClass"]
    if cc not in color_class_to_ids:
        color_class_to_ids[cc] = []
    color_class_to_ids[cc].append(node_id)
for cc, ids in color_class_to_ids.items():
    for i in range(len(ids) - 1):
        #print('merging ', ids[i], ' and ', ids[i+1])
        for machine_id in range(0, maxCpus + maxFpgas):
            model.addConstr(x[ids[i], machine_id] == x[ids[i+1], machine_id])

model.setObjective(MaxLoad, GRB.MINIMIZE)

print('Running optimizer...')
sys.stdout.flush()
model.optimize()

if model.Status == GRB.Status.INFEASIBLE:
    raise "infeasible"
elif model.Status == GRB.Status.OPTIMAL:
    print("Value is:", MaxLoad.X)
else:
    raise "Wrong status code"

print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')

result = {}
result['maxLoad'] = MaxLoad.X
debugMaxLoad = 0.0  # will compute same thing manually
result['fpgas'] = []
result['cpus'] = []
for machine_id in range(maxCpus + maxFpgas):
    resultMachine = {}
    
    # compute load of machine_id
    # we do not use load[machine_id].X as that only needs to be an upper bound
    # for the real load (as comm_in and comm_out need not be tight)
    resultMachine['load'] = 0.0
    resultMachine['nodes'] = []
    debugTotalSize = 0.0
    for node_id, node in nodes.items():
        if x[node_id, machine_id].X > 0.99:
            resultMachine['nodes'].append(node_id)
            if machine_id < maxCpus:
                resultMachine['load'] += node['cpuLatency']
            else:
                resultMachine['load'] += node['fpgaLatency']
            debugTotalSize += node['size']
    if machine_id >= maxCpus: # fpga
        assert(debugTotalSize <= maxSizePerFpga)
        nodesWithOutgoingCommunication = set()
        for edge in graph['edges']:
            u = edge['sourceId']
            v = edge['destId']
            if (u in resultMachine['nodes']) ^ (v in resultMachine['nodes']):
                # incoming or outgoing edge
                nodesWithOutgoingCommunication.add(u)
        for u in nodesWithOutgoingCommunication:
            resultMachine['load'] += outgoingConnectionCost[u]
    
    debugMaxLoad = max(debugMaxLoad, resultMachine['load'])
    if machine_id < maxCpus:
        result['cpus'].append(resultMachine)
    else:
        result['fpgas'].append(resultMachine)

del model
disposeDefaultEnv()
print(json.dumps(result))
