# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
from gurobipy import *

MAX_SUBGRAPHS_PER_FPGA = 1
if len(sys.argv) >= 2:
    MAX_SUBGRAPHS_PER_FPGA = int(sys.argv[1])
print('MAX_SUBGRAPHS_PER_FPGA =', MAX_SUBGRAPHS_PER_FPGA)

graph = json.load(sys.stdin)

maxFpgas = graph['maxFPGAs']
maxSubgraphs = MAX_SUBGRAPHS_PER_FPGA * maxFpgas
# maxCpus is ignored: assume as many as the width of the graph (equivalently: assume infinitely many)
maxSizePerFpga = graph['maxSizePerFPGA']

nodes = {}
arbitraryNumber = -2000000000
for node in graph['nodes']:
    if 'size' not in node:
        node['size'] = 0.0
    if 'colorClass' not in node:
        node['colorClass'] = arbitraryNumber  # some "fresh" number
        arbitraryNumber += 1
    if 'containedNodes' in node:
        raise "input should not have containedNodes"
    if node['isBackwardNode']:
        raise "for inference, the input shouldn't have backward nodes"
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

# figure out a (loose) upper bound for optimum latency
# we will just sum up all cpuLatencies
latencyUpperBound = 0.0
for node_id, node in nodes.items():
    latencyUpperBound += node['cpuLatency']
print('latencyUpperBound = ', latencyUpperBound)

# what is the latency if we just put everything on CPU?

predecessors = {}
for node_id, node in nodes.items():
    predecessors[node_id] = []
for edge in graph['edges']:
    u = edge['sourceId']
    v = edge['destId']
    predecessors[v].append(u)

longestPathEndingAtNode = {}  # map from node_id to longest path ending at that node
def getLongestPathEndingAtNode(v):
    if v in longestPathEndingAtNode:
        return longestPathEndingAtNode[v]
    longestPathEndingAtNode[v] = nodes[v]['cpuLatency']
    for u in predecessors[v]:
        longestPathEndingAtNode[v] = max(longestPathEndingAtNode[v], getLongestPathEndingAtNode(u) + nodes[v]['cpuLatency'])
    return longestPathEndingAtNode[v]

longestPath = 0
for node_id, node in nodes.items():
    longestPath = max(longestPath, getLongestPathEndingAtNode(node_id))

print('latency when using only CPUs:', longestPath)



# IP to minimize latency

model = Model("minimize_latency")
model.setParam("LogToConsole", 0)
model.setParam("LogFile", "gurobi.log")
model.setParam("MIPGap", 0.01)
model.setParam("TimeLimit", 3600)
model.setParam("MIPFocus", 1)

# if this is too large, then the reformulated
# ex-quadratic constraints can behave funky
model.setParam("IntFeasTol", 1e-6)


# create variables
x = {} # map from (node_id, machine_id) to variable
# all cpus together are 0
# FPGA subgraphs start from 1 and are in blocks of MAX_SUBGRAPHS_PER_FPGA many
# (like in the paper)
for node_id, node in nodes.items():
    for machine_id in range(1 + maxSubgraphs):
        x[node_id, machine_id] = model.addVar(vtype = GRB.BINARY)
        if (not node['supportedOnFpga']) and machine_id > 0:
            model.addConstr(x[node_id, machine_id] == 0)

            
# schedule every node on exactly one machine
for node_id, node in nodes.items():
    times_scheduled = LinExpr()
    for machine_id in range(1 + maxSubgraphs):
        times_scheduled += x[node_id, machine_id]
    model.addConstr(times_scheduled == 1)

    
# size (knapsack) constraints (only on FPGAs)
# constraint is per-FPGA, not per-subgraph!
for i in range(1, maxFpgas + 1):
    fpga_size = LinExpr()
    for machine_id in range((i-1)*MAX_SUBGRAPHS_PER_FPGA+1, i*MAX_SUBGRAPHS_PER_FPGA+1):
        for node_id, node in nodes.items():
            fpga_size += node['size'] * x[node_id, machine_id]
    model.addConstr(fpga_size <= maxSizePerFpga)

    
# contiguity constraints
z = {}  # map from (node_id, machine_id) to variable
for machine_id in range(1, 1+maxSubgraphs):
    for node_id, node in nodes.items():
        z[node_id, machine_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
        model.addConstr(z[node_id, machine_id] >= x[node_id, machine_id])
    for edge in graph['edges']:
        u = edge['sourceId']
        v = edge['destId']
        model.addConstr(z[v, machine_id] <= z[u, machine_id])
        model.addConstr(z[v, machine_id] <= x[v, machine_id] - x[u, machine_id] + 1)

            
# CommIn, CommOut
comm_in = {}
comm_out = {}
for machine_id in range(1, 1+maxSubgraphs):
    for node_id, node in nodes.items():
        comm_in[node_id, machine_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
        comm_out[node_id, machine_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
    for edge in graph['edges']:
        u = edge['sourceId']
        v = edge['destId']
        model.addConstr(comm_in[u, machine_id] >= x[v, machine_id] - x[u, machine_id])
        model.addConstr(comm_out[u, machine_id] >= x[u, machine_id] - x[v, machine_id])


# Latency (only create variables)
latency = {}
for node_id, node in nodes.items():
    latency[node_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)


# subgraph start, finish (only create variables)
start = [None]
finish = [None]
for machine_id in range(1, 1+maxSubgraphs):
    start.append(model.addVar(vtype = GRB.CONTINUOUS, lb=0.0))
    finish.append(model.addVar(vtype = GRB.CONTINUOUS, lb=0.0))


# subgraph can't start before the incoming results are ready
for machine_id in range(1, 1+maxSubgraphs):
    for node_id, node in nodes.items():
        # quadratic constraint!
        # model.addConstr(start[machine_id] >= latency[node_id] * comm_in[node_id, machine_id])
        # rewrite it like so:
        model.addConstr(start[machine_id] >= latency[node_id]
                        - (1 - comm_in[node_id, machine_id]) * latencyUpperBound)


# finishing time of a subgraph
for machine_id in range(1, 1+maxSubgraphs):
    fpga_load = LinExpr()
    for node_id, node in nodes.items():       
        fpga_load += node['fpgaLatency'] * x[node_id, machine_id]
        # model with "calls": communication NOT overlapped with compute
        # so we add communication here
        fpga_load += outgoingConnectionCost[node_id] * comm_in[node_id, machine_id]
        fpga_load += outgoingConnectionCost[node_id] * comm_out[node_id, machine_id]
    model.addConstr(finish[machine_id] == start[machine_id] + fpga_load)


# latency constraints for nodes on CPU
for node_id, node in nodes.items():
    model.addConstr(latency[node_id] >= node['cpuLatency'] * x[node_id, 0])
for edge in graph['edges']:
    u = edge['sourceId']
    v = edge['destId']
    model.addConstr(latency[v] >= latency[u] + nodes[v]['cpuLatency'] * x[v, 0])


# latency for nodes on a subgraph
for machine_id in range(1, 1+maxSubgraphs):
    for node_id, node in nodes.items():
        # quadratic constraint!
        # model.addConstr(latency[node_id] >= x[node_id, machine_id] * finish[machine_id])
        # rewrite it like so:
        model.addConstr(latency[node_id] >= finish[machine_id]
                        - (1 - x[node_id, machine_id]) * latencyUpperBound)

        
# ordering of subgraphs assigned to the same FPGA
for i in range(1, 1+maxFpgas):
    for machine_id in range((i-1)*MAX_SUBGRAPHS_PER_FPGA+2, i*MAX_SUBGRAPHS_PER_FPGA+1):
        assert(MAX_SUBGRAPHS_PER_FPGA > 1)  # if == 1, then we shouldn't even be here
        model.addConstr(start[machine_id] >= finish[machine_id - 1])


# TotalLatency that we are minimizing
TotalLatency = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
for node_id, node in nodes.items():
    model.addConstr(TotalLatency >= latency[node_id])
        

# colocation constraints (color classes)
color_class_to_ids = {}
for node_id, node in nodes.items():
    cc = node["colorClass"]
    if cc not in color_class_to_ids:
        color_class_to_ids[cc] = []
    color_class_to_ids[cc].append(node_id)
for cc, ids in color_class_to_ids.items():
    for t in range(len(ids) - 1):
        #print('merging ', ids[t], ' and ', ids[t+1])
        # cpu
        model.addConstr(x[ids[t], 0] == x[ids[t+1], 0])
        # FPGAs
        for i in range(1, maxFpgas + 1):
            colocationConstraint = LinExpr()
            for machine_id in range((i-1)*MAX_SUBGRAPHS_PER_FPGA+1, i*MAX_SUBGRAPHS_PER_FPGA+1):
                colocationConstraint += x[ids[t], machine_id] - x[ids[t+1], machine_id]
            model.addConstr(colocationConstraint == 0)


model.setObjective(TotalLatency, GRB.MINIMIZE)

print('Running optimizer...')
sys.stdout.flush()
model.optimize()

if model.Status == GRB.Status.INFEASIBLE:
    raise "infeasible"
elif model.Status == GRB.Status.OPTIMAL:
    print("Value is:", TotalLatency.X)
else:
    raise "Wrong status code"

print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')

result = {}
result['totalLatency'] = TotalLatency.X
result['fpgas'] = []
result['cpus'] = []
for machine_id in range(1 + maxSubgraphs):
    resultMachine = {}
    
    resultMachine['nodes'] = []
    debugTotalSize = 0.0
    for node_id, node in nodes.items():
        if x[node_id, machine_id].X > 0.99:
            resultMachine['nodes'].append(node_id)
    if machine_id == 0:
        result['cpus'].append(resultMachine)
    else:
        result['fpgas'].append(resultMachine)

del model
disposeDefaultEnv()
print(json.dumps(result))
