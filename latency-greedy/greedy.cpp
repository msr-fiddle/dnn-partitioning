// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>
#include <limits>
#include <sstream>
#include <iomanip>

#include "../nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;


// begin trivial helper stuff
ostream& dbg = cerr;

void fail (const string &s) {
    cout << "FAIL: " << s << endl;
    dbg << "FAIL: " << s << endl;
    exit(1);
}

void warn (const string &s) {
    dbg << "WARNING: " << s << endl;
}

template <typename T>
ostream& operator << (ostream &s, const vector<T> &v) {
    s << "[";
    for (const T &x : v) {
        s << x << " ";
    }
    s << "]";
    return s;
}

template <typename T>
string to_string (const vector<T> &v) {
    stringstream ss;
    ss << v;
    return ss.str();
}

template <typename T>
void append (vector<T> &v, const vector<T> &w) {
    v.insert(v.end(), w.begin(), w.end());
}

// Union-Find
int ufFind(unordered_map<int,int> &ufTab, int a) {
    if (ufTab[a] == a) return a;
    return ufTab[a] = ufFind(ufTab, ufTab[a]);
}
void ufUnion(unordered_map<int,int> &ufTab, int a, int b) {
    ufTab[ufFind(ufTab, a)] = ufFind(ufTab, b);
}
// end trivial helper stuff


// Node, Edge, InputInstance reflect the JSON structure
struct Node {
    int id;
    int supportedOnFpga;
    double cpuLatency;
    double fpgaLatency;
    double size;
    int isBackwardNode;
    int colorClass;
    vector<int> containedSubnodes;
    int layerId;
};



void from_json(const json& j, Node &n) {
    j.at("id").get_to(n.id);
    j.at("supportedOnFpga").get_to(n.supportedOnFpga);

    j.at("fpgaLatency").get_to(n.fpgaLatency);

    j.at("cpuLatency").get_to(n.cpuLatency);

    if (j.count("size")) {
        j.at("size").get_to(n.size);
    } else {
        n.size = 0.0;
    }

    if (j.count("hasBackwardNode")) {
        j.at("isBackwardNode").get_to(n.isBackwardNode);
        if (n.isBackwardNode) {
            fail("for inference there shouldn't be backward nodes");
        }
    }

    if (j.count("colorClass")) {
        j.at("colorClass").get_to(n.colorClass);
    } else {
        static int freshNumber = numeric_limits<int>::min();
        // some number that will never come up as a color class in the input
        n.colorClass = ++freshNumber;
    }

    if (j.count("containedSubnodes")) {
        fail("the input file should not have containedSubnodes");
    } else {
        n.containedSubnodes = {n.id};
    }
}


void to_json(json &j, const Node &n) {
    j = json{
        {"id", n.id},
        {"supportedOnFpga", n.supportedOnFpga},
        {"cpuLatency", n.cpuLatency},
        {"fpgaLatency", n.fpgaLatency},
        {"size", n.size},
        {"isBackwardNode", n.isBackwardNode},
        {"colorClass", n.colorClass},
        {"containedSubnodes", n.containedSubnodes}
    };
}


struct Edge {
    int sourceId;
    int destId;
};

unordered_map<int,double> outgoingConnectionCost;


void from_json(const json&j, Edge &n) {
    j.at("sourceId").get_to(n.sourceId);
    j.at("destId").get_to(n.destId);
    double cost;
    j.at("cost").get_to(cost);
    if (outgoingConnectionCost.count(n.sourceId)) {
        if (abs(outgoingConnectionCost[n.sourceId] - cost) > 1e-6) {
            fail("node " + to_string(n.sourceId) + " has two different outgoing edge costs");
        }
    } else {
        outgoingConnectionCost[n.sourceId] = cost;
    }
}


void to_json(json &j, const Edge &n) {
    j = json{{"sourceId", n.sourceId}, {"destId", n.destId}};
}


struct InputInstance {
    double maxSizePerFpga;
    int maxFpgas;
    int maxCpus;
    vector<Node> nodes;
    vector<Edge> edges;
};


void from_json(const json&j, InputInstance &n) {
    j.at("maxSizePerFPGA").get_to(n.maxSizePerFpga);
    j.at("maxFPGAs").get_to(n.maxFpgas);
    j.at("maxCPUs").get_to(n.maxCpus);
    j.at("nodes").get_to(n.nodes);
    j.at("edges").get_to(n.edges);
}


void to_json(json &j, const InputInstance &n) {
    j = json{
        {"maxSizePerFPGA", n.maxSizePerFpga},
        {"maxFPGAs", n.maxFpgas},
        {"maxCPUs", n.maxCpus},
        {"nodes", n.nodes},
        {"edges", n.edges}
    };
}




struct ResultMachine {
    vector<int> nodes;
};


struct Result {
    double totalLatency;
    vector<ResultMachine> fpgas;
    vector<ResultMachine> cpus;

    void printCounts () const {
        int countCpu = 0;
        for (const ResultMachine &rm : cpus) {
            countCpu += rm.nodes.size();
        }
        vector<int> countsFpga;
        for (const ResultMachine &rm : fpgas) {
            countsFpga.push_back(rm.nodes.size());
        }
        cout << "result counts: cpu: " << countCpu << "; fpgas: " << countsFpga << endl;
    }
};


void from_json(const json &j, ResultMachine &rs) {
    j.at("nodes").get_to(rs.nodes);
}


void from_json(const json &j, Result &r) {
    if (j.count("totalLatency")) {
        j.at("totalLatency").get_to(r.totalLatency);
    } else {
        r.totalLatency = -1000000000.0;
    }
    j.at("cpus").get_to(r.cpus);
    j.at("fpgas").get_to(r.fpgas);
}


void to_json(json &j, const ResultMachine &rs) {
    j = json{{"nodes", rs.nodes}};
}


void to_json(json &j, const Result &r) {
    j = json{{"totalLatency", r.totalLatency}, {"fpgas", r.fpgas}, {"cpus", r.cpus}};
}


vector<vector<int>> getTopologicalSort (const unordered_map<int,vector<int>>& containedSubnodes,
                                        const unordered_map<int,vector<int>>& outgoing) {
    // takes a contracted graph, outputs toposort
    // ignores self-loops
    unordered_map<int,int> indegree;
    for (const auto &it : outgoing) {
        for (int v : it.second) {
            if (it.first == v) continue;
            ++indegree[v];
        }
    }
    vector<int> deg0vertices;
    for (const auto &it : containedSubnodes) {
        if (indegree[it.first] == 0) {
            deg0vertices.emplace_back(it.first);
        }
    }
    
    vector<vector<int>> result;
    int visitedNodes = 0;
    while (!deg0vertices.empty()) {
        int v = deg0vertices.back(); // DFS-ish traversal
        deg0vertices.pop_back();

        // visiting v
        result.push_back(containedSubnodes.at(v)); // result need not be toposorted internally!
        ++visitedNodes;

        if (outgoing.count(v)) {
            for (int w : outgoing.at(v)) {
                // edge v->w
                if (v == w) continue;
                --indegree[w];
                if (indegree[w] == 0) {
                    deg0vertices.emplace_back(w);
                }
            }
        }
    }
    if (visitedNodes != containedSubnodes.size()) {
        fail("contracted graph is not acyclic");
    }
    return result;
}



struct Greedy {
    InputInstance &ii;
    
    // edges
    unordered_map<int,vector<int>> outgoing, incoming;

    unordered_map<int,int> colorClass;
    unordered_map<int,vector<int>> colorClasses;
    unordered_map<int,double> size, cpuLatency, fpgaLatency;

    Greedy (InputInstance &_ii) :
        ii(_ii) {
        // build incoming and outgoing
        for (const Node &n : ii.nodes) {
            outgoing[n.id];
            incoming[n.id];
        }
        for (const Edge &e : ii.edges) {
            outgoing[e.sourceId].push_back(e.destId);
            incoming[e.destId].push_back(e.sourceId);
        }
        // build colorClass
        for (const Node &n : ii.nodes) {
            colorClass[n.id] = n.colorClass;
        }
        // build colorClasses
        for (const Node &n : ii.nodes) {
            colorClasses[n.colorClass].push_back(n.id);
        }
        // build size, cpuLatency, fpgaLatecy
        for (const Node &n : ii.nodes) {
            size[n.id] = n.size;
            cpuLatency[n.id] = n.cpuLatency;
            fpgaLatency[n.id] = n.fpgaLatency;
        }
    }


    vector<vector<int>> inseparableBlocks; // sorted in topological order
    vector<double> inseparableBlockSizes;


    void makeBlocks() {
        // set up a reachability thing on color classes
        set<pair<int,int>> reachable;
        for (const Edge &e : ii.edges) {
            reachable.insert({colorClass[e.sourceId], colorClass[e.destId]});
        }
        // run Floyd-Warshall (O(n^3 log n) runtime... might need to replace with something faster)
        for (const auto &kt : colorClasses) {
            for (const auto &it : colorClasses) {
                if (reachable.count({it.first, kt.first})) {
                    for (const auto &jt : colorClasses) {
                        if (reachable.count({kt.first, jt.first})) {
                            reachable.insert({it.first, jt.first});
                        }
                    }
                }
            }
        }
        // fuse color classes using Union-Find
        unordered_map<int,int> ufTab;
        for (const auto &it : colorClasses) {
            ufTab[it.first] = it.first;
        }
        for (const auto &it : colorClasses) {
            for (const auto &jt : colorClasses) {
                if (reachable.count({it.first, jt.first}) && reachable.count({jt.first, it.first})) {
                    ufUnion(ufTab, it.first, jt.first);
                }
            }
        }
        // now build a contracted instance
        unordered_map<int,vector<int>> containedSubnodes, newOutgoing;
        for (const Node &n : ii.nodes) {
            containedSubnodes[ufFind(ufTab, n.colorClass)].push_back(n.id);
        }
        for (const Edge &e : ii.edges) {
            int a = ufFind(ufTab, colorClass[e.sourceId]);
            int b = ufFind(ufTab, colorClass[e.destId]);
            newOutgoing[a].push_back(b);
        }

        inseparableBlocks = getTopologicalSort(containedSubnodes, newOutgoing);

        for (const vector<int> &block : inseparableBlocks) {
            inseparableBlockSizes.push_back(0.0);
            for (int v : block) {
                inseparableBlockSizes.back() += size[v];
            }
            if (inseparableBlockSizes.back() > ii.maxSizePerFpga) {
                warn("there is an inseparable block that is too large for FPGA");
            }
        }
    }


    // turns any (possibly non-contiguous) split into a contiguous one
    // by dividing non-contiguous subgraphs into contiguous ones.
    // this is done greedily (possibly non-optimally for the latency objective)
    // using a simple dynamic program.
    // (this function will not modify a split that is already contiguous)
    Result reworkSplitIntoContiguous (const Result &r) const {
        Result newR;
        newR.totalLatency = -1;
        newR.cpus = r.cpus; // no change in CPUs

        for (const ResultMachine &rs : r.fpgas) {
            // break rs up into contiguous pieces, if necessary
            unordered_set<int> nodesInSubgraph(rs.nodes.begin(), rs.nodes.end());
            unordered_map<int,vector<int>> incoming; // incoming[v] = nodes u s.t. there is edge u->v
            for (const Edge &e : ii.edges) {
                incoming[e.destId].push_back(e.sourceId);
            }

            // now do the DP
            unordered_map<int,int> dp;
            function<int(int)> goDp = [&] (int v) -> int {
                if (!dp.count(v)) {
                    if (nodesInSubgraph.count(v)) {
                        dp[v] = 1;
                        for (int u : incoming[v]) {
                            dp[v] = max(dp[v], goDp(u) + (nodesInSubgraph.count(u) == 0));
                        }
                    } else {
                        dp[v] = 0;
                        for (int u : incoming[v]) {
                            dp[v] = max(dp[v], goDp(u));
                        }
                    }
                }
                return dp[v];
            };
            int maxDp = 1;
            for (const Node &n : ii.nodes) {
                goDp(n.id);
                maxDp = max(maxDp, dp[n.id]);
            }
            //dbg << "pieces: " << maxDp << endl;

            // now break up the subgraph into maxDp many pieces
            const int offset = newR.fpgas.size();
            for (int piece = 1; piece <= maxDp; ++piece) {
                newR.fpgas.emplace_back();
            }
            for (int id : rs.nodes) {
                newR.fpgas[offset + dp[id] - 1].nodes.push_back(id);
            }
        }

        return newR;
    }


    void checkSizeConstraint (const Result &r) const {
        for (int i = 0; i < r.fpgas.size(); ++i) {
            double totalSizeOnDevice = 0.0; // just to check that the split is valid
            for (int nodeId : r.fpgas[i].nodes) {
                totalSizeOnDevice += size.at(nodeId);
            }
            if (totalSizeOnDevice > ii.maxSizePerFpga) {
                dbg << setprecision(13);
                dbg << "total size on fpga: " << totalSizeOnDevice << endl;
                dbg << "maxSizePerFpga:     " << ii.maxSizePerFpga << endl;
                warn("Out Of Memory");
            }
        }
    }


    // should only be run for contiguous FPGA contents
    // if your split is not such, then first run reworkSplitIntoContiguous
    double computeLatency (const Result &r) const {
        unordered_map<int,int> deviceId; // id of device (0=some cpu) where node is processed
        unordered_map<int,vector<int>> containedSubnodes, newOutgoing;
        // build the contracted graph where FPGAs (not CPUs) are contracted. also build `deviceId`
        for (const ResultMachine &rs : r.cpus) {
            for (int nodeId : rs.nodes) {
                assert(!deviceId.count(nodeId));
                deviceId[nodeId] = 0; // cpu
                containedSubnodes[nodeId] = {nodeId}; // singletons
            }
        }
        constexpr int OFFSET = -200'000;
        for (int i = 0; i < r.fpgas.size(); ++i) {
            for (int nodeId : r.fpgas[i].nodes) {
                assert(!deviceId.count(nodeId));
                deviceId[nodeId] = OFFSET + i; // fpgas indexed from OFFSET
                containedSubnodes[OFFSET + i].push_back(nodeId);
            }
        }
        for (const Edge &e : ii.edges) {
            int a = (deviceId[e.sourceId] == 0) ? e.sourceId : deviceId[e.sourceId];
            int b = (deviceId[e.destId] == 0) ? e.destId : deviceId[e.destId];
            newOutgoing[a].push_back(b);
        }

        vector<vector<int>> blocksToProcess = getTopologicalSort(containedSubnodes, newOutgoing);

        unordered_map<int,double> nodeLatency;
        double finalLatency = 0.0;
        for (const vector<int>& block : blocksToProcess) {
            int nodeId = block.at(0);
            assert(deviceId.count(nodeId));
            if (deviceId[nodeId] == 0) {
                // cpu
                assert(block.size() == 1);
                double maxPrecedessorLatency = 0.0;
                for (const int u : incoming.at(nodeId)) {
                    assert(nodeLatency.count(u));
                    maxPrecedessorLatency = max(maxPrecedessorLatency, nodeLatency[u]);
                }
                nodeLatency[nodeId] = cpuLatency.at(nodeId) + maxPrecedessorLatency;
                finalLatency = max(finalLatency, nodeLatency[nodeId]);
            } else {
                // fpga. compute nodeLatency for the entire subgraph now
                unordered_set<int> nodesInThisSubgraph(r.fpgas[deviceId[nodeId] - OFFSET].nodes.begin(), r.fpgas[deviceId[nodeId] - OFFSET].nodes.end());
                double incomingCommunicationCost = 0.0;
                double maxPredecessorLatency = 0.0;
                // look at all nodes with an edge to this subgraph
                for (const Node &u : ii.nodes) {
                    if (!nodesInThisSubgraph.count(u.id)) {
                        bool hasEdgeToThisSubgraph = false;
                        for (int v : outgoing.at(u.id)) {
                            if (nodesInThisSubgraph.count(v)) {
                                hasEdgeToThisSubgraph = true;
                                break;
                            }
                        }
                        if (hasEdgeToThisSubgraph) {
                            incomingCommunicationCost += outgoingConnectionCost[u.id];
                            assert(nodeLatency.count(u.id)); // should be computed already as blocksToProcess is toposorted
                            maxPredecessorLatency = max(maxPredecessorLatency, nodeLatency[u.id]);
                        }
                    }
                }
                double outgoingCommunicationCost = 0.0;
                double totalFpgaLatency = 0.0;
                // look at all nodes in this subgraph with an edge going out
                for (int u : nodesInThisSubgraph) {
                    totalFpgaLatency += fpgaLatency.at(u);
                    bool hasEdgeOutOfThisSubgraph = false;
                    for (int v : outgoing.at(u)) {
                        if (!nodesInThisSubgraph.count(v)) {
                            hasEdgeOutOfThisSubgraph = true;
                            break;
                        }
                    }
                    if (hasEdgeOutOfThisSubgraph) {
                        outgoingCommunicationCost += outgoingConnectionCost[u];
                    }
                }

                double subgraphLatency = maxPredecessorLatency + incomingCommunicationCost + totalFpgaLatency + outgoingCommunicationCost;

                // write down answer for entire subgraph
                for (int u : nodesInThisSubgraph) {
                    nodeLatency[u] = subgraphLatency;
                    finalLatency = max(finalLatency, nodeLatency[u]);
                }
            }
        }

        return finalLatency;
    }

    // greedily fill FPGAs with blocks while staying under the size limit
    Result runGreedy () const {
        int nextBlockIndex = 0;
        Result r;
        r.cpus.emplace_back();
        while (nextBlockIndex < inseparableBlocks.size()) {
            if (inseparableBlockSizes[nextBlockIndex] > ii.maxSizePerFpga || r.fpgas.size() == ii.maxFpgas) {
                // put on CPU
                append(r.cpus[0].nodes, inseparableBlocks[nextBlockIndex]);
                ++nextBlockIndex;
            } else {
                // can put at least this block on FPGA
                r.fpgas.emplace_back();
                double totalSizeOnThisFpga = 0.0;
                // put as many as possible
                while (nextBlockIndex < inseparableBlocks.size() && totalSizeOnThisFpga + inseparableBlockSizes[nextBlockIndex] <= ii.maxSizePerFpga) {
                    totalSizeOnThisFpga += inseparableBlockSizes[nextBlockIndex];
                    append(r.fpgas.back().nodes, inseparableBlocks[nextBlockIndex]);
                    ++nextBlockIndex;
                }
            }
        }
        r.totalLatency = computeLatency(r);
        return r;
    }
};




int main(int argc, char **argv) {
    json j;
    cin >> j;
    //dbg << "original instance: " << j.dump(4) << endl << endl << endl;
    InputInstance ii = j.get<InputInstance>();

    Greedy g(ii);

    if (argc > 1 && string(argv[1]) == string("-computeLatency")) {
        cout << "will just compute latency of given output" << endl;
        assert(argc > 2);
        ifstream outputFile(argv[2]);
        if (!outputFile) {
            fail("output filename does not exist");
        }
        json outJson;
        outputFile >> outJson;
        Result out = outJson.get<Result>();
        g.checkSizeConstraint(out);
        Result reworkedOut = g.reworkSplitIntoContiguous(out);
        double finalLatency = g.computeLatency(reworkedOut);
        cout << "the latency of given split is: " << setprecision(3) << fixed << finalLatency << endl;
    } else {
        g.makeBlocks();

        Result r = g.runGreedy();
        dbg << "finished successfully. objective = " << r.totalLatency << endl;
        //r.printCounts();
        cout << json(r).dump(4) << endl;
    }
}
