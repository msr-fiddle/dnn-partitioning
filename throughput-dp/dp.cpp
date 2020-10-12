// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>
#include <limits>
#include <sstream>

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
    for (const T &x : v) {
        s << x << " ";
    }
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

constexpr double INFTY = 1e30;
// end trivial helper stuff



bool CONTRACT_LAYERS = false;
bool IGNORE_ORPHANED_BACKWARD_NODES = false;
constexpr int IDEALS_LIMIT = 60'000;
constexpr int IDEALS_EXPLORATION_LIMIT = 2'000'000;



unordered_map<int,string> nameOfId;


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
    int dcc;
};


unordered_map<int,string> originalIdToName;
unordered_map<int,double> originalOutgoingConnectionCost;
vector<pair<int,int>> originalEdges;
unordered_map<int,double> originalCpuLatency, originalFpgaLatency, originalSize;
unordered_map<int,int> originalColorClass;


void from_json(const json& j, Node &n) {
    j.at("id").get_to(n.id);
    j.at("supportedOnFpga").get_to(n.supportedOnFpga);

    if (j.count("name")) {
        nameOfId[n.id] = j.at("name");
    }

    j.at("fpgaLatency").get_to(n.fpgaLatency);
    originalFpgaLatency[n.id] = n.fpgaLatency;

    j.at("cpuLatency").get_to(n.cpuLatency);
    originalCpuLatency[n.id] = n.cpuLatency;

    if (j.count("size")) {
        j.at("size").get_to(n.size);
    } else {
        n.size = 0.0;
    }
    originalSize[n.id] = n.size;

    j.at("isBackwardNode").get_to(n.isBackwardNode);

    if (j.count("colorClass")) {
        j.at("colorClass").get_to(n.colorClass);
    } else {
        static int freshNumber = numeric_limits<int>::max() - 12345678;
        // some number that will never come up as a color class in the input
        --freshNumber;
        n.colorClass = freshNumber;
        dbg << "node " << n.id << " has no color class in the input, assigning " << freshNumber << endl;
    }
    originalColorClass[n.id] = n.colorClass;

    if (j.count("layerId")) {
        j.at("layerId").get_to(n.layerId);
    } else {
        static int freshNumber = numeric_limits<int>::max() - 50000000;
        // some number that will never come up as a layerId in the input
        ++freshNumber;
        n.layerId = freshNumber;
        if (CONTRACT_LAYERS) {
            dbg << "node " << n.id << " has no layerId in the input, assigning " << freshNumber << endl;
        }
    }

    if (j.count("containedSubnodes")) {
        fail("the input file should not have containedSubnodes");
    } else {
        n.containedSubnodes = {n.id};
    }

    if (j.count("name")) {
        originalIdToName[n.id] = j.at("name");
    }
}


vector<string> idsToNames (const vector<int> &v) {
    vector<string> res;
    for (int id : v) {
        res.emplace_back(originalIdToName[id]);
    }
    return res;
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
    bool isArtificial;
};


void from_json(const json&j, Edge &n) {
    j.at("sourceId").get_to(n.sourceId);
    j.at("destId").get_to(n.destId);
    double cost;
    j.at("cost").get_to(cost);
    if (originalOutgoingConnectionCost.count(n.sourceId)) {
        if (abs(originalOutgoingConnectionCost[n.sourceId] - cost) > 1e-6) {
            fail("node " + to_string(n.sourceId) + " has two different outgoing edge costs");
        }
    } else {
        originalOutgoingConnectionCost[n.sourceId] = cost;
    }
    if (j.count("isArtificial")) {
        // probably never going to use this in the input
        j.at("isArtificial").get_to(n.isArtificial);
    } else {
        n.isArtificial = false;
    }
    originalEdges.emplace_back(n.sourceId, n.destId);
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

    bool isDAG (bool lookOnlyAtForward = false) const;
    bool hasBackwardToForwardEdge() const;
    bool hasBackwardOnlyColorClass () const;
    InputInstance contractColorClasses () const;
    void createArtificialForwardNodes();
    void fuseStronglyConnectedComponentsInForward();
    void reportNodesAndEdges() const;
    void performTrivialNodeOptimizations();
    void fuseLayers();
    void fuseDccs();
    void linearize();
    void runScotch();
    void runLocalSearch(int numberOfRepeats);
    double _localSearchObjective (const unordered_map<int,int> &originalNodeToColorClass, const unordered_map<int,int> &groupOfCc);
    void fillDccIdsWithBiconnectedComponent();
};


void from_json(const json&j, InputInstance &n) {
    static int invocationCount = 0;
    ++invocationCount;
    if (invocationCount > 1) {
        fail("do not read more than one InputInstance from JSON, we do not support this currently");
    }
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


void InputInstance::reportNodesAndEdges() const {
    dbg << endl;
    int countForwardNodes = 0, countBackwardNodes = 0, countForwardSinkNodes = 0;
    unordered_set<int> nodesWithForwardDescendant, forwardNodes;
    for (const Node &n : nodes) {
        if (n.isBackwardNode) {
            ++countBackwardNodes;
        } else {
            ++countForwardNodes;
            forwardNodes.insert(n.id);
        }
    }
    for (const Edge &e : edges) {
        if (forwardNodes.count(e.destId)) {
            nodesWithForwardDescendant.insert(e.sourceId);
        }
    }
    for (const Node &n : nodes) {
        if (!n.isBackwardNode && !nodesWithForwardDescendant.count(n.id)) {
            ++countForwardSinkNodes;
        }
    }
    dbg << nodes.size() << " nodes (" << countForwardNodes << " forward (" << countForwardSinkNodes << " sinks), " << countBackwardNodes << " backward)" << endl;
}


void InputInstance::performTrivialNodeOptimizations() {
    unordered_set<int> forwardNodes;
    unordered_map<int,int> colorClass;
    for (const Node &n : nodes) {
        if (!n.isBackwardNode) {
            forwardNodes.insert(n.id);
        }
        colorClass[n.id] = n.colorClass;
    }
    unordered_map<int,vector<int>> forwardNeighbors, backwardNeighbors;
    for (const Edge &e : edges) {
        if (forwardNodes.count(e.sourceId)) {
            forwardNeighbors[e.destId].push_back(e.sourceId);
        } else {
            backwardNeighbors[e.destId].push_back(e.sourceId);
        }
        if (forwardNodes.count(e.destId)) {
            forwardNeighbors[e.sourceId].push_back(e.destId);
        } else {
            backwardNeighbors[e.sourceId].push_back(e.destId);
        }
    }
    unordered_map<int,double> colorClassToSumOfBackwardLatencies;
    for (const Node &n : nodes) {
        if (n.isBackwardNode) {
            colorClassToSumOfBackwardLatencies[n.colorClass] += n.cpuLatency + n.fpgaLatency;
        }
    }
    for (Node &n : nodes) {
        if (!n.isBackwardNode && n.cpuLatency < 1e-6 && n.fpgaLatency < 1e-6) {
            if (backwardNeighbors[n.id].empty() && forwardNeighbors[n.id].size() == 1) {
                if (colorClassToSumOfBackwardLatencies[n.colorClass] < 1e-6) {
                    // ok
                    int m_id = forwardNeighbors[n.id][0];
                    int m_colorClass = colorClass[m_id];
                    int n_colorClass = n.colorClass;
                    dbg << "will merge node " << n.id << " into " << m_id << endl;
                    for (Node &n2 : nodes) {
                        if (n2.colorClass == n_colorClass) {
                            n2.colorClass = m_colorClass;
                        }
                    }
                    *this = this->contractColorClasses();
                    performTrivialNodeOptimizations();
                    return;
                }
            }
        }
    }
}


bool InputInstance::isDAG (bool lookOnlyAtForward) const {
    unordered_map<int,int> indegree;
    unordered_set<int> idsOfForwardVertices;
    for (const Node &n : nodes) {
        if (!n.isBackwardNode) {
            idsOfForwardVertices.insert(n.id);
        }
    }
    for (const Edge &e : edges) {
        if (lookOnlyAtForward && !idsOfForwardVertices.count(e.destId)) {
            continue;
        }
        ++indegree[e.destId];
    }
    vector<int> deg0vertices;
    for (const Node &n : nodes) {
        // also does this with backward nodes, each of which should have indegree 0 (in the forward-only graph)
        if (indegree[n.id] == 0) {
            deg0vertices.push_back(n.id);
        }
    }
    int processed_vertices = 0;
    while (!deg0vertices.empty()) {
        int v = deg0vertices.back();
        deg0vertices.pop_back();
        ++processed_vertices;
        // inefficient but fine:
        for (const Edge &e : edges) {
            if (e.sourceId == v) {
                if (lookOnlyAtForward && !idsOfForwardVertices.count(e.destId)) {
                    continue;
                }
                --indegree[e.destId];
                if (indegree[e.destId] == 0) {
                    deg0vertices.push_back(e.destId);
                }
            }
        }
    }
    return processed_vertices == nodes.size();
}


bool InputInstance::hasBackwardToForwardEdge() const {
    // it should never have
    map<int,bool> isBackwardNode;
    for (const Node &n : nodes) {
        isBackwardNode[n.id] = n.isBackwardNode;
    }
    for (const Edge &e : edges) {
        if (isBackwardNode[e.sourceId] && !isBackwardNode[e.destId]) {
            return true;
        }
    }
    return false;
}


bool InputInstance::hasBackwardOnlyColorClass () const {
    // it should never have
    unordered_set<int> colorClassesWithForwardNodes;
    for (const Node &n : nodes) {
        if (!n.isBackwardNode) {
            colorClassesWithForwardNodes.insert(n.colorClass);
        }
    }
    for (const Node &n : nodes) {
        if (n.isBackwardNode) {
            if (!colorClassesWithForwardNodes.count(n.colorClass)) {
                //warn("backward node " + to_string(n.id) + ", in color class " + to_string(n.colorClass) + ", containing [" + to_string(n.containedSubnodes) + "], has no corresponding forward node");
                return true;
            }
        }
    }
    return false;
}


// contract separately the forward nodes and the backward nodes in every color class
InputInstance InputInstance::contractColorClasses () const {
    set<int> colorClasses;
    for (const Node &n : nodes) {
        colorClasses.insert(n.colorClass);
    }

    int maxColorClass = *colorClasses.rbegin();
    const int maxForwardColorClass = maxColorClass;
    map<int,int> backwardColorClass, forwardColorClass;
    for (int colorClass : colorClasses) {
        ++maxColorClass;
        backwardColorClass[colorClass] = maxColorClass;
        forwardColorClass[maxColorClass] = colorClass;
    }

    // now for each color class cc we have two virtual color classes (VCCs):
    // cc (fwd) and backwardColorClass[cc] (bkwd)
    // (the latter might not contain any nodes)
    unordered_map<int,int> vccOfNodeWithId;
    unordered_map<int,double> totalCpuLatency, totalFpgaLatency, totalSize; // of nodes of some VCC
    unordered_map<int,bool> allSupportedOnFpga;
    unordered_map<int,unordered_set<int>> nodesOfVcc;
    for (const Node &n : nodes) {
        vccOfNodeWithId[n.id] = n.isBackwardNode ? backwardColorClass[n.colorClass] : n.colorClass;
        nodesOfVcc[vccOfNodeWithId[n.id]].insert(n.id);
        totalCpuLatency[vccOfNodeWithId[n.id]] += n.cpuLatency;
        totalFpgaLatency[vccOfNodeWithId[n.id]] += n.fpgaLatency;
        totalSize[vccOfNodeWithId[n.id]] += n.size;
        if (allSupportedOnFpga.count(vccOfNodeWithId[n.id])) {
            allSupportedOnFpga[vccOfNodeWithId[n.id]] = allSupportedOnFpga[vccOfNodeWithId[n.id]] && n.supportedOnFpga;
        } else {
            allSupportedOnFpga[vccOfNodeWithId[n.id]] = n.supportedOnFpga;
        }
    }
    set<pair<int,int>> trueEdges, artificialEdges; // edges between some pair of VCCs
    for (const Edge &e : edges) {
        if (vccOfNodeWithId[e.sourceId] == vccOfNodeWithId[e.destId]) {
            continue; // we do not create self-loops in the contracted graph
        }
        (e.isArtificial ? artificialEdges : trueEdges).insert({vccOfNodeWithId[e.sourceId],vccOfNodeWithId[e.destId]});
    }

    // now build the output instance
    InputInstance ii;
    ii.maxSizePerFpga = maxSizePerFpga;
    ii.maxFpgas = maxFpgas;
    ii.maxCpus = maxCpus;
    for (auto it : nodesOfVcc) {
        int vcc = it.first;
        Node n;
        n.id = vcc;
        n.supportedOnFpga = allSupportedOnFpga[vcc];
        n.cpuLatency = totalCpuLatency[vcc];
        n.fpgaLatency = totalFpgaLatency[vcc];
        n.size = totalSize[vcc];
        n.isBackwardNode = (vcc > maxForwardColorClass);
        n.colorClass = n.isBackwardNode ? forwardColorClass[vcc] : vcc;
        // n.containedSubnodes = union of m.containedNodes over m in nodesOfVcc[vcc]
        // (a bit inefficient)
        for (const Node &m : nodes) {
            if (nodesOfVcc[vcc].count(m.id)) {
                append(n.containedSubnodes, m.containedSubnodes);
            }
        }
        ii.nodes.push_back(n);
    }
    for (auto it : artificialEdges) {
        Edge e;
        e.sourceId = it.first;
        e.destId = it.second;
        // if e.sourceId == forwardColorClass[e.destId], then we might as well have skipped this edge
        // because it cannot be cut anyway
        e.isArtificial = true;
        ii.edges.push_back(e);
    }
    for (auto it : trueEdges) {
        Edge e;
        e.sourceId = it.first;
        e.destId = it.second;
        // if e.sourceId == forwardColorClass[e.destId], then we might as well have skipped this edge
        // because it cannot be cut anyway
        e.isArtificial = false;
        ii.edges.push_back(e);
    }

    return ii;
}


constexpr bool ORPHAN_DEBUG = true;


void InputInstance::createArtificialForwardNodes() {
    // should be run on contracted graph (i.e. 1 fw/bw node per color class)
    unordered_set<int> currentNodeIds;
    for (const Node &n : nodes) {
        currentNodeIds.insert(n.id);
    }

    unordered_map<int,int> colorClassToForwardNode;
    unordered_map<int,int> backwardNodeToColorClass;
    for (const Node &n : nodes) {
        if (!n.isBackwardNode) {
            assert(!colorClassToForwardNode.count(n.colorClass));
            colorClassToForwardNode[n.colorClass] = n.id;
        } else {
            backwardNodeToColorClass[n.id] = n.colorClass;
        }
    }
    unordered_set<int> orphanedBackwardNodes;
    for (const Node &n : nodes) {
        if (n.isBackwardNode) {
            if (!colorClassToForwardNode.count(n.colorClass)) {
                // n has no forward mate
                orphanedBackwardNodes.insert(n.id);
                if (ORPHAN_DEBUG) {
                    dbg << "node " << n.id << " of color class " << n.colorClass <<
                            ", containing original nodes [" << n.containedSubnodes <<
                            "] (names: [" << idsToNames(n.containedSubnodes) <<
                            "]), is orphaned-backward" << endl;
                }
            }
        }
    }
    // make artificial nodes
    for (int id : orphanedBackwardNodes) {
        int colorClass = backwardNodeToColorClass[id];
        Node n;
        n.colorClass = colorClass;
        n.containedSubnodes = {};
        n.cpuLatency = 0.0;
        n.fpgaLatency = 0.0;
        n.id = 100000;
        while (currentNodeIds.count(n.id)) ++n.id; // find some free ID
        currentNodeIds.insert(n.id);
        n.isBackwardNode = false;
        n.size = 0.0;
        n.supportedOnFpga = true;
        nodes.push_back(n);
        assert(!colorClassToForwardNode.count(colorClass));
        colorClassToForwardNode[colorClass] = n.id;
        if (ORPHAN_DEBUG) {
            dbg << "added artificial node " << n.id << " for orphaned-backward node with id = " << id << " and color-class " << n.colorClass << endl;
        }
    }
    // now tie the new artificial nodes with edges like a mirror image
    const int countEdges = edges.size();
    for (int i = 0; i < countEdges; ++i) {
        int ub = edges[i].sourceId, vb = edges[i].destId;
        if (backwardNodeToColorClass.count(ub) && backwardNodeToColorClass.count(vb)) {
            // backward->backward node
            if (orphanedBackwardNodes.count(ub) || orphanedBackwardNodes.count(vb)) {
                // either of them is orphaned
                int uf = colorClassToForwardNode[backwardNodeToColorClass[ub]],
                    vf = colorClassToForwardNode[backwardNodeToColorClass[vb]];
                Edge e;
                e.sourceId = vf;
                e.destId = uf;
                e.isArtificial = true;
                edges.push_back(e);
                if (ORPHAN_DEBUG) {
                    dbg << "added artificial edge " << vf << "->" << uf << endl;
                }
            }
        }
    }
}


void InputInstance::fuseStronglyConnectedComponentsInForward() {
    // everything is done only in the forward graph, except the fusing at the end
    set<pair<int,int>> reachable; // reachable[{u,v}] = there is a path from u to v
    // initialize with edges
    for (const Edge &e : edges) {
        reachable.insert({e.sourceId, e.destId});
    }
    dbg << endl;
    dbg << "fusing strongly connected components in the forward graph, if any..." << endl;
    dbg << endl;
    // run Floyd-Warshall (O(n^3 log n) runtime... might need to replace with something faster)
    for (const Node &k : nodes) if (!k.isBackwardNode) {
        for (const Node &i : nodes) if (!i.isBackwardNode) {
            for (const Node &j : nodes) if (!j.isBackwardNode) {
                if (reachable.count({i.id, k.id}) && reachable.count({k.id, j.id})) {
                    reachable.insert({i.id, j.id});
                }
            }
        }
    }
    // fuse color classes via Union-Find
    unordered_map<int,int> ufTab;
    for (const Node &n : nodes) {
        ufTab[n.colorClass] = n.colorClass;
    }
    for (const Node &n : nodes) if (!n.isBackwardNode) {
        for (const Node &m : nodes) if (!m.isBackwardNode) {
            if (reachable.count({n.id, m.id}) && reachable.count({m.id, n.id})) {
                //dbg << "bidirectionally reachable CCs: " << n.colorClass << "," << m.colorClass << endl;
                ufUnion(ufTab, n.colorClass, m.colorClass);
            }
        }
    }
    // now change the color classes (in the entire graph, also the backward part)
    for (Node &n : nodes) {
        n.colorClass = ufFind(ufTab, n.colorClass);
    }

    for (auto it : ufTab) {
        if (it.first != it.second) {
            dbg << "old cc: " << it.first << ", new cc: " << it.second << endl;
        }
    }
}


void InputInstance::fuseLayers() {
    // use Union-Find
    unordered_map<int,int> ufTab;
    for (const Node &n : nodes) {
        ufTab[n.colorClass] = n.colorClass;
    }
    unordered_map<int,vector<int>> layerIdToNodeColorClass;
    for (const Node &n : nodes) {
        layerIdToNodeColorClass[n.layerId].push_back(n.colorClass);
    }
    for (const auto &it: layerIdToNodeColorClass) {
        for (int i = 1; i < it.second.size(); ++i) {
            ufUnion(ufTab, it.second[i-1], it.second[i]);
        }
    }
    // now change the color classes
    for (Node &n : nodes) {
        n.colorClass = ufFind(ufTab, n.colorClass);
    }
    for (auto it : ufTab) {
        if (it.first != it.second) {
            dbg << "fusing a layer together: old cc: " << it.first << ", new cc: " << it.second << endl;
        }
    }
}

void InputInstance::fuseDccs() {
    // use Union-Find
    unordered_map<int,int> ufTab;
    for (const Node &n : nodes) {
        ufTab[n.colorClass] = n.colorClass;
    }
    unordered_map<int,vector<int>> dccToNodeColorClass;
    for (const Node &n : nodes) {
        dccToNodeColorClass[n.dcc].push_back(n.colorClass);
    }
    for (const auto &it: dccToNodeColorClass) {
        for (int i = 1; i < it.second.size(); ++i) {
            ufUnion(ufTab, it.second[i-1], it.second[i]);
        }
    }
    // now change the color classes
    for (Node &n : nodes) {
        n.colorClass = ufFind(ufTab, n.colorClass);
    }
    for (auto it : ufTab) {
        if (it.first != it.second) {
            dbg << "fusing a dcc together: old cc: " << it.first << ", new cc: " << it.second << endl;
        }
    }
}


void InputInstance::linearize () {
    // operates on the forward part of the graph
    unordered_map<int,int> indegree;
    unordered_set<int> idsOfForwardVertices;
    for (const Node &n : nodes) {
        if (!n.isBackwardNode) {
            idsOfForwardVertices.insert(n.id);
        }
    }
    set<pair<int,int>> alreadyPresentEdges;
    for (const Edge &e : edges) {
        if (!idsOfForwardVertices.count(e.destId)) {
            continue;
        }
        ++indegree[e.destId];
        alreadyPresentEdges.emplace(e.sourceId, e.destId);
    }
    vector<int> deg0vertices;
    for (const Node &n : nodes) {
        if (!n.isBackwardNode && indegree[n.id] == 0) {
            deg0vertices.emplace_back(n.id);
        }
    }
    vector<pair<int,int>> edgesToAdd;
    bool firstNode = true;
    int previousNode;
    while (!deg0vertices.empty()) {
        int v = deg0vertices.back(); // DFS-ish traversal
        deg0vertices.pop_back();

        // visiting v
        if (!firstNode) {
            if (!alreadyPresentEdges.count({previousNode, v})) {
                edgesToAdd.emplace_back(previousNode, v);
            }
        }
        firstNode = false;
        previousNode = v;

        // inefficient but whatever:
        for (const Edge &e : edges) {
            if (e.sourceId == v) {
                if (!idsOfForwardVertices.count(e.destId)) {
                    continue;
                }
                --indegree[e.destId];
                if (indegree[e.destId] == 0) {
                    deg0vertices.emplace_back(e.destId);
                }
            }
        }
    }
    for (const auto &it : edgesToAdd) {
        Edge e;
        e.sourceId = it.first;
        e.destId = it.second;
        e.isArtificial = true;
        edges.push_back(e);
    }
}



struct ResultMachine {
    double load;
    vector<int> nodes;
};


struct Result {
    double maxLoad;
    vector<ResultMachine> fpgas;
    vector<ResultMachine> cpus;
};


void to_json(json &j, const ResultMachine &rs) {
    j = json{{"load", rs.load}, {"nodes", rs.nodes}};
}


void to_json(json &j, const Result &r) {
    j = json{{"maxLoad", r.maxLoad}, {"fpgas", r.fpgas}, {"cpus", r.cpus}};
}


// for human experts and the like:
void from_json(const json &j, ResultMachine &rs) {
    // ignore load
    j.at("nodes").get_to(rs.nodes);
}


void from_json(const json &j, Result &r) {
    // ignore load
    j.at("cpus").get_to(r.cpus);
    j.at("fpgas").get_to(r.fpgas);
}



// result in terms of original nodes:
// mapping original_node_id -> deviceId (cpu = 0, rest = 1..maxFpgas)
typedef unordered_map<int,int> ResultOrg;


double computeLoadForResultOrg (const InputInstance &ii, const ResultOrg &ro) {
    unordered_map<int,double> groupLatency, groupSize;
    for (const pair<int,double>& p : originalFpgaLatency) {
        int originalId = p.first;
        if (!ro.count(originalId)) {
            warn("node with original id " + to_string(originalId) + " not present in result");
        }
    }
    for (const pair<int,double>& p : originalFpgaLatency) {
        int originalId = p.first;
        if (!ro.count(originalId)) {
            fail("node with original id " + to_string(originalId) + " not present in result");
        }
        const int group = ro.at(originalId);
        if (group == 0) {
            // cpu
            groupLatency[0] += originalCpuLatency[originalId];
        } else {
            // fpga
            groupLatency[group] += originalFpgaLatency[originalId];
        }
        groupSize[group] += originalSize[originalId];
    }
    unordered_map<int,unordered_set<int>> groupsWithEdgeFrom;
    // originalId -> {groups that have an edge (originalId, v) for some v in the group}
    for (const pair<int,int> &ed : originalEdges) {
        if (ro.at(ed.first) == ro.at(ed.second)) {
            // ignore self-loops
            continue;
        }
        groupsWithEdgeFrom[ed.first].insert(ro.at(ed.second));
    }

    unordered_map<int,double> groupCommunicationCost;
    for (auto &it : groupsWithEdgeFrom) {
        it.second.erase(ro.at(it.first)); // again ignore self-loops, just in case
        if (it.second.empty()) {
            continue;
        }
        // outgoing costs
        groupCommunicationCost[ro.at(it.first)] += originalOutgoingConnectionCost[it.first];
        // incoming costs
        for (int incomingGroup : it.second) {
            groupCommunicationCost[incomingGroup] += originalOutgoingConnectionCost[it.first];
        }
    }

    double maxLoad = 0.0, maxSize = 0.0;
    for (auto &it : groupLatency) {
        int group = it.first;
        double load;
        if (group == 0) {
            // cpu does not pay for communication
            load = groupLatency[group];
        } else {
            load = groupLatency[group] + groupCommunicationCost[group];
        }
        maxLoad = max(maxLoad, load);
        if (group != 0) {
            maxSize = max(maxSize, groupSize[group]);
        }
    }

    // check feasibility of solution:
    // number of devices used
    if (groupLatency.size() > ii.maxFpgas + groupLatency.count(0)) {
        fail("given result is using too many devices");
    }

    // check sizes (warning only)
    //dbg << "maxSizePerFpga = " << ii.maxSizePerFpga << endl;
    //dbg << "max size in given solution: " << maxSize << endl;
    if (maxSize > ii.maxSizePerFpga) {
        warn("given result uses too much memory on some device");
    }

    // check CCs
    unordered_map<int,vector<int>> nodesOfColorClass;
    for (const auto &it : originalColorClass) {
        nodesOfColorClass[it.second].push_back(it.first);
    }
    for (const auto &it : nodesOfColorClass) {
        for (int i = 1; i < it.second.size(); ++i) {
            if (ro.at(it.second.at(i)) != ro.at(it.second.at(0))) {
                fail("given result separates colocated nodes");
            }
        }
    }

    return maxLoad;
}


ResultOrg resultToResultOrg (const Result &r) {
    ResultOrg ro;
    for (const ResultMachine &rm : r.cpus) {
        for (int originalId : rm.nodes) {
            ro[originalId] = 0;
        }
    }
    int fpgaId = 1;
    for (const ResultMachine &rm : r.fpgas) {
        for (int originalId : rm.nodes) {
            ro[originalId] = fpgaId;
        }
        ++fpgaId;
    }
    return ro;
}


Result resultOrgToResult (const ResultOrg &ro) {
    Result r;
    r.maxLoad = -1;
    r.cpus.emplace_back();
    r.cpus[0].load = -1;
    for (const pair<int,int> &p : ro) {
        if (p.second == 0) {
            r.cpus[0].nodes.push_back(p.first);
        } else {
            while (r.fpgas.size() < p.second) {
                r.fpgas.emplace_back();
                r.fpgas.back().load = -1;
            }
            r.fpgas[p.second - 1].nodes.push_back(p.first);
        }
    }
    return r;
}


double computeLoadForResult (const InputInstance &ii, const Result &r) {
    return computeLoadForResultOrg(ii, resultToResultOrg(r));
}


// takes a human-expert split which has all forward nodes assigned
// and assigns backward nodes to the same machine that contains the forward nodes
// that they are colocated with
void augmentHumanExpertToTrainingWithColocations (const InputInstance &ii, ResultOrg &hero) {
    unordered_map<int,vector<int>> colorClassToOriginalNodes;
    for (const auto &p : originalColorClass) {
        colorClassToOriginalNodes[p.second].push_back(p.first);
    }

    for (const auto &p : originalFpgaLatency) {
        int originalNode = p.first;
        if (!hero.count(originalNode)) {
            //dbg << p.first << endl;
            if (originalColorClass.count(originalNode)) {
                int occ = originalColorClass[originalNode];
                unordered_set<int> assignments;
                for (int otherOriginalId : colorClassToOriginalNodes[occ]) {
                    if (hero.count(otherOriginalId)) {
                        assignments.insert(hero[otherOriginalId]);
                    }
                }
                //dbg << assignments.size () << endl;
                if (assignments.size() == 0) {
                    fail("no assignment could be inferred for node with original id " + to_string(originalNode));
                }
                if (assignments.size() > 1) {
                    fail("two or more assignments could be inferred for node with original id " + to_string(originalNode));
                }
                int machineId = *assignments.begin();
                hero[originalNode] = machineId;
                //dbg << hero.count(originalNode) << endl;
            } else {
                fail("no color class for original node?");
            }
        }
    }   
}


double InputInstance::_localSearchObjective (
    const unordered_map<int,int> &originalNodeToColorClass, const unordered_map<int,int> &groupOfCc) {
    ResultOrg ro;
    for (const auto & it : originalNodeToColorClass) {
        ro[it.first] = groupOfCc.at(it.second);
    }
    return computeLoadForResultOrg(*this, ro);
}


void InputInstance::runLocalSearch (int numberOfRepeats) {    
    unordered_map<int,vector<int>> nodesOfColorClass;
    unordered_map<int,int> originalNodeToColorClass; // original_node_id -> its colorClass
    for (Node &n : nodes) {
        for (int originalNode : n.containedSubnodes) {
            originalNodeToColorClass[originalNode] = n.colorClass;
            nodesOfColorClass[n.colorClass].push_back(originalNode);
        }
    }

    // we are ignoring size constraints here!

    double bestValue = 1e30;
    //unordered_map<int,int> globalBestGroupOfCc;
    for (int restart = 1; restart <= numberOfRepeats; ++restart) {
        dbg << "starting run of local search" << endl;
        unordered_map<int,int> groupOfCc; // random group assigned to color class
        for (const auto &it : nodesOfColorClass) {
            int cc = it.first;
            groupOfCc[cc] = 1 + (rand() % maxFpgas);           
        }

        // compute current objective
        double curObj = _localSearchObjective(originalNodeToColorClass, groupOfCc);

        while (true) {
            // look at every possible change
            double bestNewObj = 1e30;
            unordered_map<int,int> bestNewGroupOfCc;

            for (const auto &it : nodesOfColorClass) {
                int cc = it.first;
                for (int newGroup = 1; newGroup <= maxFpgas; ++newGroup) {
                    if (newGroup != groupOfCc[cc]) {
                        unordered_map<int,int> newGroupOfCc = groupOfCc;
                        newGroupOfCc[cc] = newGroup;
                        double newObj = _localSearchObjective(originalNodeToColorClass, newGroupOfCc);
                        if (newObj < bestNewObj) {
                            bestNewObj = newObj;
                            bestNewGroupOfCc = newGroupOfCc;
                        }
                    }
                }
            }

            if (bestNewObj > curObj - 0.00001) {
                break; // give up, approximate local optimum
            } else {
                dbg << "making change..." << endl;
                curObj = bestNewObj;
                dbg << "curObj = " << curObj << endl;
                groupOfCc = bestNewGroupOfCc;
            }
        }

        if (bestValue > curObj) {
            bestValue = curObj;
            //globalBestGroupOfCc = groupOfCc;
        }
    }

    dbg << "local search done. maxLoad = " << bestValue << endl;
}


void InputInstance::runScotch () {
    unordered_map<int,vector<int>> nodesOfColorClass;
    unordered_map<int,int> originalNodeToColorClass; // original_node_id -> its colorClass
    for (Node &n : nodes) {
        for (int originalNode : n.containedSubnodes) {
            originalNodeToColorClass[originalNode] = n.colorClass;
            nodesOfColorClass[n.colorClass].push_back(originalNode);
        }
    }

    map<pair<int,int>, set<int>> originalNodesWithEdgesBetween; // (cc1,cc2) -> {original_node_ids in cc1 that have an outgoing edge to a node in cc2}
    for (const pair<int,int> &p : originalEdges) {
        assert(originalNodeToColorClass.count(p.first));
        assert(originalNodeToColorClass.count(p.second));
        int cc1 = originalNodeToColorClass[p.first];
        int cc2 = originalNodeToColorClass[p.second];
        originalNodesWithEdgesBetween[{cc1,cc2}].insert(p.first);
    }

    map<int, map<int, double>> edgeWeight;// cc1 -> cc2 -> sum of weights of relevant edges (but at most one per outgoing node)
    // should be symmetric: edgeWeight[cc1][cc2] == edgeWeight[cc2][cc1]
    for (const auto &it : originalNodesWithEdgesBetween) {
        int cc1 = it.first.first, cc2 = it.first.second;
        if (cc1 == cc2) continue;
        double sumOfCosts = 0.0;
        for (int originalId : it.second) {
            sumOfCosts += originalOutgoingConnectionCost[originalId];
        }
        edgeWeight[cc1][cc2] += sumOfCosts;
        edgeWeight[cc2][cc1] += sumOfCosts;
    }

    unordered_map<int,double> totalFpgaLatency; // cc -> sum of fpgaLatencies of relevant nodes
    unordered_map<int,double> totalSize; // same
    for (Node &n : nodes) {
        totalFpgaLatency[n.colorClass] += n.fpgaLatency;
        totalSize[n.colorClass] += n.size;
        if (!n.supportedOnFpga) {
            fail("node unsupported on FPGA, not good for Scotch");
        }
    }

    const int nodeCount = totalFpgaLatency.size();
    int doubleEdgeCount = 0;
    for (const auto &it : edgeWeight) {
        doubleEdgeCount += it.second.size();
    }

    ofstream of("scotch.grf");
    of << "0\n";
    of << nodeCount << " " << doubleEdgeCount << "\n";
    of << "0 111\n";
    for (const auto &it : totalFpgaLatency) {
        int rounded = 1e6 * it.second;
        of << it.first << " " << rounded << " " << edgeWeight[it.first].size();
        for (const auto &jt : edgeWeight[it.first]) {
            int rounded2 = 1e6 * jt.second;
            of << " " << rounded2 << " " << jt.first;
        }
        of << "\n";
    }
    of.close();

    const string scotchExecutable = "gpart"; // edit this line if `gpart` is not in system PATH
    dbg << "executing Scotch..." << endl;
    int returnCode = system((scotchExecutable + " " + to_string(maxFpgas) + " < scotch.grf > scotch.map").c_str());
    if (returnCode != 0) {
        fail("return code not zero. perhaps scotch (command `gpart`) is not under system PATH?");
    }
    dbg << "Scotch is done" << endl;
    ifstream res("scotch.map");
    int junk;
    res >> junk;
    assert(junk == nodeCount);
    ResultOrg ro; // map: original_id -> deviceId (0=cpu, which we don't use)
    for (int i = 0; i < nodeCount; ++i) {
        int cc, groupId;
        res >> cc >> groupId;
        ++groupId; // we number fpgas from 1, Scotch numbers groups from 0
        for (int nodeId : nodesOfColorClass[cc]) {
            ro[nodeId] = groupId;
        }
    }

    // okay. now compute its objective score
    double maxLoad = computeLoadForResultOrg(*this, ro);
    cout << "computed maxLoad for Scotch result: " << maxLoad << endl;

    // and print output
    cout << json(resultOrgToResult(ro)).dump(4);
}


ResultOrg allOnOneFpga () {
    ResultOrg ro;
    double totalFpgaLatency = 0.0;
    for (const auto &it : originalFpgaLatency) {
        ro[it.first] = 1;
        totalFpgaLatency += it.second;
    }
    dbg << "simple-formula FPGA load: " << totalFpgaLatency << endl;
    return ro;
}



struct Graph {
    InputInstance ii;

    // renumbering the vertices from inputInstance to 0,1,2...: first forward, then backward
    int countForwardNodes;
    int countBackwardNodes;
    unordered_map<int,int> newNumber;
    vector<int> oldNumber;
    vector<double> cpuLatency, fpgaLatency, size; // xxx[i] = ii.nodes[oldNumber[i]].xxx
    vector<bool> supportedOnFpga; // same

    vector<int> backwardSibling;
    // for 0<=i<countForwardNodes, backwardSibling[i] is the new-number of the backward-node
    // with the same color-class as i. if none, then it's -1

    // THESE CORRESPOND TO ALL EDGES, ALSO THE ONES ADDED ARTIFICIALLY:
    vector<vector<int>> incomingEdges; // {source}

    // THESE CORRESPOND TO ORIGINAL EDGES (mapped to the contracted graph):
    vector<vector<pair<int,vector<int>>>> outgoing, incoming;
    // outgoing[u] contains pairs {v, s} where s = original subnodes of u that have a true edge to an original subnode of v
    // incoming[v] contains pairs {u, s} where s = original subnodes of u that have a true edge to an original subnode of v

    // ideals, represented as indicator vectors
    unordered_map<vector<bool>,int> idealToId; // maps ideal to its ID
    vector<vector<bool>> ideals; // maps ID to ideal

    // pairs of ideals (that induce contiguous sets)
    vector<vector<int>> immediateSubIdeals;
    // immediateSubIdeals[id] = IDs of ideals that are immediate subsets of the ideal with ID id
    struct IdealPairCachedScores {
        double totalCommunicationCost; // of edges cut
        double totalCpuLatency;
        double totalFpgaLatency; // is INFTY if can't put this on FPGA
    };
    vector<vector<pair<int,IdealPairCachedScores>>> subIdeals;
    // subideals[id] = pairs(ID,cached scores) of ideals that are subsets of the ideal with ID id
    long long numberOfIdealPairs;

    Graph (const InputInstance &_ii);
    void generateIdeals ();
    void growIdeal (const vector<bool> &ideal, int myId);
    void prepareSubIdeals ();
    vector<bool> getExtendedContiguousSet (int id, int subId) const;
    IdealPairCachedScores prepareCachedScores (int id, int subId) const;
    Result runDP();
};


Graph::Graph (const InputInstance &_ii) :
    ii(_ii) {
    // renumber the vertices 0,1,2...; first the forward nodes
    countForwardNodes = 0;
    countBackwardNodes = 0;
    for (bool nowRenumberingForwardNodes : {true, false}) {
        for (const Node &n : ii.nodes) {
            if (n.isBackwardNode != nowRenumberingForwardNodes) {
                newNumber[n.id] = oldNumber.size();
                oldNumber.push_back(n.id);
                cpuLatency.push_back(n.cpuLatency);
                fpgaLatency.push_back(n.fpgaLatency);
                size.push_back(n.size);
                supportedOnFpga.push_back(n.supportedOnFpga);
                if (!n.isBackwardNode) {
                    ++countForwardNodes;
                } else {
                    ++countBackwardNodes;
                }
            }
        }
    }
    assert(oldNumber.size() == countForwardNodes + countBackwardNodes);

    // fill backwardSibling
    unordered_map<int,int> backwardNodeOfColorClass;
    for (const Node &n : ii.nodes) {
        if (n.isBackwardNode) {
            backwardNodeOfColorClass[n.colorClass] = newNumber[n.id];
        }
    }
    backwardSibling.resize(countForwardNodes);
    for (const Node &n : ii.nodes) {
        if (!n.isBackwardNode) {
            backwardSibling[newNumber[n.id]] = backwardNodeOfColorClass.count(n.colorClass) ? backwardNodeOfColorClass[n.colorClass] : -1;
        }
    }

    // build incomingEdges
    incomingEdges.resize(oldNumber.size());
    for (const Edge &e : ii.edges) {
        int source = newNumber[e.sourceId], dest = newNumber[e.destId];
        if (source == dest) {
            fail("self-loops are not allowed");
        }
        incomingEdges[dest].emplace_back(source);
    }

    // build `outgoing` and `incoming`
    unordered_map<int,int> originalNodeToNewSupernode; // maps original node (old number) -> new number of the supernode that now contains it
    for (const Node &n : ii.nodes) {
        int newSupernode = newNumber[n.id];
        for (int originalNode : n.containedSubnodes) {
            originalNodeToNewSupernode[originalNode] = newSupernode;
        }
    }
    map<pair<int,int>, unordered_set<int>> newEdgeToOriginalOutgoingNodes;
    // maps (new source, new dest) -> set of original nodes that are now
    // in `new source` that have an original edge to an original node that is now in `new dest`
    for (const auto &it : originalEdges) {
        assert(originalNodeToNewSupernode.count(it.first));
        assert(originalNodeToNewSupernode.count(it.second));
        int newSource = originalNodeToNewSupernode[it.first];
        int newDest = originalNodeToNewSupernode[it.second];
        newEdgeToOriginalOutgoingNodes[{newSource,newDest}].insert(it.first);
    }
    // now really build `outgoing` and `incoming`
    outgoing.resize(oldNumber.size());
    incoming.resize(oldNumber.size());
    for (const auto &it : newEdgeToOriginalOutgoingNodes) {
        int u = it.first.first, v = it.first.second; // u -> v
        assert(0 <= u && u < oldNumber.size());
        assert(0 <= v && v < oldNumber.size());
        outgoing[u].emplace_back(v, vector<int>(it.second.begin(), it.second.end()));
        incoming[v].emplace_back(u, vector<int>(it.second.begin(), it.second.end()));
    }    

    generateIdeals();
    // immediateSubIdeals is prepared. now prepare subIdeals
    prepareSubIdeals();
}


void Graph::generateIdeals () {
    if (!ideals.empty()) {
        fail("generating ideals twice?");
    }

    // start with empty set
    vector<bool> emptySet(countForwardNodes, false);
    idealToId[emptySet] = 0;
    ideals.push_back(emptySet);
    immediateSubIdeals.emplace_back();
    growIdeal(emptySet, 0);

    dbg << ideals.size() << " ideals" << endl;
    if (ideals.size() > IDEALS_LIMIT) {
        fail("too many ideals (current limit set at " + to_string(IDEALS_LIMIT) + "); this isn't going to work...");
    }
    /*for (int i = 0; i < ideals.size(); ++i) {
        dbg << "ideals[" << i << "] = " << ideals[i] << endl;
    }*/
}


void Graph::growIdeal (const vector<bool> &ideal, int myId) {
    // myId == idealToId[ideal]
    // try to add every vertex
    for (int v = 0; v < countForwardNodes; ++v) {
        if (!ideal[v]) {
            // check if valid: do all v's predecessors belong to ideal?
            bool valid = true;
            for (int u : incomingEdges[v]) { // u->v
                if (!ideal[u]) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                vector<bool> newIdeal = ideal;
                newIdeal[v] = true;
                // check if newIdeal had already been generated
                if (!idealToId.count(newIdeal)) {
                    int newId = ideals.size();
                    idealToId[newIdeal] = newId;
                    ideals.push_back(newIdeal);
                    if (ideals.size() >= IDEALS_EXPLORATION_LIMIT) {
                        fail("already over " + to_string(IDEALS_EXPLORATION_LIMIT) + " ideals. this isn't going to work...");
                    }
                    immediateSubIdeals.emplace_back();
                    growIdeal(newIdeal, newId);
                }
                immediateSubIdeals[idealToId[newIdeal]].push_back(myId);
            }
        }
    }
}


void Graph::prepareSubIdeals () {
    // subideals = transitive closure of immediateSubIdeals, with also caching the scores

    numberOfIdealPairs = 0;
    subIdeals.resize(ideals.size());

    for (int id = 0; id < ideals.size(); ++id) {
        // we will generate subIdeals[id] using some BFS/DFS
        vector<int> queue = {id};
        unordered_set<int> enqueuedIdeals = {id};
        while (!queue.empty()) {
            int subId = queue.back();
            queue.pop_back();

            // now visiting subId
            subIdeals[id].emplace_back(subId, prepareCachedScores(id, subId));
            ++numberOfIdealPairs;

            // expand further from subId
            for (int subSubId : immediateSubIdeals[subId]) {
                if (enqueuedIdeals.insert(subSubId).second == true) {
                    // subSubId was not in enqueuedIdeals before
                    queue.push_back(subSubId);
                }
            }
        }
    }

    dbg << numberOfIdealPairs << " ideal pairs" << endl;
}


vector<bool> Graph::getExtendedContiguousSet (int id, int subId) const {
    // returns the contiguous set being:
    // the difference ideals[id] - ideals[subId],
    // plus also the corresponding backward nodes, if any (hence name 'extended')
    vector<bool> extendedSet(oldNumber.size(), false);
    for (int v = 0; v < countForwardNodes; ++v) {
        if (ideals[id][v] && !ideals[subId][v]) {
            extendedSet[v] = true;
            if (backwardSibling[v] != -1) {
                extendedSet[backwardSibling[v]] = true;
            }
        }
    }
    return extendedSet;
}


Graph::IdealPairCachedScores Graph::prepareCachedScores (int id, int subId) const {
    // prepare scores for the corresponding extended contiguous set
    vector<bool> extendedSet = getExtendedContiguousSet(id, subId);

    unordered_set<int> originalNodesWithOutgoingCommunication;

    IdealPairCachedScores s;
    s.totalCpuLatency = 0.0;
    s.totalFpgaLatency = 0.0;
    double totalSize = 0.0;
    bool allNodesSupportedOnFpga = true;
    for (int u = 0; u < oldNumber.size(); ++u) {
        if (extendedSet[u]) {
            s.totalCpuLatency += cpuLatency[u];
            s.totalFpgaLatency += fpgaLatency[u];
            if (!supportedOnFpga[u]) {
                allNodesSupportedOnFpga = false;
            }
            totalSize += size[u];
            // edges
            for (const pair<int,vector<int>> &p : incoming[u]) {
                int v = p.first;
                // does this edge, which is (v,u), cross this set?
                if (!extendedSet[v]) {
                    // yes
                    for (const int originalNodeWithOutgoingCommunication : p.second) {
                        originalNodesWithOutgoingCommunication.insert(originalNodeWithOutgoingCommunication);
                    }
                }
            }
            for (const pair<int,vector<int>> &p : outgoing[u]) {
                int v = p.first;
                // does this edge, which is (u,v), cross this set?
                if (!extendedSet[v]) {
                    // yes
                    for (const int originalNodeWithOutgoingCommunication : p.second) {
                        originalNodesWithOutgoingCommunication.insert(originalNodeWithOutgoingCommunication);
                    }
                }
            }
        }
    }
    s.totalCommunicationCost = 0.0;
    for (const int originalNodeWithOutgoingCommunication : originalNodesWithOutgoingCommunication) {
        s.totalCommunicationCost += originalOutgoingConnectionCost[originalNodeWithOutgoingCommunication];
    }
    if (!allNodesSupportedOnFpga || totalSize > ii.maxSizePerFpga) {
        s.totalFpgaLatency = INFTY;
    }
    return s;
}


Result Graph::runDP () {
    // compute the best split

    vector<vector<vector<double>>> dp(ii.maxCpus+1, vector<vector<double>>(ii.maxFpgas+1, vector<double>(ideals.size(), INFTY)));

    // if memory is a concern, this can be removed in lieu of partially rerunning the DP:
    vector<vector<vector<pair<bool,int>>>> backlinks(ii.maxCpus+1,
        vector<vector<pair<bool,int>>>(ii.maxFpgas+1, vector<pair<bool,int>>(ideals.size())));
    // {fpga=true, cpu=false; best subId}

    // get ID of ideal that contains all (forward) nodes
    const int idOfFullSet = idealToId[vector<bool>(countForwardNodes, true)];

    // ok, finally. here we go!
    for (int cpus = 0; cpus <= ii.maxCpus; ++cpus) {
        for (int fpgas = 0; fpgas <= ii.maxFpgas; ++fpgas) {
            // empty set, which has id=0
            dp[cpus][fpgas][0] = 0;

            for (int id = 1; id < ideals.size(); ++id) {

                // small optimization: if cpus==maxCpus and fpgas==maxFpgas,
                // then we don't care to compute this unless id == idOfFullSet
                if (cpus == ii.maxCpus && fpgas == ii.maxFpgas && id != idOfFullSet) {
                    continue;
                }

                double &currentDp = dp[cpus][fpgas][id];
                auto &backlink = backlinks[cpus][fpgas][id];
                currentDp = INFTY;
                if (cpus > 0) {
                    // we can use CPU
                    const vector<double> &prevDp = dp[cpus-1][fpgas];
                    // option 1: put nothing there
                    if (currentDp > prevDp[id]) {
                        currentDp = prevDp[id];
                        backlink = make_pair(false, id);
                    }
                    // option 2: iterate over subideals of ideal id
                    for (const pair<int,IdealPairCachedScores>& p : subIdeals[id]) {
                        int subId = p.first;
                        // cost of putting the set induced by ideal pair (id,subId) on CPU
                        double cost = p.second.totalCpuLatency; // CPU does not pay for communication
                        double candidate = max(prevDp[subId], cost);
                        if (currentDp > candidate) {
                            currentDp = candidate;
                            backlink = make_pair(false, subId);
                        }
                    }
                }
                if (fpgas > 0) {
                    // we can use FPGA
                    const vector<double> &prevDp = dp[cpus][fpgas-1];
                    // option 1: put nothing there
                    if (currentDp > prevDp[id]) {
                        currentDp = prevDp[id];
                        backlink = make_pair(true, id);
                    }
                    // option 2: iterate over subideals of ideal id
                    for (const pair<int,IdealPairCachedScores>& p : subIdeals[id]) {
                        int subId = p.first;
                        // cost of putting the set induced by ideal pair (id,subId) on FPGA
                        double cost = p.second.totalFpgaLatency + p.second.totalCommunicationCost; // sum! could be max as well
                        double candidate = max(prevDp[subId], cost);
                        if (currentDp > candidate) {
                            currentDp = candidate;
                            backlink = make_pair(true, subId);
                        }
                    }
                }
            }
        }
    }
    
    Result r;
    // we have the answer (max-load)
    r.maxLoad = dp[ii.maxCpus][ii.maxFpgas][idOfFullSet];
    double debugMaxLoad = 0; // will compute the same manually
    // now we backtrack to compute the optimal split
    int curId = idOfFullSet, curFpgas = ii.maxFpgas, curCpus = ii.maxCpus;
    while (curId != 0) { // curId is not empty set
        int subId = backlinks[curCpus][curFpgas][curId].second;
        bool isFpga = backlinks[curCpus][curFpgas][curId].first;
        IdealPairCachedScores score = prepareCachedScores(curId, subId);
        ResultMachine rm;
        rm.load = isFpga
                    ? score.totalFpgaLatency + score.totalCommunicationCost // sum! could be max as well
                    : score.totalCpuLatency;
        debugMaxLoad = max(debugMaxLoad, rm.load);
        vector<bool> extendedSet = getExtendedContiguousSet(curId, subId);
        // rm.nodes == {}
        for (int v = 0; v < extendedSet.size(); ++v) {
            if (extendedSet[v]) {
                // we insert the ORIGINAL nodes contained in supernode v.
                // to that end, find the node that has old-number v
                int debugCountSuchNodes = 0;
                for (const Node &n : ii.nodes) {
                    if (n.id == oldNumber[v]) {
                        // found it
                        append(rm.nodes, n.containedSubnodes);
                        ++debugCountSuchNodes;
                    }
                }
                assert(debugCountSuchNodes == 1);
            }
        }
        sort(rm.nodes.begin(), rm.nodes.end());
        (isFpga ? r.fpgas : r.cpus).push_back(rm);
        (isFpga ? curFpgas : curCpus)--;
        curId = subId;
    }
    if (r.maxLoad != debugMaxLoad) {
        fail("computed max load != max load from DP");
    }
    // NOTE: the output result might have fewer FPGAs than maxFpgas and fewer CPUs than maxCpus

    return r;
}



// helper stuff for fillDccIdsWithBiconnectedComponent()
struct BiconnectedComponents {

    map<pair<int,int>,int> component; // output

    unordered_map<int,vector<int>> adj;

    // give input using this
    void addEdge (int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
        component[make_pair(u,v)] = -1;
    }

    // temp
    unordered_map<int,int> low, entryTime;
    int currentBcc;
    int currentTime;
    vector<pair<int,int>> queue;

    void bccSearch (int v, int parent) {
        entryTime[v] = low[v] = currentTime++;
        for (const int w : adj[v]) {
            // edge (v,w)
            if (!entryTime.count(w)) {
                queue.emplace_back(v, w);
                bccSearch(w, v);
                if (low[w] >= entryTime[v]) {
                    // found a biconnected component
                    pair<int,int> e;
                    do {
                        e = queue.back();
                        // edge e goes into bcc with id `currentBcc`
                        if (component.count(e)) {
                            component[e] = currentBcc;
                        }
                        pair<int,int> eRev(e.second, e.first);
                        if (component.count(eRev)) {
                            component[eRev] = currentBcc;
                        }
                        queue.pop_back();
                    } while (e != make_pair(v, w));
                    currentBcc++;
                } else {
                    low[v] = min(low[v], low[w]);
                }
            } else if (entryTime[w] < entryTime[v] && w != parent) {
                queue.emplace_back(v, w);
                low[v] = min(low[v], entryTime[w]);
            }
        }
    }

    void go () {
        currentBcc = currentTime = 0;
        for (const auto &it : adj) {
            // for every vertex (it.first)
            if (!entryTime.count(it.first)) {
                bccSearch(it.first, -1);
            }
        }
    }
};
// end helper stuff for fillDccIdsWithBiconnectedComponent()



void InputInstance::fillDccIdsWithBiconnectedComponent() {
    constexpr bool BICONNECTED_DEBUG = false;
    BiconnectedComponents bicon;

    for (const Edge &e : edges) {
        // skip edges out of vertices that PipeDream would have ignored for this
        // (input layers, zero-size layers, and the like)
        if (originalFpgaLatency.size() == 32 || originalFpgaLatency.size() == 64) {
            // bert-24
            if (e.sourceId == 4 || e.sourceId == 37) {
                continue;
            }
        }
        if (originalFpgaLatency.size() == 96 || originalFpgaLatency.size() == 192) {
            // gnmt
            if (e.sourceId == 2  || e.sourceId == 100 ||
                e.sourceId == 40 || e.sourceId == 137 ||
                e.sourceId == 46 || e.sourceId == 142) {
                continue;
            }
        }
        bicon.addEdge(e.sourceId, e.destId);
    }

    bicon.go(); // compute biconnected components

    // then, for each biconnected component (which is a set of edges),
    // merge all vertices that are a tail of some edge in the component
    unordered_map<int,vector<int>> dccs; // dcc_id -> set of tails
    for (const auto &it : bicon.component) {
        // it = ((u,v),dcc)
        dccs[it.second].push_back(it.first.first);
    }
    unordered_map<int,int> ufTab;
    for (const Node &n : nodes) {
        ufTab[n.id] = n.id;
    }
    for (auto &it : dccs) {
        if (BICONNECTED_DEBUG) {
            dbg << "DCC: [";
            // sort and unique just to get nicer debug printout
            sort(it.second.begin(), it.second.end());
            it.second.erase(unique(it.second.begin(), it.second.end()), it.second.end());
            for (int x : it.second) {
                dbg << " " << x;
            }
            dbg << "]\n";
        }
        for (int i = 1; i < it.second.size(); ++i) {
            ufUnion(ufTab, it.second[0], it.second[i]);
        }
    }
    if (BICONNECTED_DEBUG) {
        dbg << "\nmapping:\n";
    }
    for (Node &n : nodes) {
        n.dcc = ufFind(ufTab, n.id);
        if (BICONNECTED_DEBUG) {
            dbg << n.id << " -> " << n.dcc << endl;
        }
    }
}









int main(int argc, char **argv) {

    // supported command-line options (only one at a time):
    // -scotch
    // -oneFpga
    // -localSearch numberOfRepeats
    // -expert splitFile
    // -pipeDream
    // -contractLayers
    // -linearize

    constexpr bool PERFORM_TRIVIAL_NODE_OPTIMIZATIONS = true;

    bool LINEARIZE = false;
    bool CONTRACT_BICONNECTED_COMPONENTS = false;


    if (argc > 1 && string(argv[1]) == string("-pipeDream")) {
        CONTRACT_BICONNECTED_COMPONENTS = true;
        LINEARIZE = true;
    }

    if (argc > 1 && string(argv[1]) == string("-linearize")) {
        LINEARIZE = true;
    }

    if (argc > 1 && string(argv[1]) == string("-contractLayers")) {
        CONTRACT_LAYERS = true;
    }


    json j;
    cin >> j;
    //dbg << "original instance: " << j.dump(4) << endl << endl << endl;
    InputInstance ii = j.get<InputInstance>();

    unordered_set<int> originalNodeIds;
    for (const Node &n : ii.nodes) {
        originalNodeIds.insert(n.id);
    }

    // sanity check
    if (!ii.isDAG()) {
        fail("input instance is not a DAG");
    }

    if (ii.hasBackwardToForwardEdge()) {
        fail("input instance has a backward-to-forward edge");
    }

    ii.reportNodesAndEdges();
    dbg << endl << endl;


    if (CONTRACT_LAYERS) {
        dbg << "fusing layers using color classes..." << endl;
        ii.fuseLayers();
    }

    dbg << endl;

    if (CONTRACT_LAYERS) {
        dbg << "contracting color classes AND LAYERS..." << endl;
    } else {
        dbg << "contracting color classes..." << endl;
    }
    
    ii = ii.contractColorClasses();
    ii.reportNodesAndEdges();
    dbg << endl << endl;


    if (CONTRACT_BICONNECTED_COMPONENTS) {
        dbg << "running biconnected components like PipeDream..." << endl;
        ii.fillDccIdsWithBiconnectedComponent();
        ii.fuseDccs();
        ii = ii.contractColorClasses();
    }

    //dbg << "contracted instance: " << json(ii).dump(4) << endl << endl << endl;
    
    if (!IGNORE_ORPHANED_BACKWARD_NODES) {
        dbg << "creating artificial forward nodes for orphaned backward nodes, if any..." << endl;
        ii.createArtificialForwardNodes();
    }

    ii.fuseStronglyConnectedComponentsInForward();

    ii = ii.contractColorClasses();
    ii.reportNodesAndEdges();
    dbg << endl << endl;

    if (argc > 1 && string(argv[1]) == string("-scotch")) {
        ii.runScotch();
        return 0;
    }

    if (argc > 1 && string(argv[1]) == string("-oneFpga")) {
        const double load = computeLoadForResultOrg(ii, allOnOneFpga());
        dbg << "single FPGA. computed load: " << load << endl;
        return 0;
    }

    if (argc > 1 && string(argv[1]) == string("-expert")) {
        if (argc <= 2) {
            fail("path to file containing human expert split required as second parameter");
        }
        const string filename = argv[2];
        json he;
        ifstream hefile(filename);
        hefile >> he;
        Result her = he.get<Result>();
        //dbg << "human expert from file " << filename << endl;
        ResultOrg hero = resultToResultOrg(her);
        const bool AUGMENT_HUMAN_EXPERT_TO_TRAINING_WITH_COLOCATIONS = true;
        if (AUGMENT_HUMAN_EXPERT_TO_TRAINING_WITH_COLOCATIONS) {
            //dbg << "will augment for training..." << endl;
            augmentHumanExpertToTrainingWithColocations(ii, hero);
        }
        const double maxLoad = computeLoadForResultOrg(ii, hero);
        dbg << "max-load of the split: " << maxLoad << endl;
        return 0;
    }

    if (argc > 1 && string(argv[1]) == string("-localSearch")) {
        if (argc <= 2) {
            fail("number of repeats required as second parameter. in the paper experiments we used 10");
        }
        int numberOfRepeats = atoi(argv[2]);
        ii.runLocalSearch(numberOfRepeats);
        return 0;
    }

    if (PERFORM_TRIVIAL_NODE_OPTIMIZATIONS) {
        ii.performTrivialNodeOptimizations();
        ii.reportNodesAndEdges();
    }

   if (LINEARIZE) {
        dbg << "linearizing..." << endl;
        ii.linearize();
        ii.reportNodesAndEdges();
    }

    //dbg << "contracted instance: " << json(ii).dump(4) << endl << endl << endl;

    // sanity check
    if (!ii.isDAG(true)) { // true = check only the forward part
        fail("input instance is not a DAG");
    }

    if (!IGNORE_ORPHANED_BACKWARD_NODES) {
        // sanity check
        if (ii.hasBackwardOnlyColorClass()) {
            fail("there is STILL a color class containing only backward nodes");
        }
    }

    dbg << endl << endl;
    //dbg << "contracted instance: " << json(ii).dump(4) << endl << endl << endl;
    dbg << "sanity checks and contractions done, now generate ideals..." << endl;

    Graph g(ii);

    dbg << "and run the DP..." << endl;

    Result r = g.runDP();
    cout << json(r).dump(4) << endl;

    // debug / sanity check
    unordered_set<int> unscheduledOriginalNodeIds = originalNodeIds;
    for (const auto &it : {r.cpus, r.fpgas}) {
        for (const auto &rm : it) {
            for (int id : rm.nodes) {
                assert(unscheduledOriginalNodeIds.count(id));
                unscheduledOriginalNodeIds.erase(id);
            }
        }
    }
    if (!unscheduledOriginalNodeIds.empty()) {
        fail("some nodes are not scheduled!");
    }

    dbg << "finished successfully. objective = " << r.maxLoad << endl;
    const double otherObjective = computeLoadForResult(ii, r);
    if (abs(r.maxLoad - otherObjective) > 1e-5) {
        dbg << "other objective computation: " << otherObjective << endl;
        fail("different maxLoad according to manual check");
    }
}
