/**
 * @file   min_cost_flow_cpu.h
 * @author Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_GLOBAL_MOVE_MIN_COST_FLOW_CPU_H
#define _DREAMPLACE_GLOBAL_MOVE_MIN_COST_FLOW_CPU_H

#include "lemon/smart_graph.h"
#include "lemon/network_simplex.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
class MinCostFlowCPULauncher
{
    public:
        const char* name() const 
        {
            return "MinCostFlowCPULauncher";
        }
        /// @brief solve assignment problem with min-cost flow algorithm 
        /// @param cost a nxn row-major cost matrix 
        /// @param sol solution mapping from row to column 
        /// @param n dimension 
        /// @param skip_threshold if the weight is larger than the threshold, do not add the edge 
        T run(const T* cost, int* sol, int n, T skip_threshold = std::numeric_limits<T>::max())
        {
            typedef lemon::SmartDigraph graph_type;
            graph_type graph; 
            graph.reserveNode(n*2);
            for (int i = 0; i < n*2; ++i)
            {
                graph.addNode();
            }
            graph.reserveArc(n*n);
            graph_type::ArcMap<T> edge_costs (graph); 
            graph_type::ArcMap<int> edge_capacities (graph); 
            graph_type::NodeMap<int> node_supply (graph);

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    auto c = cost[i*n+j];
                    if (c < skip_threshold)
                    {
                        auto arc = graph.addArc(graph.nodeFromId(i), graph.nodeFromId(n+j));
                        edge_costs[arc] = c;
                        edge_capacities[arc] = 1; 
                    }
                }
            }
            for (int i = 0; i < n; ++i)
            {
                node_supply[graph.nodeFromId(i)] = 1; 
            }
            for (int j = 0; j < n; ++j)
            {
                node_supply[graph.nodeFromId(n+j)] = -1; 
            }

            typedef lemon::NetworkSimplex<graph_type, 
                    int, 
                    T> alg_type;

            // 1. choose algorithm 
            alg_type alg (graph);

            // 2. run 
            typename alg_type::ProblemType status = alg.resetParams()
                .upperMap(edge_capacities)
                .costMap(edge_costs)
                .supplyMap(node_supply)
                .run();

            // 3. check results 
            if (status != alg_type::OPTIMAL)
            {
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        auto c = cost[i*n+j];
                        dreamplacePrint(kNONE, "%g ", (float)c);
                    }
                    dreamplacePrint(kNONE, "\n");
                }
                //dreamplaceAssertMsg(status == alg_type::OPTIMAL, "invalid status %d", status);
                dreamplacePrint(kDEBUG, "status is not OPTIMAL, use original solution instead\n");
                for (int i = 0; i < n; ++i)
                {
                    sol[i] = i; 
                }
                return std::numeric_limits<T>::max();
            }

            // 4. apply results 
            for (graph_type::ArcIt a (graph); a != lemon::INVALID; ++a)
            {
                if (alg.flow(a) > 0)
                {
                    int source = graph.id(graph.source(a));
                    int target = graph.id(graph.target(a));
                    dreamplaceAssert(source < n);
                    dreamplaceAssert(target >= n && target < 2*n);
                    sol[source] = target-n;
                }
            }
            // set total cost of min-cost flow 
            return alg.totalCost(); 
        }
};

DREAMPLACE_END_NAMESPACE

#endif
