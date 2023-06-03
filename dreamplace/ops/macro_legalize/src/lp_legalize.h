/**
 * @file   lp_legalize.h
 * @author Yibo Lin
 * @date   Nov 2019
 */

#ifndef DREAMPLACE_MACRO_LEGALIZE_LP_LEGALIZE_H
#define DREAMPLACE_MACRO_LEGALIZE_LP_LEGALIZE_H

#include <vector>
#include <array>
#include <functional>
#include <limbo/solvers/DualMinCostFlow.h>
#include <lemon/list_graph.h>
#include <lemon/smart_graph.h>
#include <lemon/connectivity.h>
#include "place_io/src/Util.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct NodeAttribute 
{
    T cost[2];
    T demand[2]; 
    T require[2]; 
    T pos[2]; ///< preferred location 

    T slack(int xy) const 
    {
        return require[xy] - demand[xy]; 
    }
};

template <typename T>
void longestPathLegalizeLauncher(LegalizationDB<T> db, const std::vector<int>& macros, const std::vector<int>& fixed_macros, 
        std::array<lemon::ListDigraph, 2>& cg, unsigned int& source, unsigned int& terminal)
{
    typedef lemon::ListDigraph lemon_graph_type; 
    unsigned int num_fixed_nodes = fixed_macros.size(); 
    unsigned int num_graph_nodes = macros.size() + num_fixed_nodes + 2;
    std::vector<NodeAttribute<T> > attribute_map (num_graph_nodes);
    source = macros.size() + num_fixed_nodes;
    terminal = source + 1; 

    dreamplacePrint(kDEBUG, "source %u, terminal %u\n", source, terminal);

    for (unsigned int i = 0; i < num_graph_nodes; ++i)
    {
        cg[kX].addNode();
        cg[kY].addNode();
    }

    for (unsigned int i = 0; i < num_graph_nodes; ++i)
    {
        auto& attr = attribute_map.at(i);

        if (i == source)
        {
            attr.cost[kX] = 0; 
            attr.cost[kY] = 0; 
            attr.demand[kX] = db.xl;
            attr.demand[kY] = db.yl;
            attr.require[kX] = std::numeric_limits<T>::max(); 
            attr.require[kY] = std::numeric_limits<T>::max(); 
            attr.pos[kX] = db.xl; 
            attr.pos[kY] = db.yl; 
        }
        else if (i == terminal)
        {
            attr.cost[kX] = 0; 
            attr.cost[kY] = 0; 
            attr.demand[kX] = std::numeric_limits<T>::lowest(); 
            attr.demand[kY] = std::numeric_limits<T>::lowest();  
            attr.require[kX] = db.xh; 
            attr.require[kY] = db.yh; 
            attr.pos[kX] = db.xh; 
            attr.pos[kY] = db.yh; 
        }
        else if (i < macros.size()) // movable macros 
        {
            int node_id = macros[i];
            attr.cost[kX] = db.node_size_x[node_id];
            attr.cost[kY] = db.node_size_y[node_id];
            attr.demand[kX] = std::numeric_limits<T>::lowest(); 
            attr.demand[kY] = std::numeric_limits<T>::lowest();  
            attr.require[kX] = std::numeric_limits<T>::max(); 
            attr.require[kY] = std::numeric_limits<T>::max(); 
            attr.pos[kX] = db.x[node_id];
            attr.pos[kY] = db.y[node_id];

            // arcs between S/T and movable macros 
            cg[kX].addArc(cg[kX].nodeFromId(source), cg[kX].nodeFromId(i));
            cg[kY].addArc(cg[kY].nodeFromId(source), cg[kY].nodeFromId(i));
            cg[kX].addArc(cg[kX].nodeFromId(i), cg[kX].nodeFromId(terminal));
            cg[kY].addArc(cg[kY].nodeFromId(i), cg[kY].nodeFromId(terminal));
        }
        else // fixed cells 
        {
            int node_id = fixed_macros.at(i - macros.size()); 
            attr.cost[kX] = db.node_size_x[node_id];
            attr.cost[kY] = db.node_size_y[node_id];
            attr.demand[kX] = attr.require[kX] = db.init_x[node_id];
            attr.demand[kY] = attr.require[kY] = db.init_y[node_id];
            attr.pos[kX] = db.x[node_id];
            attr.pos[kY] = db.y[node_id];

            // arcs between S/T and fixed cells  
            cg[kX].addArc(cg[kX].nodeFromId(source), cg[kX].nodeFromId(i));
            cg[kY].addArc(cg[kY].nodeFromId(source), cg[kY].nodeFromId(i));
            cg[kX].addArc(cg[kX].nodeFromId(i), cg[kX].nodeFromId(terminal));
            cg[kY].addArc(cg[kY].nodeFromId(i), cg[kY].nodeFromId(terminal));
        }
    }

    auto add2Hcg = [&](int i, T xl1, T width1, int j, T xl2, T width2){
        auto var1 = cg[kX].nodeFromId(i);
        auto var2 = cg[kX].nodeFromId(j);
        T dx1 = std::max(xl1 - xl2 + width1, (T)0);
        T dx2 = std::max(xl2 - xl1 + width2, (T)0);
        if (dx1 < dx2) // (i, j) is easier to resolve overlap 
        {
            cg[kX].addArc(var1, var2); 
        }
        else // (j, i) is easier to resolve overlap  
        {
            cg[kX].addArc(var2, var1); 
        }
    };
    auto add2Vcg = [&](int i, T yl1, T height1, int j, T yl2, T height2){
        auto var1 = cg[kY].nodeFromId(i);
        auto var2 = cg[kY].nodeFromId(j);
        T dy1 = std::max(yl1 - yl2 + height1, (T)0);
        T dy2 = std::max(yl2 - yl1 + height2, (T)0);
        if (dy1 < dy2) // (i, j) is easier to resolve overlap 
        {
            cg[kY].addArc(var1, var2); 
        }
        else // (j, i) is easier to resolve overlap 
        {
            cg[kY].addArc(var2, var1); 
        }
    };

    auto process2Nodes = [&](int i, T xl1, T yl1, T width1, T height1, int j, T xl2, T yl2, T width2, T height2) {
        T xh1 = xl1 + width1;
        T yh1 = yl1 + height1;
        T xh2 = xl2 + width2;
        T yh2 = yl2 + height2;
        T dx = std::max(xl1, xl2) - std::min(xh1, xh2);
        T dy = std::max(yl1, yl2) - std::min(yh1, yh2);

        if (dx < 0 && dy < 0) // case I: overlap
        {
            T hmove = std::min(xh2 - xl1, xh1 - xl2);
            T vmove = std::min(yh2 - yl1, yh1 - yl2);
            if (hmove < vmove) // horizontal movement has better displacement
            {
                add2Hcg(i, xl1, width1, j, xl2, width2);
            }
            else // vertical movement has better displacement
            {
                add2Vcg(i, yl1, height1, j, yl2, height2);
            }
        }
        else if (dx >= 0 && dy < 0) // case II: two cells intersect in y direction
        {
            add2Hcg(i, xl1, width1, j, xl2, width2);
        }
        else if (dx < 0 && dy >= 0) // case III: two cells intersect in x direction
        {
            add2Vcg(i, yl1, height1, j, yl2, height2);
        }
        else // case IV: diagonal, dx > 0 && dy > 0
        {
            if (dx < dy) // vertical constraint is easier to satisfy 
            {
                add2Vcg(i, yl1, height1, j, yl2, height2);
            }
            else // horizontal constraint is easier to satisfy
            {
                add2Hcg(i, xl1, width1, j, xl2, width2);
            }
        }
    };

    // construct horizontal and vertical constraint graph 
    // use current locations for constraint graphs 
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id1 = macros[i];
        T xl1 = db.x[node_id1];
        T yl1 = db.y[node_id1];
        T width1 = db.node_size_x[node_id1];
        T height1 = db.node_size_y[node_id1];

        // constraints with other macros 
        for (unsigned int j = i+1; j < ie; ++j)
        {
            int node_id2 = macros[j];
            T xl2 = db.x[node_id2];
            T yl2 = db.y[node_id2];
            T width2 = db.node_size_x[node_id2];
            T height2 = db.node_size_y[node_id2];

            process2Nodes(i, xl1, yl1, width1, height1, j, xl2, yl2, width2, height2);
        }
        // constraints with fixed macros 
        // when considering fixed macros, there is no guarantee to find legal solution 
        // with current ad-hoc constraint graphs 
        for (unsigned int j = 0, je = fixed_macros.size(); j < je; ++j)
        {
            int node_id2 = fixed_macros.at(j); 
            T xl2 = db.init_x[node_id2];
            T yl2 = db.init_y[node_id2];
            T width2 = db.node_size_x[node_id2];
            T height2 = db.node_size_y[node_id2];

            process2Nodes(i, xl1, yl1, width1, height1, j + macros.size(), xl2, yl2, width2, height2);
        }
    }

    // v must be the terminal 
    std::function<void(int, int)> update_demand_recursive = [&](int xy, int v){
        if (v != (int)source)
        {
            attribute_map.at(v).demand[xy] = attribute_map.at(source).demand[xy]; 
            for (lemon_graph_type::InArcIt a (cg[xy], cg[xy].nodeFromId(v)); a != lemon::INVALID; ++a)
            {
                auto u = cg[xy].id(cg[xy].source(a));
                dreamplaceAssert(u != v);
                dreamplaceAssert((unsigned)v < num_graph_nodes);
                dreamplaceAssert((unsigned)u < num_graph_nodes);
                if (attribute_map.at(u).demand[xy] < attribute_map.at(source).demand[xy]) // not computed 
                {
                    update_demand_recursive(xy, u);
                }
                dreamplaceAssertMsg(attribute_map.at(u).demand[xy] >= attribute_map.at(source).demand[xy], "demand[%d] = %g, demand[s%u] = %g, v = %d", u, attribute_map.at(u).demand[xy], source, attribute_map.at(source).demand[xy], v);
                attribute_map.at(v).demand[xy] = std::max(attribute_map.at(u).demand[xy] + attribute_map.at(u).cost[xy], attribute_map.at(v).demand[xy]);
            }
        }
    };

    // v must be the source 
    std::function<void(int, int)> update_require_recursive = [&](int xy, int v){
        if (v != (int)terminal)
        {
            attribute_map.at(v).require[xy] = attribute_map.at(terminal).require[xy]; 
            for (lemon_graph_type::OutArcIt a (cg[xy], cg[xy].nodeFromId(v)); a != lemon::INVALID; ++a)
            {
                auto u = cg[xy].id(cg[xy].target(a));
                dreamplaceAssert(u != v);
                dreamplaceAssert((unsigned)v < num_graph_nodes);
                dreamplaceAssert((unsigned)u < num_graph_nodes);
                if (attribute_map.at(u).require[xy] > attribute_map.at(terminal).require[xy]) // not computed 
                {
                    update_require_recursive(xy, u);
                }
                dreamplaceAssert(attribute_map.at(u).require[xy] <= attribute_map.at(terminal).require[xy]);
                attribute_map.at(v).require[xy] = std::min(attribute_map.at(u).require[xy] - attribute_map.at(v).cost[xy], attribute_map.at(v).require[xy]);
            }
        }
    };

    // must be the source 
    auto unset_downstream_demand = [&](int xy, int v){
        std::deque<int> queue; 
        std::vector<unsigned char> visited (num_graph_nodes, false);

        visited[v] = true; 
        queue.push_back(v);

        while (!queue.empty())
        {
            v = queue.front();
            queue.pop_front();

            if (v != (int)source)
            {
                attribute_map.at(v).demand[xy] = std::numeric_limits<T>::lowest(); 
            }

            for (lemon_graph_type::OutArcIt a (cg[xy], cg[xy].nodeFromId(v)); a != lemon::INVALID; ++a)
            {
                auto u = cg[xy].id(cg[xy].target(a));
                dreamplaceAssert(u != v);
                if (!visited[u])
                {
                    visited[u] = true; 
                    queue.push_back(u);
                }
            }
        }
    };

    // must be the terminal 
    auto unset_upstream_require = [&](int xy, int v){
        std::deque<int> queue; 
        std::vector<unsigned char> visited (num_graph_nodes, false);

        visited[v] = true; 
        queue.push_back(v);

        while (!queue.empty())
        {
            v = queue.front();
            queue.pop_front();

            if (v != (int)terminal)
            {
                attribute_map.at(v).require[xy] = std::numeric_limits<T>::max(); 
            }

            for (lemon_graph_type::InArcIt a (cg[xy], cg[xy].nodeFromId(v)); a != lemon::INVALID; ++a)
            {
                auto u = cg[xy].id(cg[xy].source(a));
                dreamplaceAssert(u != v);
                if (!visited[u])
                {
                    visited[u] = true; 
                    queue.push_back(u);
                }
            }
        }
    };

#ifdef DEBUG
    auto unset_slack = [&](){
        for (unsigned int v = 0; v < macros.size(); ++v)
        {
            auto& attr = attribute_map.at(v);
            attr.demand[kX] = std::numeric_limits<T>::lowest();
            attr.demand[kY] = std::numeric_limits<T>::lowest();
            attr.require[kX] = std::numeric_limits<T>::max();
            attr.require[kY] = std::numeric_limits<T>::max();
        }
        attribute_map.at(source).require[kX] = std::numeric_limits<T>::max();
        attribute_map.at(source).require[kY] = std::numeric_limits<T>::max();
        attribute_map.at(terminal).demand[kX] = std::numeric_limits<T>::lowest();
        attribute_map.at(terminal).demand[kY] = std::numeric_limits<T>::lowest();
    };
#endif

    auto evaluate_slack = [&](const char* msg){
        T wns[2] = {0, 0};
        T tns[2] = {0, 0};
        for (unsigned int v = 0; v < macros.size(); ++v)
        {
            T slack[2] = {std::min(attribute_map.at(v).slack(kX), (T)0), std::min(attribute_map.at(v).slack(kY), (T)0)};
            wns[kX] = std::min(wns[kX], slack[kX]);
            wns[kY] = std::min(wns[kY], slack[kY]);
            tns[kX] += slack[kX];
            tns[kY] += slack[kY];
        }
        dreamplacePrint(kDEBUG, "%s TNS[X/Y] = %g/%g, WNS[X/Y] = %g/%g\n", msg, tns[kX], tns[kY], wns[kX], wns[kY]);
    };

#ifdef DEBUG
    auto print_slack = [&](){
        for (unsigned int v = 0; v < num_graph_nodes; ++v)
        {
            auto const& attr = attribute_map.at(v); 
            dreamplacePrint(kDEBUG, "[%u] demand %g/%g, require %g/%g, slack %g/%g\n", v, attr.demand[kX], attr.demand[kY], attr.require[kX], attr.require[kY], attr.slack(kX), attr.slack(kY));
        }
    };
#endif

    auto compute_slack = [&](){
        for (int xy = 0; xy < 2; ++xy)
        {
            // from source to compute demand_map 
            update_demand_recursive(xy, terminal);
            // from terminal to compute require_map 
            update_require_recursive(xy, source);
        }
    };

    // compute slack 
    compute_slack();

    //print_slack();
    evaluate_slack("Original slack:");

    std::vector<int> orders (macros.size()); 
    std::iota(orders.begin(), orders.end(), 0);
    std::sort(orders.begin(), orders.end(), 
            [&](int i, int j){
                int node_id1 = macros[i];
                int node_id2 = macros[j];
                T a1 = db.node_size_x[node_id1] * db.node_size_y[node_id1];
                T a2 = db.node_size_x[node_id2] * db.node_size_y[node_id2];
                T x1 = db.x[node_id1]; 
                T x2 = db.x[node_id2]; 
                return a1 < a2 || (a1 == a2 && (x1 < x2 || (x1 == x2 && node_id1 < node_id2))); 
            });

    for (auto v : orders)
    {
        auto& attr = attribute_map.at(v); 
        int xy = std::numeric_limits<int>::max();
        if (attr.slack(kX) < 0 && attr.slack(kY) >= 0)
        {
            xy = kX; 
        }
        else if (attr.slack(kY) < 0 && attr.slack(kX) >= 0)
        {
            xy = kY; 
        }
        if (xy < std::numeric_limits<int>::max())
        {
            // record the sources of arcs to change 
            std::vector<std::pair<lemon_graph_type::InArcIt, int>> arcs; 
            for (lemon_graph_type::InArcIt a (cg[xy], cg[xy].nodeFromId(v)); a != lemon::INVALID; ++a)
            {
                auto u = cg[xy].id(cg[xy].source(a));
                auto const& attr_u = attribute_map.at(u);
                if (attr_u.demand[xy] + attr_u.cost[xy] == attr.demand[xy])
                {
                    bool uv_flag = false; 
                    bool vu_flag = false; 
                    // the heuristic here is that 
                    // any insertion of arcs cannot change the demand of the sinks of u or v 
                    if (attr_u.demand[!xy] + attr_u.cost[!xy] <= attr.require[!xy])
                    {
                        uv_flag = true; 
                        for (lemon_graph_type::OutArcIt b (cg[!xy], cg[!xy].nodeFromId(v)); b != lemon::INVALID; ++b)
                        {
                            auto w = cg[!xy].id(cg[!xy].target(b));
                            auto const& attr_w = attribute_map.at(w);
                            if (attr_u.demand[!xy] + attr_u.cost[!xy] + attr.cost[!xy] > attr_w.demand[!xy]) // u -> v
                            {
                                uv_flag = false; 
                                break; 
                            }
                        }
                    }
                    if (attr.demand[!xy] + attr.cost[!xy] <= attr_u.require[!xy])
                    {
                        vu_flag = true; 
                        for (lemon_graph_type::OutArcIt b (cg[!xy], cg[!xy].nodeFromId(u)); b != lemon::INVALID; ++b)
                        {
                            auto w = cg[!xy].id(cg[!xy].target(b));
                            auto const& attr_w = attribute_map.at(w);
                            if (attr.demand[!xy] + attr.cost[!xy] + attr_u.cost[!xy] > attr_w.demand[!xy]) // v -> u
                            {
                                vu_flag = false; 
                                break; 
                            }
                        }
                    }
                    if (uv_flag || vu_flag)
                    {
                        arcs.push_back(std::make_pair(a, (uv_flag)? 0 : 1));
                    }
                    else // give up 
                    {
                        arcs.clear();
                        break; 
                    }
                }
            }
            for (auto aa : arcs)
            {
                auto a = aa.first;
                bool uv_flag = (aa.second == 0)? 1 : 0;
                unsigned int u = cg[xy].id(cg[xy].source(a)); 
                if (u == source || u == terminal) // skip source/terminal 
                {
                    continue; 
                }
                // add arc to another graph 
                if (uv_flag) // u, v
                {
                    cg[!xy].addArc(cg[!xy].nodeFromId(u), cg[!xy].nodeFromId(v));
                }
                else // v, u
                {
                    cg[!xy].addArc(cg[!xy].nodeFromId(v), cg[!xy].nodeFromId(u));
                }
                // remove arc 
                cg[xy].erase(a);

                dreamplacePrint(kDEBUG, "switch arc (%d, %d) from %d to %d\n", u, v, xy, !xy);

                unset_downstream_demand(xy, v);
                update_demand_recursive(xy, terminal); 
                unset_upstream_require(xy, u);
                update_require_recursive(xy, source);

                unset_downstream_demand(!xy, v);
                update_demand_recursive(!xy, terminal); 
                unset_upstream_require(!xy, u);
                update_require_recursive(!xy, source);
            }
        }
    }

    //print_slack();
    evaluate_slack("Adjusted slack:");

#ifdef DEBUG
    unset_slack(); 
    compute_slack();
    //print_slack();
    evaluate_slack();

    for (int xy = 0; xy < 2; ++xy)
    {
        std::vector<unsigned char> markers (num_graph_nodes - 2, false); 
        for (lemon_graph_type::OutArcIt a (cg[xy], cg[xy].nodeFromId(source)); a != lemon::INVALID; ++a)
        {
            auto u = cg[xy].id(cg[xy].target(a)); 
            markers.at(u) = true; 
        }
        dreamplaceAssert(std::find(markers.begin(), markers.end(), 0) == markers.end());
    }
    for (int xy = 0; xy < 2; ++xy)
    {
        std::vector<unsigned char> markers (num_graph_nodes - 2, false); 
        for (lemon_graph_type::InArcIt a (cg[xy], cg[xy].nodeFromId(terminal)); a != lemon::INVALID; ++a)
        {
            auto u = cg[xy].id(cg[xy].source(a)); 
            markers.at(u) = true; 
        }
        dreamplaceAssert(std::find(markers.begin(), markers.end(), 0) == markers.end());
    }
#endif
}

/// @brief A linear programming (LP) based algorithm to legalize macros. 
/// It assumes the relative order of macros are determined. 
/// By constructing the horizontal and vertical constraint graph, 
/// an optimization problem is formulated to minimize the total displacement. 
/// The LP problem can be solved by dual min-cost flow algorithm. 
/// 
/// If the input macro solution is not legal, there is no guarantee to find a legal solution. 
/// But if it is legal, the output should still be legal. 
template <typename T>
void lpLegalizeGraphLauncher(LegalizationDB<T> db, const std::vector<int>& macros, const std::vector<int>& fixed_macros)
{
    dreamplacePrint(kINFO, "Legalize movable macros with linear programming on constraint graphs\n");

    // numeric type can be int, long ,double, not never use float. 
    // It will cause incorrect results and introduce overlap. 
    // Meanwhile, integers are recommended, as the coefficients are forced to be integers. 
    typedef long numeric_type; 
    typedef limbo::solvers::LinearModel<numeric_type, numeric_type> model_type;
    typedef limbo::solvers::DualMinCostFlow<numeric_type, numeric_type> solver_type; 
    typedef limbo::solvers::NetworkSimplex<numeric_type, numeric_type> solver_alg_type; 

    typedef lemon::ListDigraph lemon_graph_type; 
    std::array<lemon_graph_type, 2> cg; 
    unsigned int source = std::numeric_limits<unsigned int>::max();
    unsigned int terminal = std::numeric_limits<unsigned int>::max();
    longestPathLegalizeLauncher(db, macros, fixed_macros, cg, source, terminal);

    char buf[64];
    // two linear programming models represent horizontal and vertical constraint graphs
    model_type model_hcg; 
    model_hcg.reserveVariables(macros.size()*3); // position variables + displace variables (l, u)
    typename model_type::expression_type obj_hcg; 
    model_type model_vcg; 
    model_vcg.reserveVariables(macros.size()*3); // position variables + displace variables (l, u)
    typename model_type::expression_type obj_vcg; 

    // position variables x
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];
        T width = db.node_size_x[node_id];
        T height = db.node_size_y[node_id];

        dreamplaceSPrint(kNONE, buf, "x%d", node_id);
        model_hcg.addVariable(db.xl, db.xh-width, limbo::solvers::CONTINUOUS, buf);
        model_vcg.addVariable(db.yl, db.yh-height, limbo::solvers::CONTINUOUS, buf);
    }
    // displacement variables l = min(x, x0)
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];

        dreamplaceSPrint(kNONE, buf, "l%d", node_id);
        model_hcg.addVariable(0, db.xh, limbo::solvers::CONTINUOUS, buf);
        model_vcg.addVariable(0, db.yh, limbo::solvers::CONTINUOUS, buf);
    }
    // displacement variables u = max(x, x0)
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];

        dreamplaceSPrint(kNONE, buf, "u%d", node_id);
        model_hcg.addVariable(0, db.xh, limbo::solvers::CONTINUOUS, buf);
        model_vcg.addVariable(0, db.yh, limbo::solvers::CONTINUOUS, buf);
    }

    for (int xy = 0; xy < 2; ++xy)
    {
        auto& model = (xy == kX)? model_hcg : model_vcg;
        const T* db_x = (xy == kX)? db.x : db.y; 
        const T* db_node_size_x = (xy == kX)? db.node_size_x : db.node_size_y;
        for (lemon_graph_type::ArcIt a (cg[xy]); a != lemon::INVALID; ++a)
        {
            unsigned int v = cg[xy].id(cg[xy].source(a));
            unsigned int u = cg[xy].id(cg[xy].target(a));

            if (v < macros.size()) // v is movable macro
            {
                int node_id1 = macros[v];
                T width1 = db_node_size_x[node_id1];
                auto var1 = model.variable(v);

                if (u < macros.size()) // u is movable macro 
                {
                    auto var2 = model.variable(u);
                    dreamplaceAssertMsg(model.addConstraint(var1 - var2 <= -width1), "failed to add %s constraint", (xy == kX)? "HCG" : "VCG");
                }
                else if (u != source && u != terminal) // u is fixed cell 
                {
                    int node_id2 = fixed_macros.at(u - macros.size()); 
                    T xl2 = db_x[node_id2];
                    model.updateVariableUpperBound(var1, floor(xl2 - width1));
                }
            }
            else if (v != source && v != terminal) // v is fixed cell 
            {
                int node_id1 = fixed_macros.at(v - macros.size());
                T xl1 = db_x[node_id1];
                T width1 = db_node_size_x[node_id1];

                if (u < macros.size()) // u is movable macro 
                {
                    auto var2 = model.variable(u);
                    model.updateVariableLowerBound(var2, ceil(xl1 + width1));
                }
            }
        }
    }

    // displacement constraints and objectives
    // Use initial locations for objective computation 
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];
        T xl = round(db.init_x[node_id]);
        T yl = round(db.init_y[node_id]);

        auto var_x = model_hcg.variable(i); 
        auto var_l = model_hcg.variable(i + macros.size());
        auto var_u = model_hcg.variable(i + macros.size()*2);
        dreamplaceAssertMsg(model_hcg.addConstraint(var_l - var_x <= 0), "failed to add HCG lower bound constraint");
        model_hcg.updateVariableUpperBound(var_l, xl);
        dreamplaceAssertMsg(model_hcg.addConstraint(var_u - var_x >= 0), "failed to add HCG upper bound constraint");
        model_hcg.updateVariableLowerBound(var_u, xl);
        obj_hcg += var_u - var_l;

        var_x = model_vcg.variable(i); 
        var_l = model_vcg.variable(i + macros.size());
        var_u = model_vcg.variable(i + macros.size()*2);
        dreamplaceAssertMsg(model_vcg.addConstraint(var_l - var_x <= 0), "failed to add VCG lower bound constraint");
        model_vcg.updateVariableUpperBound(var_l, yl);
        dreamplaceAssertMsg(model_vcg.addConstraint(var_u - var_x >= 0), "failed to add VCG upper bound constraint");
        model_vcg.updateVariableLowerBound(var_u, yl);
        obj_vcg += var_u - var_l;
    }

    model_hcg.setObjective(obj_hcg);
    model_hcg.setOptimizeType(limbo::solvers::MIN);
    model_vcg.setObjective(obj_vcg);
    model_vcg.setOptimizeType(limbo::solvers::MIN);

#ifdef DEBUG
    model_hcg.print("hcg.lp");
    model_vcg.print("vcg.lp");
#endif

    // solve linear programming for horizontal constraint graph
    {
        solver_alg_type alg; 
        solver_type solver (&model_hcg); 
        auto status = solver(&alg);
        dreamplaceAssertMsg(status == limbo::solvers::OPTIMAL, "Horizontal graph not solved optimally");

        for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
        {
            int node_id = macros[i];
            db.x[node_id] = model_hcg.variableSolution(model_hcg.variable(i));
        }
    }
    // solve linear programming for vertical constraint graph
    {
        solver_alg_type alg; 
        solver_type solver (&model_vcg); 
        auto status = solver(&alg);
        dreamplaceAssertMsg(status == limbo::solvers::OPTIMAL, "Vertical graph not solved optimally");

        for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
        {
            int node_id = macros[i];
            db.y[node_id] = model_vcg.variableSolution(model_vcg.variable(i));
        }
    }

#ifdef DEBUG
    model_hcg.printSolution("hcg.sol");
    model_vcg.printSolution("vcg.sol");
#endif
}

template <typename T>
void lpLegalizeLauncher(LegalizationDB<T> db, const std::vector<int>& macros, const std::vector<int>& fixed_macros)
{
    dreamplacePrint(kINFO, "Legalize movable macros with linear programming on constraint graphs\n");

    // numeric type can be int, long ,double, not never use float. 
    // It will cause incorrect results and introduce overlap. 
    // Meanwhile, integers are recommended, as the coefficients are forced to be integers. 
    typedef long numeric_type; 
    typedef limbo::solvers::LinearModel<numeric_type, numeric_type> model_type;
    typedef limbo::solvers::DualMinCostFlow<numeric_type, numeric_type> solver_type; 
    typedef limbo::solvers::NetworkSimplex<numeric_type, numeric_type> solver_alg_type; 

    char buf[64];
    // two linear programming models represent horizontal and vertical constraint graphs
    model_type model_hcg; 
    model_hcg.reserveVariables(macros.size()*3); // position variables + displace variables (l, u)
    typename model_type::expression_type obj_hcg; 
    model_type model_vcg; 
    model_vcg.reserveVariables(macros.size()*3); // position variables + displace variables (l, u)
    typename model_type::expression_type obj_vcg; 

    // position variables x
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];
        T width = db.node_size_x[node_id];
        T height = db.node_size_y[node_id];

        dreamplaceSPrint(kNONE, buf, "x%d", node_id);
        model_hcg.addVariable(db.xl, db.xh-width, limbo::solvers::CONTINUOUS, buf);
        model_vcg.addVariable(db.yl, db.yh-height, limbo::solvers::CONTINUOUS, buf);
    }
    // displacement variables l = min(x, x0)
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];

        dreamplaceSPrint(kNONE, buf, "l%d", node_id);
        model_hcg.addVariable(0, db.xh, limbo::solvers::CONTINUOUS, buf);
        model_vcg.addVariable(0, db.yh, limbo::solvers::CONTINUOUS, buf);
    }
    // displacement variables u = max(x, x0)
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];

        dreamplaceSPrint(kNONE, buf, "u%d", node_id);
        model_hcg.addVariable(0, db.xh, limbo::solvers::CONTINUOUS, buf);
        model_vcg.addVariable(0, db.yh, limbo::solvers::CONTINUOUS, buf);
    }

    auto add2Hcg = [&](int i, T xl1, T width1, int j, T xl2, T width2){
        auto var1 = model_hcg.variable(i);
        if (j < db.num_movable_nodes) // movable macro 
        {
            auto var2 = model_hcg.variable(j);
            if (xl1 < xl2)
            {
                dreamplaceAssertMsg(model_hcg.addConstraint(var1 - var2 <= -width1), "failed to add HCG constraint");
            }
            else 
            {
                dreamplaceAssertMsg(model_hcg.addConstraint(var2 - var1 <= -width2), "failed to add HCG constraint");
            }
        }
        else // j is fixed macro 
        {
            if (xl1 < xl2)
            {
                model_hcg.updateVariableUpperBound(var1, floor(xl2 - width1));
                //dreamplacePrint(kDEBUG, "HCG: %s <= x%d (%g) - %g\n", model_hcg.variableName(var1).c_str(), j, xl2, width1);
            }
            else 
            {
                model_hcg.updateVariableLowerBound(var1, ceil(xl2 + width2));
                //dreamplacePrint(kDEBUG, "HCG: %s >= x%d (%g) + %g\n", model_hcg.variableName(var1).c_str(), j, xl2, width2);
            }
        }
    };
    auto add2Vcg = [&](int i, T yl1, T height1, int j, T yl2, T height2){
        auto var1 = model_vcg.variable(i);
        if (j < db.num_movable_nodes) // movable macro 
        {
            auto var2 = model_vcg.variable(j);
            if (yl1 < yl2)
            {
                dreamplaceAssertMsg(model_vcg.addConstraint(var1 - var2 <= -height1), "failed to add VCG constraint");
            }
            else 
            {
                dreamplaceAssertMsg(model_vcg.addConstraint(var2 - var1 <= -height2), "failed to add VCG constraint");
            }
        }
        else // j is fixed macro 
        {
            if (yl1 < yl2)
            {
                model_vcg.updateVariableUpperBound(var1, floor(yl2 - height1)); 
                //dreamplacePrint(kDEBUG, "VCG: %s <= x%d (%g) - %g\n", model_vcg.variableName(var1).c_str(), j, yl2, height1);
            }
            else 
            {
                model_vcg.updateVariableLowerBound(var1, ceil(yl2 + height2));
                //dreamplacePrint(kDEBUG, "VCG: %s >= x%d (%g) + %g\n", model_vcg.variableName(var1).c_str(), j, yl2, height2);
            }
        }
    };

    auto process2Nodes = [&](int i, T xl1, T yl1, T width1, T height1, int j, T xl2, T yl2, T width2, T height2) {
        T xh1 = xl1 + width1;
        T yh1 = yl1 + height1;
        T xh2 = xl2 + width2;
        T yh2 = yl2 + height2;
        T dx = std::max(xl1, xl2) - std::min(xh1, xh2);
        T dy = std::max(yl1, yl2) - std::min(yh1, yh2);

        if (dx < 0 && dy < 0) // case I: overlap
        {
            T hmove = std::min(xh2 - xl1, xh1 - xl2);
            T vmove = std::min(yh2 - yl1, yh1 - yl2);
            if (hmove < vmove) // horizontal movement has better displacement
            {
                add2Hcg(i, xl1, width1, j, xl2, width2);
            }
            else // vertical movement has better displacement
            {
                add2Vcg(i, yl1, height1, j, yl2, height2);
            }
        }
        else if (dx >= 0 && dy < 0) // case II: two cells intersect in y direction
        {
            add2Hcg(i, xl1, width1, j, xl2, width2);
        }
        else if (dx < 0 && dy >= 0) // case III: two cells intersect in x direction
        {
            add2Vcg(i, yl1, height1, j, yl2, height2);
        }
        else // case IV: diagonal, dx > 0 && dy > 0
        {
            if (dx < dy) // vertical constraint is easier to satisfy 
            {
                add2Vcg(i, yl1, height1, j, yl2, height2);
            }
            else // horizontal constraint is easier to satisfy
            {
                add2Hcg(i, xl1, width1, j, xl2, width2);
            }
        }
    };

    // construct horizontal and vertical constraint graph 
    // use current locations for constraint graphs 
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id1 = macros[i];
        T xl1 = db.x[node_id1];
        T yl1 = db.y[node_id1];
        T width1 = db.node_size_x[node_id1];
        T height1 = db.node_size_y[node_id1];
        // constraints with other macros 
        for (unsigned int j = i+1; j < ie; ++j)
        {
            int node_id2 = macros[j];
            T xl2 = db.x[node_id2];
            T yl2 = db.y[node_id2];
            T width2 = db.node_size_x[node_id2];
            T height2 = db.node_size_y[node_id2];

            process2Nodes(i, xl1, yl1, width1, height1, j, xl2, yl2, width2, height2);
        }
        // constraints with fixed macros 
        // when considering fixed macros, there is no guarantee to find legal solution 
        // with current ad-hoc constraint graphs 
        for (unsigned int j = 0, je = fixed_macros.size(); j < je; ++j)
        {
            int node_id2 = fixed_macros.at(j); 
            T xl2 = db.init_x[node_id2];
            T yl2 = db.init_y[node_id2];
            T width2 = db.node_size_x[node_id2];
            T height2 = db.node_size_y[node_id2];

            process2Nodes(i, xl1, yl1, width1, height1, node_id2, xl2, yl2, width2, height2);
        }
    }

    // displacement constraints and objectives
    // Use initial locations for objective computation 
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];
        T xl = round(db.init_x[node_id]);
        T yl = round(db.init_y[node_id]);

        auto var_x = model_hcg.variable(i); 
        auto var_l = model_hcg.variable(i + macros.size());
        auto var_u = model_hcg.variable(i + macros.size()*2);
        dreamplaceAssertMsg(model_hcg.addConstraint(var_l - var_x <= 0), "failed to add HCG lower bound constraint");
        model_hcg.updateVariableUpperBound(var_l, xl);
        dreamplaceAssertMsg(model_hcg.addConstraint(var_u - var_x >= 0), "failed to add HCG upper bound constraint");
        model_hcg.updateVariableLowerBound(var_u, xl);
        obj_hcg += var_u - var_l;

        var_x = model_vcg.variable(i); 
        var_l = model_vcg.variable(i + macros.size());
        var_u = model_vcg.variable(i + macros.size()*2);
        dreamplaceAssertMsg(model_vcg.addConstraint(var_l - var_x <= 0), "failed to add VCG lower bound constraint");
        model_vcg.updateVariableUpperBound(var_l, yl);
        dreamplaceAssertMsg(model_vcg.addConstraint(var_u - var_x >= 0), "failed to add VCG upper bound constraint");
        model_vcg.updateVariableLowerBound(var_u, yl);
        obj_vcg += var_u - var_l;
    }

    model_hcg.setObjective(obj_hcg);
    model_hcg.setOptimizeType(limbo::solvers::MIN);
    model_vcg.setObjective(obj_vcg);
    model_vcg.setOptimizeType(limbo::solvers::MIN);

//#ifdef DEBUG
    model_hcg.print("hcg.lp");
    model_vcg.print("vcg.lp");
//#endif

    // solve linear programming for horizontal constraint graph
    {
        solver_alg_type alg; 
        solver_type solver (&model_hcg); 
        auto status = solver(&alg);
        dreamplaceAssertMsg(status == limbo::solvers::OPTIMAL, "Horizontal graph not solved optimally");

        for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
        {
            int node_id = macros[i];
            db.x[node_id] = model_hcg.variableSolution(model_hcg.variable(i));
        }
    }
    // solve linear programming for vertical constraint graph
    {
        solver_alg_type alg; 
        solver_type solver (&model_vcg); 
        auto status = solver(&alg);
        dreamplaceAssertMsg(status == limbo::solvers::OPTIMAL, "Vertical graph not solved optimally");

        for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
        {
            int node_id = macros[i];
            db.y[node_id] = model_vcg.variableSolution(model_vcg.variable(i));
        }
    }

#ifdef DEBUG
    model_hcg.printSolution("hcg.sol");
    model_vcg.printSolution("vcg.sol");
#endif
}

DREAMPLACE_END_NAMESPACE

#endif
