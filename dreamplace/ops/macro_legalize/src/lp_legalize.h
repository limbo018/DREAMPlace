/**
 * @file   lp_legalize.h
 * @author Yibo Lin
 * @date   Nov 2019
 */

#ifndef DREAMPLACE_MACRO_LEGALIZE_LP_LEGALIZE_H
#define DREAMPLACE_MACRO_LEGALIZE_LP_LEGALIZE_H

#include <vector>
#include <limbo/solvers/DualMinCostFlow.h>

DREAMPLACE_BEGIN_NAMESPACE

/// @brief A linear programming (LP) based algorithm to legalize macros. 
/// It assumes the relative order of macros are determined. 
/// By constructing the horizontal and vertical constraint graph, 
/// an optimization problem is formulated to minimize the total displacement. 
/// The LP problem can be solved by dual min-cost flow algorithm. 
/// 
/// If the input macro solution is not legal, there is no guarantee to find a legal solution. 
/// But if it is legal, the output should still be legal. 
template <typename T>
void lpLegalizeLauncher(LegalizationDB<T> db, std::vector<int>& macros)
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
        for (int j = db.num_movable_nodes; j < db.num_nodes; ++j)
        {
            int node_id2 = j; 
            T xl2 = db.init_x[node_id2];
            T yl2 = db.init_y[node_id2];
            T width2 = db.node_size_x[node_id2];
            T height2 = db.node_size_y[node_id2];

            process2Nodes(i, xl1, yl1, width1, height1, j, xl2, yl2, width2, height2);
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

DREAMPLACE_END_NAMESPACE

#endif
