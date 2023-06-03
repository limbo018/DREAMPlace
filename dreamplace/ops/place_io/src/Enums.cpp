/*************************************************************************
    > File Name: Enums.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 03 Aug 2015 10:50:21 AM CDT
 ************************************************************************/

#include "Enums.h"
#include <typeinfo>

DREAMPLACE_BEGIN_NAMESPACE

#ifndef ENUM2STR
#define ENUM2STR(map, var) \
    map[enum_wrap_type::var] = #var
#endif

#ifndef STR2ENUM
#define STR2ENUM(map, var) \
    map[#var] = enum_wrap_type::var
#endif

std::string Orient::enum2Str(enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, N);
        ENUM2STR(mEnum2Str, S);
        ENUM2STR(mEnum2Str, W);
        ENUM2STR(mEnum2Str, E);
        ENUM2STR(mEnum2Str, FN);
        ENUM2STR(mEnum2Str, FS);
        ENUM2STR(mEnum2Str, FW);
        ENUM2STR(mEnum2Str, FE);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

Orient::enum_type Orient::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, N);
        STR2ENUM(mStr2Enum, S);
        STR2ENUM(mStr2Enum, W);
        STR2ENUM(mStr2Enum, E);
        STR2ENUM(mStr2Enum, FN);
        STR2ENUM(mStr2Enum, FS);
        STR2ENUM(mStr2Enum, FW);
        STR2ENUM(mStr2Enum, FE);
        STR2ENUM(mStr2Enum, UNKNOWN);

        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string PlaceStatus::enum2Str(PlaceStatus::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, UNPLACED);
        ENUM2STR(mEnum2Str, PLACED);
        ENUM2STR(mEnum2Str, FIXED);
        ENUM2STR(mEnum2Str, DUMMY_FIXED);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

PlaceStatus::enum_type PlaceStatus::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, UNPLACED);
        STR2ENUM(mStr2Enum, PLACED);
        STR2ENUM(mStr2Enum, FIXED);
        STR2ENUM(mStr2Enum, DUMMY_FIXED);
        STR2ENUM(mStr2Enum, UNKNOWN);

        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string MultiRowAttr::enum2Str(MultiRowAttr::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, SINGLE_ROW);
        ENUM2STR(mEnum2Str, MULTI_ROW_ANY);
        ENUM2STR(mEnum2Str, MULTI_ROW_N);
        ENUM2STR(mEnum2Str, MULTI_ROW_S);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

MultiRowAttr::enum_type MultiRowAttr::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, SINGLE_ROW);
        STR2ENUM(mStr2Enum, MULTI_ROW_ANY);
        STR2ENUM(mStr2Enum, MULTI_ROW_N);
        STR2ENUM(mStr2Enum, MULTI_ROW_S);
        STR2ENUM(mStr2Enum, UNKNOWN);

        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string SignalDirect::enum2Str(SignalDirect::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, INPUT);
        ENUM2STR(mEnum2Str, OUTPUT);
        ENUM2STR(mEnum2Str, INOUT);
        ENUM2STR(mEnum2Str, OUTPUT_TRISTATE); ///< observed from Mentor Graphics benchmarks
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

SignalDirect::enum_type SignalDirect::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, INPUT);
        STR2ENUM(mStr2Enum, OUTPUT);
        STR2ENUM(mStr2Enum, INOUT);
        STR2ENUM(mStr2Enum, OUTPUT_TRISTATE); ///< observed from Mentor Graphics benchmarks
        STR2ENUM(mStr2Enum, UNKNOWN);

        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string PlanarDirect::enum2Str(PlanarDirect::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, HORIZONTAL);
        ENUM2STR(mEnum2Str, VERTICAL);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

PlanarDirect::enum_type PlanarDirect::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, HORIZONTAL);
        STR2ENUM(mStr2Enum, VERTICAL);
        STR2ENUM(mStr2Enum, UNKNOWN);

        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string ReportFlag::enum2Str(ReportFlag::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, NONE);
        ENUM2STR(mEnum2Str, FIRST);
        ENUM2STR(mEnum2Str, ABORT);
        ENUM2STR(mEnum2Str, ALL);
        ENUM2STR(mEnum2Str, TOTAL);
        ENUM2STR(mEnum2Str, COLLECT);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

ReportFlag::enum_type ReportFlag::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, NONE);
        STR2ENUM(mStr2Enum, FIRST);
        STR2ENUM(mStr2Enum, ABORT);
        STR2ENUM(mStr2Enum, ALL);
        STR2ENUM(mStr2Enum, TOTAL);
        STR2ENUM(mStr2Enum, COLLECT);
        STR2ENUM(mStr2Enum, UNKNOWN);

        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string RowPlaceSolver::enum2Str(RowPlaceSolver::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, LP_WL);
        ENUM2STR(mEnum2Str, LP_DISP);
        ENUM2STR(mEnum2Str, MCF_WL);
        ENUM2STR(mEnum2Str, MCF_DISP);
        ENUM2STR(mEnum2Str, DP_WL);
        ENUM2STR(mEnum2Str, DP_DISP);
        ENUM2STR(mEnum2Str, DP_WL_PRUNE);
        ENUM2STR(mEnum2Str, DP_DISP_PRUNE);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

RowPlaceSolver::enum_type RowPlaceSolver::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, LP_WL);
        STR2ENUM(mStr2Enum, LP_DISP);
        STR2ENUM(mStr2Enum, MCF_WL);
        STR2ENUM(mStr2Enum, MCF_DISP);
        STR2ENUM(mStr2Enum, DP_WL);
        STR2ENUM(mStr2Enum, DP_DISP);
        STR2ENUM(mStr2Enum, DP_WL_PRUNE);
        STR2ENUM(mStr2Enum, DP_DISP_PRUNE);
        STR2ENUM(mStr2Enum, UNKNOWN);
        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string MinCostFlowSolver::enum2Str(MinCostFlowSolver::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, NETWORK_SIMPLEX);
        ENUM2STR(mEnum2Str, COST_SCALING);
        ENUM2STR(mEnum2Str, CAPACITY_SCALING);
        ENUM2STR(mEnum2Str, CYCLE_CANCELING);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

MinCostFlowSolver::enum_type MinCostFlowSolver::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, NETWORK_SIMPLEX);
        STR2ENUM(mStr2Enum, COST_SCALING);
        STR2ENUM(mStr2Enum, CAPACITY_SCALING);
        STR2ENUM(mStr2Enum, CYCLE_CANCELING);
        STR2ENUM(mStr2Enum, UNKNOWN);
        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string RegionAssignSolver::enum2Str(RegionAssignSolver::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, LPLR);
        ENUM2STR(mEnum2Str, ILP);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

RegionAssignSolver::enum_type RegionAssignSolver::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, LPLR);
        STR2ENUM(mStr2Enum, ILP);
        STR2ENUM(mStr2Enum, UNKNOWN);
        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string GlobalMoveEffort::enum2Str(GlobalMoveEffort::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, LEGALIZE);
        ENUM2STR(mEnum2Str, LEGALIZE_DISP);
        ENUM2STR(mEnum2Str, WIRELENGTH);
        ENUM2STR(mEnum2Str, DENSITY);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

GlobalMoveEffort::enum_type GlobalMoveEffort::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, LEGALIZE);
        STR2ENUM(mStr2Enum, LEGALIZE_DISP);
        STR2ENUM(mStr2Enum, WIRELENGTH);
        STR2ENUM(mStr2Enum, DENSITY);
        STR2ENUM(mStr2Enum, UNKNOWN);
        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string GlobalMoveAlgo::enum2Str(GlobalMoveAlgo::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, GLOBALMOVE);
        ENUM2STR(mEnum2Str, GLOBALMOVECHAIN);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

GlobalMoveAlgo::enum_type GlobalMoveAlgo::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, GLOBALMOVE);
        STR2ENUM(mStr2Enum, GLOBALMOVECHAIN);
        STR2ENUM(mStr2Enum, UNKNOWN);
        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}

std::string RegionType::enum2Str(RegionType::enum_type const& e) const
{
    static std::map<enum_type, std::string> mEnum2Str;
    static bool init = true;

    if (init)
    {
        ENUM2STR(mEnum2Str, FENCE);
        ENUM2STR(mEnum2Str, GUIDE);
        ENUM2STR(mEnum2Str, UNKNOWN);
        init = false;
    }

    return mEnum2Str.at(e);
}

RegionType::enum_type RegionType::str2Enum(std::string const& s) const
{
    static std::map<std::string, enum_type> mStr2Enum;
    static bool init = true;

    if (init)
    {
        STR2ENUM(mStr2Enum, FENCE);
        STR2ENUM(mStr2Enum, GUIDE);
        STR2ENUM(mStr2Enum, UNKNOWN);
        init = false;
    }

    std::map<std::string, enum_type>::const_iterator found = mStr2Enum.find(s);
    if (found == mStr2Enum.end())
    {
        dreamplacePrint(kWARN, "%s::%s unknown enum type %s, set to UNKNOWN\n", typeid(*this).name(), __func__, s.c_str());
        return enum_wrap_type::UNKNOWN; 
    }
    else 
    {
        return found->second;
    }
}
DREAMPLACE_END_NAMESPACE
