/*************************************************************************
    > File Name: Enums.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon Jun 15 21:43:53 2015
 ************************************************************************/

#ifndef DREAMPLACE_ENUMS_H
#define DREAMPLACE_ENUMS_H

#include <string>
#include <map>
#include <ostream>
#include "Util.h"

DREAMPLACE_BEGIN_NAMESPACE

/// base class for enumeration types 
/// these types are not recommended for storage, since they takes larger memory 
template <typename EnumType>
class EnumExt
{
	public:
        typedef EnumType enum_type;
		EnumExt() {}
		EnumExt& operator=(EnumExt const& rhs)
		{
			if (this != &rhs)
				m_value = rhs.m_value;
			return *this;
		}
		EnumExt& operator=(enum_type const& rhs)
		{
			m_value = rhs;
			return *this;
		}
		EnumExt& operator=(std::string const& rhs)
        {
            m_value = str2Enum(rhs);
            return *this;
        }
		virtual operator std::string() const
        {
            return enum2Str(m_value);
        }
        operator int() const 
        {
            return value(); 
        }
        enum_type value() const 
        {
            return m_value;
        }

		bool operator==(EnumExt const& rhs) const {return m_value == rhs.m_value;}
		bool operator==(enum_type const& rhs) const {return m_value == rhs;}
		bool operator==(std::string const& rhs) const {return *this == EnumExt(rhs);}
		bool operator!=(EnumExt const& rhs) const {return m_value != rhs.m_value;}
		bool operator!=(enum_type const& rhs) const {return m_value != rhs;}
		bool operator!=(std::string const& rhs) const {return *this != EnumExt(rhs);}

		friend std::ostream& operator<<(std::ostream& os, const EnumExt& rhs)
		{
			rhs.print(os);
			return os;
		}
	protected:
		virtual void print(std::ostream& os) const {os << this->enum2Str(m_value);}

        virtual std::string enum2Str(enum_type const&) const = 0;
        virtual enum_type str2Enum(std::string const&) const = 0;

        enum_type m_value;
};

/// class Orient denotes orientation of cells 
struct OrientEnum ///< enum type protector 
{
    enum OrientType 
    {
        N = 0, 
        S = 1, 
        W = 2, 
        E = 3, 
        FN = 4, 
        FS = 5, 
        FW = 6, 
        FE = 7, 
        UNKNOWN = 8
    };
};
class Orient : public EnumExt<OrientEnum::OrientType>
{
	public:
        typedef OrientEnum enum_wrap_type;
        typedef OrientEnum::OrientType enum_type;
        typedef EnumExt<enum_type> base_type;

		Orient() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		Orient(Orient const& rhs) : base_type() {m_value = rhs.m_value;}
		Orient(enum_type const& rhs) : base_type() {m_value = rhs;}
		Orient(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		Orient& operator=(Orient const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		Orient& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		Orient& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		void hflip() 
		{
            // N => FN; S => FS; W => FW; E => FE
            // FN => N; FS => S; FW => W; FE => E
			if (m_value == enum_wrap_type::UNKNOWN) return;
			else if (m_value < 4) m_value = (enum_type)(m_value+4);
			else m_value = (enum_type)(m_value-4);
		}
		void rotate_180()
		{
            // N => S;   S => N;   W => E;   E => W
            // FN => FS; FS => FN; FW => FE; FE => FW
			if (m_value == enum_wrap_type::UNKNOWN) return;
            else m_value = (enum_type)(m_value^1); // simply flip LSB 
		}
		void vflip()
		{
			rotate_180();
			hflip();
		}

		static Orient hflip(Orient const& rhs) 
		{
			Orient tmp (rhs);
			tmp.hflip();
			return tmp;
		}
		static Orient rotate_180(Orient const& rhs) 
		{
			Orient tmp (rhs);
			tmp.rotate_180();
			return tmp;
		}
		static Orient vflip(Orient const& rhs) 
		{
			Orient tmp (rhs);
			tmp.vflip();
			return tmp;
		}
	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class PlaceStatus denotes placement status 
struct PlaceStatusEnum
{
    enum PlaceStatusType 
    {
        UNPLACED = 0x0, 
        PLACED = 0x1, 
        FIXED = 0x2, 
        DUMMY_FIXED = 0x3, ///< initially not fixed, but fixed by the placer 
        UNKNOWN = 0x4 
    };
};
class PlaceStatus : public EnumExt<PlaceStatusEnum::PlaceStatusType>
{
	public:
        typedef PlaceStatusEnum enum_wrap_type;
        typedef enum_wrap_type::PlaceStatusType enum_type;
        typedef EnumExt<enum_type> base_type;

		PlaceStatus() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		PlaceStatus(PlaceStatus const& rhs) : base_type() {m_value = rhs.m_value;}
		PlaceStatus(enum_type const& rhs) : base_type() {m_value = rhs;}
		PlaceStatus(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		PlaceStatus& operator=(PlaceStatus const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		PlaceStatus& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		PlaceStatus& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class MultiRowAttr denotes the multi-row attributes 
/// for those fixed cells, they may also be one of them 
struct MultiRowAttrEnum
{
    enum MultiRowAttrType 
    {
        SINGLE_ROW = 0x0, ///< single-row with any alignment
        MULTI_ROW_ANY = 0x1, ///< odd-rows with any alignment 
        MULTI_ROW_N = 0x2, ///< even-rows with N/FN alignment 
        MULTI_ROW_S = 0x3, ///< even-rows with S/FS alignment
        UNKNOWN = 0x4
    };
};
class MultiRowAttr : public EnumExt<MultiRowAttrEnum::MultiRowAttrType>
{
	public:
        typedef MultiRowAttrEnum enum_wrap_type;
        typedef enum_wrap_type::MultiRowAttrType enum_type;
        typedef EnumExt<enum_type> base_type;

		MultiRowAttr() : base_type() {m_value = enum_wrap_type::SINGLE_ROW;}
		MultiRowAttr(MultiRowAttr const& rhs) : base_type() {m_value = rhs.m_value;}
		MultiRowAttr(enum_type const& rhs) : base_type() {m_value = rhs;}
		MultiRowAttr(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		MultiRowAttr& operator=(MultiRowAttr const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		MultiRowAttr& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		MultiRowAttr& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class SignalDirect denotes direction of signal or pins 
struct SignalDirectEnum 
{
    enum SignalDirectType {INPUT, OUTPUT, INOUT, OUTPUT_TRISTATE, UNKNOWN};
};
class SignalDirect : public EnumExt<SignalDirectEnum::SignalDirectType>
{
	public:
        typedef SignalDirectEnum enum_wrap_type;
        typedef enum_wrap_type::SignalDirectType enum_type;
        typedef EnumExt<enum_type> base_type;

		SignalDirect() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		SignalDirect(SignalDirect const& rhs) : base_type() {m_value = rhs.m_value;}
		SignalDirect(enum_type const& rhs) : base_type() {m_value = rhs;}
		SignalDirect(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		SignalDirect& operator=(SignalDirect const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		SignalDirect& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		SignalDirect& operator=(std::string const& rhs)
		{
            m_value = str2Enum(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;

};

/// class PlanarDirect denotes geometric directions 
struct PlanarDirectEnum
{
    enum PlanarDirectType 
    {
        HORIZONTAL = 0, 
        VERTICAL = 1, 
        UNKNOWN = 2
    };
};
class PlanarDirect : public EnumExt<PlanarDirectEnum::PlanarDirectType>
{
	public:
        typedef PlanarDirectEnum enum_wrap_type;
        typedef enum_wrap_type::PlanarDirectType enum_type;
        typedef EnumExt<enum_type> base_type;

		PlanarDirect() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		PlanarDirect(PlanarDirect const& rhs) : base_type() {m_value = rhs.m_value;}
		PlanarDirect(enum_type const& rhs) : base_type() {m_value = rhs;}
		PlanarDirect(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		PlanarDirect& operator=(PlanarDirect const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		PlanarDirect& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		PlanarDirect& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class ReportFlag denotes various report flags 
struct ReportFlagEnum
{
    enum ReportFlagType 
    {
        NONE = 0, ///< report nothing  
        FIRST = 1, ///< report first 
        ABORT = 2, ///< report and abort 
        ALL = 4, ///< report all 
        TOTAL = 8, ///< report total number 
        COLLECT = 16, ///< collect all items 
        UNKNOWN = 32
    };
};
class ReportFlag : public EnumExt<ReportFlagEnum::ReportFlagType>
{
	public:
        typedef ReportFlagEnum enum_wrap_type;
        typedef enum_wrap_type::ReportFlagType enum_type;
        typedef EnumExt<enum_type> base_type;

		ReportFlag() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		ReportFlag(ReportFlag const& rhs) : base_type() {m_value = rhs.m_value;}
		ReportFlag(enum_type const& rhs) : base_type() {m_value = rhs;}
		ReportFlag(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		ReportFlag& operator=(ReportFlag const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		ReportFlag& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		ReportFlag& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class RowPlaceSolverType denotes the solver type for row placement 
struct RowPlaceSolverEnum
{
    enum RowPlaceSolverType 
    {
        LP_WL = 0, ///< linear programming with wirelength as objective 
        LP_DISP = 1, ///< linear programming with displacement as objective 
        MCF_WL = 2, ///< min-cost flow with wirelength as objective 
        MCF_DISP = 3, ///< min-cost flow with displacement as objective
        DP_WL = 4, ///< dynamic programming with wirelength as objective 
        DP_DISP = 5, ///< dynamic programming with displacement as objective
        DP_WL_PRUNE = 6, ///< DP_WL with pruning 
        DP_DISP_PRUNE = 7, ///< DP_DISP with pruning 
        UNKNOWN = 8
    };
};
class RowPlaceSolver : public EnumExt<RowPlaceSolverEnum::RowPlaceSolverType>
{
	public:
        typedef RowPlaceSolverEnum enum_wrap_type;
        typedef enum_wrap_type::RowPlaceSolverType enum_type;
        typedef EnumExt<enum_type> base_type;

		RowPlaceSolver() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		RowPlaceSolver(RowPlaceSolver const& rhs) : base_type() {m_value = rhs.m_value;}
		RowPlaceSolver(enum_type const& rhs) : base_type() {m_value = rhs;}
		RowPlaceSolver(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		RowPlaceSolver& operator=(RowPlaceSolver const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		RowPlaceSolver& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		RowPlaceSolver& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class MinCostFlowSolverType denotes the solver type for row placement 
struct MinCostFlowSolverEnum
{
    enum MinCostFlowSolverType
    {
        NETWORK_SIMPLEX = 0, ///< network simplex 
        COST_SCALING = 1, ///< cost scaling 
        CAPACITY_SCALING = 2, ///< capacity scaling which generalizes successive shortest path 
        CYCLE_CANCELING = 3, ///< cycle canceling 
        UNKNOWN = 4 
    };
};
class MinCostFlowSolver : public EnumExt<MinCostFlowSolverEnum::MinCostFlowSolverType>
{
	public:
        typedef MinCostFlowSolverEnum enum_wrap_type;
        typedef enum_wrap_type::MinCostFlowSolverType enum_type;
        typedef EnumExt<enum_type> base_type;

		MinCostFlowSolver() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		MinCostFlowSolver(MinCostFlowSolver const& rhs) : base_type() {m_value = rhs.m_value;}
		MinCostFlowSolver(enum_type const& rhs) : base_type() {m_value = rhs;}
		MinCostFlowSolver(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		MinCostFlowSolver& operator=(MinCostFlowSolver const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		MinCostFlowSolver& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		MinCostFlowSolver& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class RegionAssignSolverType denotes the solver type for region assignment 
struct RegionAssignSolverEnum
{
    enum RegionAssignSolverType 
    {
        LPLR = 0, ///< formulate as linear programming and solve with lagrangian relaxation 
        ILP = 1, ///< formulate as integer linear programming and solve with GUROBI 
        UNKNOWN = 2
    };
};
class RegionAssignSolver : public EnumExt<RegionAssignSolverEnum::RegionAssignSolverType>
{
	public:
        typedef RegionAssignSolverEnum enum_wrap_type;
        typedef enum_wrap_type::RegionAssignSolverType enum_type;
        typedef EnumExt<enum_type> base_type;

		RegionAssignSolver() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		RegionAssignSolver(RegionAssignSolver const& rhs) : base_type() {m_value = rhs.m_value;}
		RegionAssignSolver(enum_type const& rhs) : base_type() {m_value = rhs;}
		RegionAssignSolver(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		RegionAssignSolver& operator=(RegionAssignSolver const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		RegionAssignSolver& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		RegionAssignSolver& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class GlobalMoveEffortType denotes the effort for global move 
struct GlobalMoveEffortEnum 
{
    enum GlobalMoveEffortType
    {
        LEGALIZE = 1, ///< apply all movements to remove overlap, see findBestHPWLAndApply() 
        LEGALIZE_DISP = 2, ///< the only objective is displacement; apply all movements to remove overlap, see findBestHPWLAndApply() 
        WIRELENGTH = 4, ///< apply the best movements for wirelength improvement
        DENSITY = 8, ///< aims for density improvement 
        UNKNOWN = 16
    };
};
class GlobalMoveEffort : public EnumExt<GlobalMoveEffortEnum::GlobalMoveEffortType>
{
	public:
        typedef GlobalMoveEffortEnum enum_wrap_type;
        typedef enum_wrap_type::GlobalMoveEffortType enum_type;
        typedef EnumExt<enum_type> base_type;

		GlobalMoveEffort() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		GlobalMoveEffort(GlobalMoveEffort const& rhs) : base_type() {m_value = rhs.m_value;}
		GlobalMoveEffort(enum_type const& rhs) : base_type() {m_value = rhs;}
		GlobalMoveEffort(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		GlobalMoveEffort& operator=(GlobalMoveEffort const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		GlobalMoveEffort& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		GlobalMoveEffort& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class GlobalMoveAlgoType denotes the algorithm for global move 
struct GlobalMoveAlgoEnum 
{
    enum GlobalMoveAlgoType
    {
        GLOBALMOVE = 0, ///< global move 
        GLOBALMOVECHAIN = 1, ///< chain move 
        UNKNOWN = 2
    };
};
class GlobalMoveAlgo : public EnumExt<GlobalMoveAlgoEnum::GlobalMoveAlgoType>
{
	public:
        typedef GlobalMoveAlgoEnum enum_wrap_type;
        typedef enum_wrap_type::GlobalMoveAlgoType enum_type;
        typedef EnumExt<enum_type> base_type;

		GlobalMoveAlgo() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		GlobalMoveAlgo(GlobalMoveAlgo const& rhs) : base_type() {m_value = rhs.m_value;}
		GlobalMoveAlgo(enum_type const& rhs) : base_type() {m_value = rhs;}
		GlobalMoveAlgo(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		GlobalMoveAlgo& operator=(GlobalMoveAlgo const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		GlobalMoveAlgo& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		GlobalMoveAlgo& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

/// class RegionEnumType denotes the region type defined in DEF 
struct RegionTypeEnum
{
    enum RegionEnumType
    {
        FENCE = 0, 
        GUIDE = 1, 
        UNKNOWN = 2
    };
};
class RegionType : public EnumExt<RegionTypeEnum::RegionEnumType>
{
	public:
        typedef RegionTypeEnum enum_wrap_type;
        typedef enum_wrap_type::RegionEnumType enum_type;
        typedef EnumExt<enum_type> base_type;

		RegionType() : base_type() {m_value = enum_wrap_type::UNKNOWN;}
		RegionType(RegionType const& rhs) : base_type() {m_value = rhs.m_value;}
		RegionType(enum_type const& rhs) : base_type() {m_value = rhs;}
		RegionType(std::string const& rhs) : base_type() {m_value = str2Enum(rhs);}
		RegionType& operator=(RegionType const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		RegionType& operator=(enum_type const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}
		RegionType& operator=(std::string const& rhs)
		{
            this->base_type::operator=(rhs);
			return *this;
		}

	protected:
        virtual std::string enum2Str(enum_type const& e) const;
        virtual enum_type str2Enum(std::string const& s) const;
};

DREAMPLACE_END_NAMESPACE

#endif
