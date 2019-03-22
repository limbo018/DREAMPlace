/*************************************************************************
    > File Name: Macro.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon Jun 15 22:57:05 2015
 ************************************************************************/

#ifndef DREAMPLACE_MACRO_H
#define DREAMPLACE_MACRO_H

#include <string>
#include <vector>
#include <map>
#include "MacroPin.h"
#include "MacroObs.h"

DREAMPLACE_BEGIN_NAMESPACE

class Macro : public Box<Object::coordinate_type>, public Object
{
    public:
        typedef Object base_type2;
        typedef base_type2::coordinate_type coordinate_type;
        typedef Box<coordinate_type> base_type1;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef Point<coordinate_type> point_type;
        typedef std::map<std::string, index_type> string2index_map_type;

        /// default constructor 
        Macro();
        /// copy constructor
        Macro(Macro const& rhs);
        /// assignment
        Macro& operator=(Macro const& rhs);

        /// member functions
        std::string const& name() const {return m_name;}
        Macro& setName(std::string const& s) {m_name = s; return *this;}

        std::string const& className() const {return m_className;}
        Macro& setClassName(std::string const& s) {m_className = s; return *this;}

        std::string const& siteName() const {return m_siteName;}
        Macro& setSiteName(std::string const& s) {m_siteName = s; return *this;}

        std::string const& edgeName(Direction1DType d) const {return m_edgeName[d];}
        Macro& setEdgeName(Direction1DType d, std::string const& s) {m_edgeName[d] = s; return *this;}
        Macro& setEdgeName(std::string const& sl, std::string const& sr) {m_edgeName[kLEFT] = sl; m_edgeName[kRIGHT] = sr; return *this;}

        unsigned char symmetry() const {return m_symmetry;}
        Macro& setSymmetry(unsigned char s) {m_symmetry = s; return *this;}

        point_type const& initOrigin() const {return m_initOrigin;}
        Macro& setInitOrigin(point_type const& p) {m_initOrigin = p; return *this;}
        Macro& setInitOrigin(coordinate_type x, coordinate_type y) {m_initOrigin.set(x, y); return *this;}

        MacroObs const& obs() const {return m_obs;}
        MacroObs& obs() {return m_obs;}

        std::vector<MacroPin> const& macroPins() const {return m_vMacroPin;}
        std::vector<MacroPin>& macroPins() {return m_vMacroPin;}

        string2index_map_type const& macroPinName2Index() const {return m_mMacroPinName2Index;}
        string2index_map_type& macroPinName2Index() {return m_mMacroPinName2Index;}

        /// \return macro pin with given index 
        MacroPin const& macroPin(index_type id) const {return m_vMacroPin.at(id);}
        MacroPin& macroPin(index_type id) {return m_vMacroPin.at(id);}

        /// \return macro pin index with given name 
        index_type macroPinIndex(std::string const& s) const 
        {
            string2index_map_type::const_iterator found = m_mMacroPinName2Index.find(s);
            return (found != m_mMacroPinName2Index.end())? found->second : std::numeric_limits<index_type>::max();
        }

        /// add a macro pin to m_vMacroPin and insert its index to m_mMacroPinName2Index
        /// \return true if succeed, false if already exists 
        /// the first value of return pair is the index in m_vMacroPin
        /// \param n denotes name of the pin 
        std::pair<Macro::index_type, bool> addMacroPin(std::string const& n);
    protected:
        void copy(Macro const& rhs);

        std::string m_name; ///< macro name 
        std::string m_className; ///< class name, usually not useful
        std::string m_siteName; ///< site name, usually not useful
        std::string m_edgeName[2]; ///< edge name (left/right) 
        unsigned char m_symmetry; ///< 3-bit: x, y, R90
        point_type m_initOrigin; ///< initial origin in LEF file, the actual origins are adjusted to (0, 0) after read-in

        MacroObs m_obs; ///< obstructions

        std::vector<MacroPin> m_vMacroPin; ///< standard cell pins 
        string2index_map_type m_mMacroPinName2Index; ///< map names of standard cell pins to index 
};

inline Macro::Macro() 
    : Macro::base_type1()
    , Macro::base_type2()
    , m_name("")
    , m_className("")
    , m_siteName("")
    , m_edgeName()
    , m_symmetry(std::numeric_limits<unsigned char>::max())
    , m_initOrigin()
    , m_obs()
    , m_vMacroPin()
    , m_mMacroPinName2Index()
{
}
inline Macro::Macro(Macro const& rhs)
    : Macro::base_type1(rhs)
    , Macro::base_type2(rhs)
{
    copy(rhs);
}
inline Macro& Macro::operator=(Macro const& rhs)
{
    if (this != &rhs)
    {
        this->base_type1::operator=(rhs);
        this->base_type2::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void Macro::copy(Macro const& rhs)
{
    m_name = rhs.m_name;
    m_className = rhs.m_className;
    m_siteName = rhs.m_siteName;
    m_edgeName[0] = rhs.m_edgeName[0];
    m_edgeName[1] = rhs.m_edgeName[1];
    m_symmetry = rhs.m_symmetry;
    m_initOrigin = rhs.m_initOrigin;
    m_obs = rhs.m_obs;
    m_vMacroPin = rhs.m_vMacroPin;
    m_mMacroPinName2Index = rhs.m_mMacroPinName2Index;
}
inline std::pair<Macro::index_type, bool> Macro::addMacroPin(std::string const& n)
{
    string2index_map_type::iterator found = m_mMacroPinName2Index.find(n);
    if (found != m_mMacroPinName2Index.end()) // already exist 
        return std::make_pair(found->second, false);
    else // not exist, create macro pin 
    {
        m_vMacroPin.push_back(MacroPin());
        MacroPin& mp = m_vMacroPin.back();
        mp.setName(n);
        mp.setId(m_vMacroPin.size()-1);
        std::pair<string2index_map_type::iterator, bool> insertRet = m_mMacroPinName2Index.insert(std::make_pair(mp.name(), mp.id()));
        dreamplaceAssertMsg(insertRet.second, "failed to insert macro pin (%s, %d).(%s, %d)", name().c_str(), id(), mp.name().c_str(), mp.id());
        return std::make_pair(mp.id(), true);
    }
}

DREAMPLACE_END_NAMESPACE

#endif
