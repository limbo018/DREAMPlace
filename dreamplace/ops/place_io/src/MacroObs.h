/*************************************************************************
    > File Name: MacroObs.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed 22 Jul 2015 11:26:26 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_MACROOBS_H
#define DREAMPLACE_MACROOBS_H

#include <string>
#include <vector>
#include <map>
#include "Box.h"

DREAMPLACE_BEGIN_NAMESPACE

class MacroObs : public Object
{
    public:
        typedef Object base_type;
        typedef base_type::coordinate_type coordinate_type;
        typedef Box<coordinate_type> box_type;
        typedef std::map<std::string, std::vector<box_type> > obs_map_type;
        typedef obs_map_type::iterator ObsIterator;
        typedef obs_map_type::const_iterator ObsConstIterator;

        /// default constructor 
        MacroObs();
        /// copy constructor
        MacroObs(MacroObs const& rhs);
        /// assignment
        MacroObs& operator=(MacroObs const& rhs);

        /// member functions
        obs_map_type const& obsMap() const {return m_mObs;}
        obs_map_type& obsMap() {return m_mObs;}
        MacroObs& add(std::string const& layerName, box_type const& box);
        template <typename Iterator>
        MacroObs& add(std::string const& layerName, Iterator first, Iterator last);

        bool empty() const {return m_mObs.empty();}
        ObsIterator begin() {return m_mObs.begin();}
        ObsIterator end() {return m_mObs.end();}
        ObsConstIterator begin() const {return m_mObs.begin();}
        ObsConstIterator end() const {return m_mObs.end();}

    protected:
        void copy(MacroObs const& rhs);

        obs_map_type m_mObs; ///< map of obstructions, layer -> rectangles
};

inline MacroObs::MacroObs() 
    : MacroObs::base_type()
    , m_mObs()
{
}
inline MacroObs::MacroObs(MacroObs const& rhs)
    : MacroObs::base_type(rhs)
{
    copy(rhs);
}
inline MacroObs& MacroObs::operator=(MacroObs const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void MacroObs::copy(MacroObs const& rhs)
{
    m_mObs = rhs.m_mObs;
}
inline MacroObs& MacroObs::add(std::string const& layerName, MacroObs::box_type const& box)
{
    obs_map_type::iterator found = m_mObs.find(layerName);
    if (found == m_mObs.end())
        m_mObs.insert(std::make_pair(layerName, std::vector<box_type>(1, box)));
    else 
        found->second.push_back(box);
    return *this;
}
template <typename Iterator>
inline MacroObs& MacroObs::add(std::string const& layerName, Iterator first, Iterator last)
{
    obs_map_type::iterator found = m_mObs.find(layerName);
    if (found == m_mObs.end())
        m_mObs.insert(std::make_pair(layerName, std::vector<box_type>(first, last)));
    else 
        found->second.insert(found->second.end(), first, last);
    return *this;
}

typedef MacroObs::ObsIterator ObsIterator;
typedef MacroObs::ObsConstIterator ObsConstIterator;

DREAMPLACE_END_NAMESPACE

#endif
