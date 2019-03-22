/*************************************************************************
    > File Name: PlaceWriter.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 20 Jul 2015 11:34:51 AM CDT
 ************************************************************************/

#ifndef DREAMPLACE_PLACEWRITER_H
#define DREAMPLACE_PLACEWRITER_H

#include <cstdio>
#include <vector>
#include "PlaceDB.h"

DREAMPLACE_BEGIN_NAMESPACE

/// class PlaceSolWriter is the base class to write placement solutions 
class PlaceSolWriter
{
    public:
        PlaceSolWriter(PlaceDB const& db) : m_db(db) {}
        PlaceSolWriter(PlaceSolWriter const& rhs) : m_db(rhs.m_db) {}
        PlaceSolWriter& operator=(PlaceSolWriter const& rhs); 
    protected:
        PlaceDB const& m_db;
};

DREAMPLACE_END_NAMESPACE

#endif
