/*************************************************************************
    > File Name: DefWriter.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 22 Jun 2015 08:11:03 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_DEFWRITER_H
#define DREAMPLACE_DEFWRITER_H

#include <cstdio>
#include <vector>
#include "PlaceWriter.h"

DREAMPLACE_BEGIN_NAMESPACE

class DefWriter : public PlaceSolWriter
{
    public:
        typedef PlaceSolWriter base_type;

        DefWriter(PlaceDB const& db) : base_type(db) {}
        DefWriter(DefWriter const& rhs) : base_type(rhs) {}

        /// write DEF file, by replacing the component block of the input DEF file 
        /// \param first, last should contain components to write 
        bool write(std::string const& outFile, std::string const& inFile, 
                std::vector<Node> const& vNode, std::vector<PlaceDB::index_type> const& vNodeIndex, 
                PlaceDB::coordinate_type const* x = NULL, PlaceDB::coordinate_type const* y = NULL) const;
        /// write simplified DEF file for iccad contest 
        /// \param first, last should contain components to write 
        bool writeSimple(std::string const& outFile, std::string const& version, std::string const& designName, 
                std::vector<Node> const& vNode, std::vector<PlaceDB::index_type> const& vNodeIndex, 
                PlaceDB::coordinate_type const* x = NULL, PlaceDB::coordinate_type const* y = NULL) const;

    protected:
        /// write components block 
        void writeCompBlock(FILE* os, std::vector<Node> const& vNode, std::vector<PlaceDB::index_type> const& vNodeIndex, 
                PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const;
        void writeComp(FILE* os, Node const& n, 
                PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const;
        /// trim leading whitespaces 
        std::string ltrim(std::string const& s) const; 
        /// trim tailing whitespaces 
        std::string rtrim(std::string const& s) const;
        /// trim leading and tailing whitespaces
        std::string trim(std::string const& s) const; 
};

DREAMPLACE_END_NAMESPACE

#endif
