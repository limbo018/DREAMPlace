/*************************************************************************
    > File Name: BookshelfWriter.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 20 Jul 2015 11:36:16 AM CDT
 ************************************************************************/

#ifndef DREAMPLACE_BOOKSHELFWRITER_H
#define DREAMPLACE_BOOKSHELFWRITER_H

#include <cstdio>
#include <vector>
#include "PlaceWriter.h"

DREAMPLACE_BEGIN_NAMESPACE

class BookShelfWriter : public PlaceSolWriter
{
    public:
        typedef PlaceSolWriter base_type;
        typedef PlaceDB::index_type index_type;

        BookShelfWriter(PlaceDB const& db) : base_type(db) {}
        BookShelfWriter(BookShelfWriter const& rhs) : base_type(rhs) {}

        /// write .plx file  
        /// \param outFile is plx file name
        /// \param first, last should contain components to write 
        bool write(std::string const& outFile, 
                PlaceDB::coordinate_type const* x = NULL, PlaceDB::coordinate_type const* y = NULL) const;
        /// write all files in book shelf format 
        /// \param outFile is aux file name 
        /// \param first, last should contain components to write 
        bool writeAll(std::string const& outFile, std::string const& designName, 
                PlaceDB::coordinate_type const* x = NULL, PlaceDB::coordinate_type const* y = NULL) const;

    protected:
        bool writeAux(std::string const& outFileNoSuffix, std::string const& designName) const;
        bool writeNodes(std::string const& outFileNoSuffix) const;
        bool writeNets(std::string const& outFileNoSuffix) const;
        bool writeWts(std::string const& outFileNoSuffix) const;
        bool writeScl(std::string const& outFileNoSuffix) const;
        bool writeShapes(std::string const& outFileNoSuffix) const;
        bool writePlx(std::string const& outFileNoSuffix, 
                PlaceDB::coordinate_type const* x = NULL, PlaceDB::coordinate_type const* y = NULL) const;
        bool writeRoute(std::string const& outFileNoSuffix) const; 
        void writeHeader(FILE* os, std::string const& fileType) const;
        FILE* openFile(std::string const& outFileNoSuffix, std::string const& fileType) const;
        void closeFile(FILE* os) const;
};

DREAMPLACE_END_NAMESPACE

#endif
