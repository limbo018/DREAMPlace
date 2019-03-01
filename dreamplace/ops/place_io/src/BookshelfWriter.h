/*************************************************************************
    > File Name: BookshelfWriter.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 20 Jul 2015 11:36:16 AM CDT
 ************************************************************************/

#ifndef GPF_BOOKSHELFWRITER_H
#define GPF_BOOKSHELFWRITER_H

#include <cstdio>
#include <vector>
#include "PlaceWriter.h"

GPF_BEGIN_NAMESPACE

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
        bool write(std::string const& outFile) const;
        /// write all files in book shelf format 
        /// \param outFile is aux file name 
        /// \param first, last should contain components to write 
        bool writeAll(std::string const& outFile, std::string const& designName) const;

    protected:
        bool writeAux(std::string const& outFileNoSuffix, std::string const& designName) const;
        bool writeNodes(std::string const& outFileNoSuffix) const;
        bool writeNets(std::string const& outFileNoSuffix) const;
        bool writeWts(std::string const& outFileNoSuffix) const;
        bool writeScl(std::string const& outFileNoSuffix) const;
        bool writeShapes(std::string const& outFileNoSuffix) const;
        bool writePlx(std::string const& outFileNoSuffix) const;
        bool writeRoute(std::string const& outFileNoSuffix) const; 
        void writeHeader(FILE* os, std::string const& fileType) const;
        FILE* openFile(std::string const& outFileNoSuffix, std::string const& fileType) const;
        void closeFile(FILE* os) const;
};

GPF_END_NAMESPACE

#endif
