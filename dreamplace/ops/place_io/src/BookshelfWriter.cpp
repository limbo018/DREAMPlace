/*************************************************************************
    > File Name: BookshelfWriter.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 20 Jul 2015 12:12:54 PM CDT
 ************************************************************************/

#include "BookshelfWriter.h"
#include "Iterators.h"
#include "PlaceDB.h"
#include <cstdio>
#include <limbo/string/String.h>

DREAMPLACE_BEGIN_NAMESPACE

bool BookShelfWriter::write(std::string const& outFile, 
                PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const
{
    std::string outFileNoSuffix = limbo::trim_file_suffix(outFile);
    return writePlx(outFileNoSuffix, x, y);
}
bool BookShelfWriter::writeAll(std::string const& outFile, std::string const& designName, 
                PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const
{
    std::string outFileNoSuffix = limbo::trim_file_suffix(outFile);

    bool flag = writeAux(outFileNoSuffix, designName);
    if (flag)
        flag = writeNodes(outFileNoSuffix);
    if (flag)
        flag = writeNets(outFileNoSuffix);
    if (flag)
        flag = writeWts(outFileNoSuffix);
    if (flag) 
        flag = writeScl(outFileNoSuffix);
    if (flag)
        flag = writeShapes(outFileNoSuffix);
    if (flag)
        flag = writePlx(outFileNoSuffix, x, y);
    if (flag)
        flag = writeRoute(outFileNoSuffix); 

    return true;
}

bool BookShelfWriter::writeAux(std::string const& outFileNoSuffix, std::string const& /*designName*/) const
{
    FILE* out = openFile(outFileNoSuffix, "aux");
    if (out == NULL) 
        return false;

    std::string filename = limbo::get_file_name(outFileNoSuffix); // remove path 
    fprintf(out, "%s : %s.nodes %s.nets %s.wts %s.pl %s.scl %s.shapes %s.route", 
            //designName.c_str(), 
            "RowBasedPlacement", // must be this name due to the evaluation script 
            filename.c_str(), filename.c_str(), filename.c_str(), 
            filename.c_str(), filename.c_str(), filename.c_str(), 
            filename.c_str());

    closeFile(out);
    return true;
}
bool BookShelfWriter::writeNodes(std::string const& outFileNoSuffix) const
{
    FILE* out = openFile(outFileNoSuffix, "nodes");
    if (out == NULL) 
        return false;

    writeHeader(out, "nodes");

    std::vector<Node> const& vNode = m_db.nodes();
    // write total number of nodes 
    fprintf(out, "NumNodes : %lu\n", vNode.size());
    // write number of terminals 
#if 0
    // complex version 
    fprintf(out, "NumTerminals : %lu num macros: %lu num pads: %lu num fixed insts: %lu num multi row height insts(not fixed): %lu\n", 
            m_db.numFixed()+m_db.numIOPin(), 
            m_db.numIOPin(), 
            (std::size_t)0, 
            m_db.numFixed(), 
            (std::size_t)0);
#endif
    fprintf(out, "NumTerminals : %lu\n", m_db.numFixed()+m_db.numIOPin());
    fprintf(out, "\n");

    // write movable nodes 
    for (MovableNodeConstIterator it = m_db.movableNodeBegin(); it.inRange(); ++it)
        fprintf(out, "%s %ld %ld\n", m_db.nodeName(*it).c_str(), it->width(), it->height());
    // write io pins 
    for (IOPinNodeConstIterator it = m_db.iopinNodeBegin(); it.inRange(); ++it)
        fprintf(out, "%s %ld %ld terminal_NI\n", m_db.nodeName(*it).c_str(), it->width(), it->height());
    // write fixed nodes 
    for (FixedNodeConstIterator it = m_db.fixedNodeBegin(); it.inRange(); ++it)
        fprintf(out, "%s %ld %ld terminal\n", m_db.nodeName(*it).c_str(), it->width(), it->height());

    closeFile(out);
    return true;
}
bool BookShelfWriter::writeNets(std::string const& outFileNoSuffix) const
{
    FILE* out = openFile(outFileNoSuffix, "nets");
    if (out == NULL)
        return false;

    writeHeader(out, "nets");

    std::vector<Net> const& vNet = m_db.nets();
    std::vector<Pin> const& vPin = m_db.pins();

    // write total number of nets 
    fprintf(out, "NumNets : %lu\n", vNet.size());
    // write total number of pins 
    fprintf(out, "NumPins : %lu\n", vPin.size());
    fprintf(out, "\n");
    // write nets 
    for (std::vector<Net>::const_iterator it = vNet.begin(), ite = vNet.end(); it != ite; ++it)
    {
        std::vector<index_type> const& vNetPin = it->pins();
        fprintf(out, "NetDegree : %lu %s\n", vNetPin.size(), m_db.netName(*it).c_str());
        for (std::vector<index_type>::const_iterator itp = vNetPin.begin(), itpe = vNetPin.end(); itp != itpe; ++itp)
        {
            Pin const& pin = vPin[*itp];
            Node const& node = m_db.getNode(pin); 
            // we need the offset that corresponds to orientation N
            Point<PlaceDB::coordinate_type> pinOffset = m_db.getNodePinOffset(pin, node.orient(), OrientEnum::N);
            Point<PlaceDB::coordinate_type> pinPos = ll(node)+pinOffset;
            fprintf(out, "    %s %s : %d %d\n", 
                    m_db.nodeName(m_db.getNode(pin)).c_str(), 
                    ((pin.direct() == SignalDirectEnum::INPUT)? "I" : "O"), 
                    pinPos.x()-center(node, kX), // pin position with respect to center 
                    pinPos.y()-center(node, kY) // pin position with respect to center 
                    );
        }
    }

    closeFile(out);
    return true;
}
bool BookShelfWriter::writeWts(std::string const& outFileNoSuffix) const
{
    FILE* out = openFile(outFileNoSuffix, "wts");
    if (out == NULL)
        return false;

    writeHeader(out, "wts");

    closeFile(out);
    return true;
}
bool BookShelfWriter::writeScl(std::string const& outFileNoSuffix) const
{
    FILE* out = openFile(outFileNoSuffix, "scl");
    if (out == NULL)
        return false;

    writeHeader(out, "scl");

    std::vector<Row> const& vRow = m_db.rows();
#if 0 // only in complex version 
    // write site width and row height 
    fprintf(out, "Default_Site_Width : %d\n", m_db.siteWidth());
    fprintf(out, "Default_Row_Height : %d\n", m_db.rowHeight());
#endif
    // write total number of rows 
    fprintf(out, "NumRows : %lu\n", vRow.size());
    fprintf(out, "\n");
    // write rows 
    for (std::vector<Row>::const_iterator it = vRow.begin(), ite = vRow.end(); it != ite; ++it)
    {
        //fprintf(out, "%s Horizontal\n", it->macroName().c_str());
        fprintf(out, "%s Horizontal\n", "CoreRow");
        fprintf(out, "\tCoordinate : %d\n", it->yl());
        fprintf(out, "\tHeight : %ld\n", it->height());
        fprintf(out, "\tSitewidth : %d\n", it->step(kX));
        fprintf(out, "\tSitespacing : %d\n", it->step(kX)); // I do not know what is this for, simply set the same as Sitewidth
        fprintf(out, "\tSiteorient : %d\n", ((it->orient() == OrientEnum::N)? 0 : 1));
        fprintf(out, "\tSitesymmetry : %d\n", 1); 
        fprintf(out, "\tSubrowOrigin : %d NumSites : %d\n", it->xl(), it->numSites(kX));
        fprintf(out, "End\n");
    }

    closeFile(out);
    return true;
}
bool BookShelfWriter::writeShapes(std::string const& outFileNoSuffix) const
{
    FILE* out = openFile(outFileNoSuffix, "shapes");
    if (out == NULL)
        return false;

    writeHeader(out, "shapes");

    fprintf(out, "NumNonRectangularNodes : %d\n", 0);

    closeFile(out);
    return true;
}
bool BookShelfWriter::writePlx(std::string const& outFileNoSuffix, 
                PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const 
{
    FILE* out = openFile(outFileNoSuffix, "pl");
    if (out == NULL)
        return false;

    writeHeader(out, "pl"); // use pl instead of plx to accommodate parser

    std::vector<Node> const& vNode = m_db.nodes();
    for (std::vector<Node>::const_iterator it = vNode.begin(), ite = vNode.end(); it != ite; ++it)
    {
        Node const& node = *it; 

        PlaceDB::coordinate_type xx = node.xl(); 
        PlaceDB::coordinate_type yy = node.yl(); 
        if (node.id() < m_db.numMovable())
        {
            if (x)
            {
                xx = x[node.id()];
            }
            if (y)
            {
                yy = y[node.id()];
            }
        }

        fprintf(out, "%s %d %d : %s", m_db.nodeName(node).c_str(), xx, yy, std::string(Orient(node.orient())).c_str());
        if (node.id() < m_db.numMovable()+m_db.numFixed() && node.status() == PlaceStatusEnum::FIXED) // fixed instance
            fprintf(out, " /FIXED"); 
        else if (node.id() >= m_db.numMovable()+m_db.numFixed()) // io pins
            fprintf(out, " /FIXED_NI"); 
        fprintf(out, "\n"); 
    }

    closeFile(out);
    return true;
}
bool BookShelfWriter::writeRoute(std::string const& outFileNoSuffix) const 
{
    FILE* out = openFile(outFileNoSuffix, "route");
    if (out == NULL)
        return false;

    writeHeader(out, "route"); 

    int const tileSize[2] = {9, 9}; // tile size in terms of row height 
    int const numLayer = 6; 
    int const numTracks[2] = {1, 1}; // number of tracks per site width 
    int const minWireWidth[numLayer] = {1, 1, 1, 1, 2, 2}; 
    int const minWireSpacing[numLayer] = {1, 1, 1, 1, 2, 2}; 
    int const viaSpacing[numLayer] = {0, 0, 0, 0, 0, 0};
    int const blockagePorosity = 0; 
    int const numBlockedLayer = 4;
    int const blockedLayer[numBlockedLayer] = {1, 2, 3, 4}; 

    fprintf(out, "Grid : %d %d %d\n", (int)ceil((float)m_db.dieArea().width()/m_db.rowHeight()/tileSize[kX]), (int)ceil((float)m_db.dieArea().height()/m_db.rowHeight()/tileSize[kY]), numLayer); 
    fprintf(out, "VerticalCapacity : "); 
    for (int i = 0; i < numLayer; ++i)
    {
        if (i%2 == 1) // odd layer 
            fprintf(out, "%d ", numTracks[kX]*tileSize[kX]*m_db.rowHeight()/m_db.siteWidth()); 
        else 
            fprintf(out, "%d ", 0); 
    }
    fprintf(out, "\n"); 
    fprintf(out, "HorizontalCapacity : "); 
    for (int i = 0; i < numLayer; ++i)
    {
        if (i%2 == 0) // even layer 
            fprintf(out, "%d ", numTracks[kY]*tileSize[kY]*m_db.rowHeight()/m_db.siteWidth()); 
        else 
            fprintf(out, "%d ", 0); 
    }
    fprintf(out, "\n"); 
    fprintf(out, "MinWireWidth : "); 
    for (int i = 0; i < numLayer; ++i)
        fprintf(out, "%d ", minWireWidth[i]); 
    fprintf(out, "\n"); 
    fprintf(out, "MinWireSpacing : "); 
    for (int i = 0; i < numLayer; ++i)
        fprintf(out, "%d ", minWireSpacing[i]); 
    fprintf(out, "\n"); 
    fprintf(out, "ViaSpacing : "); 
    for (int i = 0; i < numLayer; ++i)
        fprintf(out, "%d ", viaSpacing[i]); 
    fprintf(out, "\n"); 
    fprintf(out, "GridOrigin : %d %d\n", m_db.dieArea().xl(), m_db.dieArea().yl()); 
    fprintf(out, "TileSize : %d %d\n", tileSize[kX]*m_db.rowHeight(), tileSize[kY]*m_db.rowHeight()); 
    fprintf(out, "BlockagePorosity : %d\n", blockagePorosity); 
    fprintf(out, "\n"); 

    // routing blockages from fixed instances 
    std::vector<Node> const& vNode = m_db.nodes();
    int numBlockages = 0; 
    for (std::vector<Node>::const_iterator it = vNode.begin(), ite = vNode.end(); it != ite; ++it)
    {
        Node const& node = *it; 
        if (node.status() == PlaceStatusEnum::FIXED || node.status() == PlaceStatusEnum::DUMMY_FIXED)
            ++numBlockages; 
    }
    fprintf(out, "NumBlockageNodes : %d\n", numBlockages); 
    fprintf(out, "\n"); 
    for (std::vector<Node>::const_iterator it = vNode.begin(), ite = vNode.end(); it != ite; ++it)
    {
        Node const& node = *it; 
        if (node.status() == PlaceStatusEnum::FIXED || node.status() == PlaceStatusEnum::DUMMY_FIXED)
        {
            fprintf(out, "%s : %d ", m_db.nodeName(node).c_str(), numBlockedLayer);
            for (int i = 0; i < numBlockedLayer; ++i)
                fprintf(out, "%d ", blockedLayer[i]); 
            fprintf(out, "\n"); 
        }
    }

    closeFile(out);
    return true;
}
void BookShelfWriter::writeHeader(FILE* os, std::string const& fileType) const
{
    if (fileType == "shapes")
        fprintf(os, "%s 1.0\n", fileType.c_str());
    else
        fprintf(os, "UCLA %s 1.0\n", fileType.c_str());
    fprintf(os, "\n");
}
FILE* BookShelfWriter::openFile(std::string const& outFileNoSuffix, std::string const& fileType) const
{
    dreamplacePrint(kINFO, "writing placement to %s\n", (outFileNoSuffix+"."+fileType).c_str());

    FILE* out = fopen((outFileNoSuffix+"."+fileType).c_str(), "w");
    if (out == NULL)
        dreamplacePrint(kERROR, "unable to open %s for write\n", (outFileNoSuffix+"."+fileType).c_str());
    return out;
}
void BookShelfWriter::closeFile(FILE* os) const 
{
    fclose(os);
}

DREAMPLACE_END_NAMESPACE
