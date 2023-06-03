/*************************************************************************
    > File Name: DefWriter.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 22 Jun 2015 08:18:29 PM CDT
 ************************************************************************/

#include "DefWriter.h"
#include <fstream>

DREAMPLACE_BEGIN_NAMESPACE

bool DefWriter::write(std::string const& outFile, std::string const& inFile, 
        std::vector<Node> const& vNode, std::vector<PlaceDB::index_type> const& vNodeIndex, 
        PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const 
{
    std::ifstream in (inFile.c_str());
    FILE* out = fopen(outFile.c_str(), "w");
    std::string line;
    std::string nodeName;
    std::size_t pos1, pos2;
    bool flag = false; // whether in COMPONENTS block 
    std::size_t rowCount = 0; 

    dreamplacePrint(kINFO, "writing placement to %s\n", outFile.c_str());

    if (!in.good())
    {
        dreamplacePrint(kERROR, "unable to open %s for read\n", inFile.c_str());
        return false;
    }
    if (out == NULL)
    {
        dreamplacePrint(kERROR, "unable to open %s for write\n", outFile.c_str());
        return false;
    }

    while (getline(in, line))
    {
        line = trim(line);
        pos1 = line.find("END");
        pos2 = line.find("COMPONENTS");
        if (pos1 != std::string::npos && 
                pos2 != std::string::npos) // match "END COMPONENTS"
        {
            // found "END COMPONENTS"
            // dump positions here 
            writeCompBlock(out, vNode, vNodeIndex, x, y);

            flag = false;
            continue;
        }
        else if (pos2 != std::string::npos) // match "COMPONENTS"
        {
            flag = true;
            continue;
        }

        if (flag) {/* skip everything in a COMPONENTS block */}
        else 
        {
            if (line.substr(0, 3) == "ROW") // match "ROW" entry, it does not hurt even if not matched; mainly for modification of benchmarks   
            {
                Row const& row = m_db.row(rowCount);
                fprintf(out, "ROW %s %s %d %d %s DO %u BY %u STEP %d %d ", 
                        row.name().c_str(), row.macroName().c_str(), 
                        row.xl(), row.yl(), std::string(row.orient()).c_str(), 
                        row.numSites(kX), (row.step(kY) == 0)? 1 : row.numSites(kY), row.step(kX), row.step(kY));
                if (line.back() == ';')
                {
                    fprintf(out, ";");
                }
                fprintf(out, "\n");
                ++rowCount;
            }
            else 
            {
                fprintf(out, "%s\n", line.c_str()); 
            }
        }
    }

    in.close();
    fclose(out);
    return true;
}
bool DefWriter::writeSimple(std::string const& outFile, std::string const& version, std::string const& designName, 
        std::vector<Node> const& vNode, std::vector<PlaceDB::index_type> const& vNodeIndex, 
        PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const 
{
    dreamplacePrint(kINFO, "writing placement to %s\n", outFile.c_str());

    FILE* out = fopen(outFile.c_str(), "w");
    if (out == NULL)
    {
        dreamplacePrint(kERROR, "failed to open %s for write\n", outFile.c_str());
        return false;
    }

    fprintf(out, "VERSION %s ;\n", version.c_str());
    fprintf(out, "DESIGN %s ;\n\n", designName.c_str());
    writeCompBlock(out, vNode, vNodeIndex, x, y);
    fprintf(out, "\nEND DESIGN");

    fclose(out);
    return true;
}
void DefWriter::writeCompBlock(FILE* os, std::vector<Node> const& vNode, std::vector<PlaceDB::index_type> const& vNodeIndex, 
                PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const 
{
    fprintf(os, "COMPONENTS %lu ;\n", vNodeIndex.size());
    for (auto node_id : vNodeIndex)
    {
        writeComp(os, vNode.at(node_id), x, y);
    }
    fprintf(os, "END COMPONENTS\n");
}
void DefWriter::writeComp(FILE* os, Node const& n, 
                PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const
{
    PlaceDB::coordinate_type xx = n.xl(); 
    PlaceDB::coordinate_type yy = n.yl(); 
    if (n.id() < m_db.numMovable())
    {
        if (x)
        {
            xx = x[n.id()];
        }
        if (y)
        {
            yy = y[n.id()];
        }
    }

    fprintf(os, "  - %s %s\n", m_db.nodeName(n).c_str(), m_db.macroName(n).c_str());
    fprintf(os, "    + %s ( %d %d ) %s ;\n", 
            std::string(PlaceStatus(n.status())).c_str(), 
            xx, yy, 
            std::string(Orient(n.orient())).c_str());
}
std::string DefWriter::ltrim(std::string const& s) const 
{
    size_t start = s.find_first_not_of(" \n\r\t\f\v");
    return (start == std::string::npos) ? "" : s.substr(start);
}
std::string DefWriter::rtrim(std::string const& s) const 
{
    std::size_t end = s.find_last_not_of(" \n\r\t\f\v");
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}
std::string DefWriter::trim(std::string const& s) const 
{
    return rtrim(ltrim(s));
}

DREAMPLACE_END_NAMESPACE
