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
        std::vector<Node>::const_iterator first, std::vector<Node>::const_iterator last, 
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
        pos1 = line.find("END");
        pos2 = line.find("COMPONENTS");
        if (pos1 != std::string::npos && 
                pos2 != std::string::npos) // match "END COMPONENTS"
        {
            // found "END COMPONENTS"
            // dump positions here 
            writeCompBlock(out, first, last, x, y);

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
                fprintf(out, "ROW %s %s %d %d %s DO %u BY %u STEP %d %d ;\n", 
                        row.name().c_str(), row.macroName().c_str(), 
                        row.xl(), row.yl(), std::string(row.orient()).c_str(), 
                        row.numSites(kX), (row.step(kY) == 0)? 1 : row.numSites(kY), row.step(kX), row.step(kY));
                ++rowCount;
            }
            else if (line == ";")
            {
                /* skip lines only containing ; */
            }
            else fprintf(out, "%s\n", line.c_str()); 
        }
    }

    in.close();
    fclose(out);
    return true;
}
bool DefWriter::writeSimple(std::string const& outFile, std::string const& version, std::string const& designName, 
        std::vector<Node>::const_iterator first, std::vector<Node>::const_iterator last, 
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
    writeCompBlock(out, first, last, x, y);
    fprintf(out, "\nEND DESIGN");

    fclose(out);
    return true;
}
void DefWriter::writeCompBlock(FILE* os, std::vector<Node>::const_iterator first, std::vector<Node>::const_iterator last, 
                PlaceDB::coordinate_type const* x, PlaceDB::coordinate_type const* y) const 
{
    fprintf(os, "COMPONENTS %lu ;\n", last-first);
    for (; first != last; ++first)
        writeComp(os, *first, x, y);
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

DREAMPLACE_END_NAMESPACE
