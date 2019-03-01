/*************************************************************************
    > File Name: BenchMetrics.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 14 Mar 2016 09:22:46 PM CDT
 ************************************************************************/

#include "BenchMetrics.h"
#include "Msg.h"

GPF_BEGIN_NAMESPACE

BenchMetrics::BenchMetrics()
{
    initPlaceDBFlag = false; 
    initAlgoDBFlag = false; 
}

void BenchMetrics::print() const
{
    if (initPlaceDBFlag)
    {
        gpfPrint(kINFO, "design name = %s\n", designName.c_str());
        gpfPrint(kINFO, "lef unit = %d\n", lefUnit);
        gpfPrint(kINFO, "def unit = %d\n", defUnit);
        gpfPrint(kINFO, "number of macros = %lu\n", numMacro);
        gpfPrint(kINFO, "number of nodes = %lu (movable %lu, fixed %lu, io %lu)\n", numNodes, numMovable, numFixed, numIOPin);
        gpfPrint(kINFO, "number of multirow nodes = %lu\n", numMultiRowMovable);
        gpfPrint(kINFO, "number of 2-row nodes = %lu\n", num2RowMovable);
        gpfPrint(kINFO, "number of 3-row nodes = %lu\n", num3RowMovable);
        gpfPrint(kINFO, "number of 4-row nodes = %lu\n", num4RowMovable);
        gpfPrint(kINFO, "number of nets = %lu\n", numNets);
        gpfPrint(kINFO, "number of rows = %lu\n", numRows);
        gpfPrint(kINFO, "number of pin connections = %lu\n", numPins);
        gpfPrint(kINFO, "number of placement blockages = %lu\n", numPlaceBlockage);
        gpfPrint(kINFO, "site dimensions = (%d, %d)\n", siteWidth, rowHeight);
        gpfPrint(kINFO, "die dimensions = (%d, %d, %d, %d)\n", dieArea.xl(), dieArea.yl(), dieArea.xh(), dieArea.yh());
        gpfPrint(kINFO, "row dimensions = (%d, %d, %d, %d)\n", rowBbox.xl(), rowBbox.yl(), rowBbox.xh(), rowBbox.yh());
        gpfPrint(kINFO, "utilization = %g\n", movableUtil);
        if (numIgnoredNet)
            gpfPrint(kWARN, "# ingored nets = %lu (nets belong to the same cells)\n", numIgnoredNet);
        if (numDuplicateNet)
            gpfPrint(kWARN, "# duplicate nets = %lu\n", numDuplicateNet);
    }
    if (initAlgoDBFlag)
    {
        gpfPrint(kINFO, "bin dimensions = %ux%u (%ux%u sites)\n", binDimension[kX], binDimension[kY], binSize[kX]/siteWidth, binSize[kY]/rowHeight);
        gpfPrint(kINFO, "sbin dimensions = %ux%u (%ux%u sites)\n", sbinDimension[kX], sbinDimension[kY], binSize[kX]/siteWidth, binSize[kY]/rowHeight);
        gpfPrint(kINFO, "number of sub rows = %u\n", numSubRows);
        gpfPrint(kINFO, "number of bin rows = %u\n", numBinRows);
        gpfPrint(kINFO, "max displacement = %d (%g um)\n", maxDisplace, maxDisplaceUm);
        gpfPrint(kINFO, "target utilization = %g\n", targetUtil);
        gpfPrint(kINFO, "target pin utilization = %g\n", targetPinUtil);
        gpfPrint(kINFO, "target PPR = %g\n", targetPPR);
    }
}

GPF_END_NAMESPACE
