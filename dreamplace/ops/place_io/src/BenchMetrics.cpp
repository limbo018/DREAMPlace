/*************************************************************************
    > File Name: BenchMetrics.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 14 Mar 2016 09:22:46 PM CDT
 ************************************************************************/

#include "BenchMetrics.h"

DREAMPLACE_BEGIN_NAMESPACE

BenchMetrics::BenchMetrics() {
  initPlaceDBFlag = false;
  initAlgoDBFlag = false;
}

void BenchMetrics::print() const {
  if (initPlaceDBFlag) {
    dreamplacePrint(kINFO, "design name = %s\n", designName.c_str());
    dreamplacePrint(kINFO, "lef unit = %d\n", lefUnit);
    dreamplacePrint(kINFO, "def unit = %d\n", defUnit);
    dreamplacePrint(kINFO, "number of macros = %lu\n", numMacro);
    dreamplacePrint(kINFO,
                    "number of nodes = %lu (movable %lu, fixed %lu, io %lu)\n",
                    numNodes, numMovable, numFixed, numIOPin);
    dreamplacePrint(kINFO, "number of multirow nodes = %lu\n",
                    numMultiRowMovable);
    dreamplacePrint(kINFO, "number of 2-row nodes = %lu\n", num2RowMovable);
    dreamplacePrint(kINFO, "number of 3-row nodes = %lu\n", num3RowMovable);
    dreamplacePrint(kINFO, "number of 4-row nodes = %lu\n", num4RowMovable);
    dreamplacePrint(kINFO, "number of nets = %lu\n", numNets);
    dreamplacePrint(kINFO, "number of rows = %lu\n", numRows);
    dreamplacePrint(kINFO, "number of pin connections = %lu\n", numPins);
    dreamplacePrint(kINFO, "number of placement blockages = %lu\n",
                    numPlaceBlockage);
    dreamplacePrint(kINFO, "site dimensions = (%d, %d)\n", siteWidth,
                    rowHeight);
    dreamplacePrint(kINFO, "die dimensions = (%d, %d, %d, %d)\n", dieArea.xl(),
                    dieArea.yl(), dieArea.xh(), dieArea.yh());
    dreamplacePrint(kINFO, "row dimensions = (%d, %d, %d, %d)\n", rowBbox.xl(),
                    rowBbox.yl(), rowBbox.xh(), rowBbox.yh());
    dreamplacePrint(kINFO, "utilization = %g\n", movableUtil);
    if (numIgnoredNet)
      dreamplacePrint(kWARN,
                      "# ingored nets = %lu (nets belong to the same cells)\n",
                      numIgnoredNet);
    if (numDuplicateNet)
      dreamplacePrint(kWARN, "# duplicate nets = %lu\n", numDuplicateNet);
  }
  if (initAlgoDBFlag) {
    dreamplacePrint(kINFO, "bin dimensions = %ux%u (%ux%u sites)\n",
                    binDimension[kX], binDimension[kY], binSize[kX] / siteWidth,
                    binSize[kY] / rowHeight);
    dreamplacePrint(kINFO, "sbin dimensions = %ux%u (%ux%u sites)\n",
                    sbinDimension[kX], sbinDimension[kY],
                    binSize[kX] / siteWidth, binSize[kY] / rowHeight);
    dreamplacePrint(kINFO, "number of sub rows = %u\n", numSubRows);
    dreamplacePrint(kINFO, "number of bin rows = %u\n", numBinRows);
    dreamplacePrint(kINFO, "max displacement = %d (%g um)\n", maxDisplace,
                    maxDisplaceUm);
    dreamplacePrint(kINFO, "target utilization = %g\n", targetUtil);
    dreamplacePrint(kINFO, "target pin utilization = %g\n", targetPinUtil);
    dreamplacePrint(kINFO, "target PPR = %g\n", targetPPR);
  }
}

DREAMPLACE_END_NAMESPACE
