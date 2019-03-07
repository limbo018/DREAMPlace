/*************************************************************************
    > File Name: BenchMetrics.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 14 Mar 2016 09:15:53 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_BENCHMETRICS_H
#define DREAMPLACE_BENCHMETRICS_H

#include <string>
#include <vector>
#include "Box.h"

DREAMPLACE_BEGIN_NAMESPACE

/// ================================================
/// a simple class storing metrics for benchmarks 
/// which is to help report benchmark statistics 
/// ================================================

struct BenchMetrics
{
    /// metrics from PlaceDB
    std::string designName; 
    int lefUnit; 
    int defUnit;
    std::size_t numMacro;
    std::size_t numNodes; 
    std::size_t numMovable;
    std::size_t numFixed;
    std::size_t numIOPin;
    std::size_t numMultiRowMovable;
    std::size_t num2RowMovable;
    std::size_t num3RowMovable;
    std::size_t num4RowMovable;
    std::size_t numNets;
    std::size_t numRows;
    std::size_t numPins;
    std::size_t numPlaceBlockage; 
    unsigned siteWidth;
    unsigned rowHeight;
    Box<int> dieArea;
    Box<int> rowBbox;
    double movableUtil; 
    std::size_t numIgnoredNet; 
    std::size_t numDuplicateNet; 

    bool initPlaceDBFlag; ///< a flag indicates whether it is initialized, must set to true after initialization, from PlaceDB

    /// metrics from AlgoDB 
    unsigned binDimension[2]; ///< number of bins 
    int binSize[2]; 
    unsigned sbinDimension[2]; ///< number of sbins 
    int sbinSize[2];
    unsigned numSubRows; 
    unsigned numBinRows;
    int maxDisplace; ///< max displacement in database unit 
    double maxDisplaceUm; ///< max displacement in micron 
    double targetUtil;
    double targetPinUtil;
    double targetPPR;

    bool initAlgoDBFlag; ///< initialized flag form AlgoDB

    BenchMetrics();

    void print() const;
};

DREAMPLACE_END_NAMESPACE

#endif
