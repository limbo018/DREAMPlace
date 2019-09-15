/*************************************************************************
    > File Name: Params.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Thu Jun 18 01:03:56 2015
 ************************************************************************/

#ifndef DREAMPLACE_PARAMS_H
#define DREAMPLACE_PARAMS_H

#include <string>
#include <vector>
#include <set>
#include <limbo/programoptions/ProgramOptions.h>

#include "Enums.h"

DREAMPLACE_BEGIN_NAMESPACE

/// placement solution format 
enum SolutionFileFormat 
{
    DEF, // full DEF format
    DEFSIMPLE, // simplified DEF format with only component positions
    BOOKSHELF, // write placement solution .plx in bookshlef format
    BOOKSHELFALL // write .nodes, .nets, ... in bookshlef format
};

/// convert enums to string 
extern std::string toString(SolutionFileFormat ff);

/// an extension for UserParam
/// user must derive a new class with following three virtual member functions
/// addOptions(desc)
/// processAhead(desc)
/// processLater(desc)
struct UserParamExtHelper
{
    typedef limbo::programoptions::ProgramOptions po_type;

    /// constructor
    UserParamExtHelper() {}

    virtual void addOptions(po_type& /*desc*/) const {}
    /// process at the beginning of try/catch
    virtual void processAhead(po_type& /*desc*/) const {}
    /// process at the end of try/catch
    virtual void processLater(po_type& /*desc*/) const {}
};


/// user defined parameters
struct UserParam
{
    /// constructor
    UserParam();
    /// read command line options
    bool read(int argc, char** argv);
    /// print parameters
    virtual void printParams() const;
    /// print welcome message
    virtual void printWelcome() const;

    /// LEF/DEF input files 
    std::vector<std::string> vLefInput;
    std::string defInput;
    std::string verilogInput;

    /// Bookshelf input file
    std::string bookshelfAuxInput;
    std::string bookshelfPlInput; ///< additional .pl file

    /// DEF size input file, only appear in the ISPD 2015 benchmarks from CUHK
    std::string defSizeInput;

    /// DEF output file
    std::string defOutput;

    /// report output file
    std::string rptOutput; ///< report output in html format

    /// specific metrics
    double targetUtil; ///< target utilization
    double targetPinUtil; ///< target pin utilization
    double targetPPR; ///< target pin pair ratio
    double maxDisplace;
    unsigned binSize[4]; ///< bin/sbin dimensions of x and y in # rows
    double binSpaceThreshold; ///< when the capacity of a bin (exclude fixed macros) is smaller than a specific percentage of the bin area
                            ///< do not take into the calculation of abu density
    std::vector<std::pair<int, int> > vAbuPerc; ///< top percentage of bins and weights for abu calculation
    std::set<std::string> sDefIgnoreCellType; ///< ignored cells in def input file
    std::set<std::string> sMacroObsAwareLayer; ///< record the layers for macro obstruction that should be overlap-free during placement

    /// flow switches
    bool enablePlace; ///< top level enable flag for placement, if true, perform placement
    bool enableLegalize; ///< whether enable legalization, if true, perform legalization flow, not follow enablePlace

    bool evaluateOverlap; ///< evalute overlap or not
    bool moveMultiRowCell; ///< whether move multi-row cell or not
    bool alignPowerLine; ///< whether align multi-row cell to the correct row for power line alignment
    bool clusterCell; ///< whether enable cell clustering in chain global move
    bool sortNetsByDegree; ///< whether sort nets by net degrees after reading the benchmark

    bool drawPlaceInit; ///< draw initial placement
    bool drawPlaceFinal; ///< draw final placement
    bool drawPlaceAnime; ///< draw placement for animation
    int drawRegion[4]; ///< draw placement region (xl, yl, xh, yh) in database unit

    /// additional options
    SolutionFileFormat fileFormat; ///< file format to write placement solution
    unsigned maxIters; ///< maximum optimization iterations

    protected:
        /// read command line options
        bool readNormal(int argc, char** argv, UserParamExtHelper const& helper);
};

DREAMPLACE_END_NAMESPACE

#endif
