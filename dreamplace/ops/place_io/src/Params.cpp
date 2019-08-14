/*************************************************************************
    > File Name: Params.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Thu 18 Jun 2015 08:48:09 PM CDT
 ************************************************************************/

#include "Params.h"
#include "Util.h"
#include <iostream>
#include <fstream>
#include <boost/bind.hpp>
#include <boost/algorithm/string.hpp>
#include <limbo/string/String.h>

DREAMPLACE_BEGIN_NAMESPACE

std::string toString(SolutionFileFormat ff)
{
    switch (ff)
    {
        case DEF: return "DEF";
        case DEFSIMPLE: return "DEFSIMPLE";
        case BOOKSHELF: return "BOOKSHELF";
        case BOOKSHELFALL: return "BOOKSHELFALL";
        default: return "UNKNOWN";
    }
}

UserParam::UserParam()
{
    defOutput = "";
    rptOutput = "";
    targetUtil = 0;
    targetPinUtil = 0;
    targetPPR = 0;
    maxDisplace = 0;
    binSize[0] = binSize[1] = 10;
    binSize[2] = binSize[3] = 5;
    binSpaceThreshold = 0.2;

    enablePlace = true;
    enableLegalize = true;
    evaluateOverlap = false;
    moveMultiRowCell = true;
    alignPowerLine = true;
    clusterCell = false;
    sortNetsByDegree = false;

    drawPlaceInit = false;
    drawPlaceFinal = false;
    drawPlaceAnime = false;
    drawRegion[0] = std::numeric_limits<int>::min();
    drawRegion[1] = std::numeric_limits<int>::min();
    drawRegion[2] = std::numeric_limits<int>::max();
    drawRegion[3] = std::numeric_limits<int>::max();

    fileFormat = DEF;
    maxIters = 6;
}
bool UserParam::read(int argc, char** argv)
{
    return readNormal(argc, argv, UserParamExtHelper());
}
bool UserParam::readNormal(int argc, char** argv, UserParamExtHelper const& helper)
{
    printWelcome();
    // not all arguments can be initialized by defaultParam
    UserParam defaultParam; // default parameters from default constructor
    char buf[64];
    // some default vectors
    // program_options does not support passing a pair of values, so add comma to make it a std::string and convert to pairs later
    std::string defaultAbuPercStr[4] = {"2,10", "5,5", "10,2", "20,1"};
    dreamplaceSPrint(kNONE, buf, "%d,%d,%d,%d", defaultParam.drawRegion[0], defaultParam.drawRegion[1], defaultParam.drawRegion[2], defaultParam.drawRegion[3]);
    std::string defaultDrawRegionStr = buf;
    std::vector<std::string> vAbuPercStr;
    std::vector<std::string> vDefIgnoreCellType;
    bool help = false;
    std::string fileFormatStr;
    std::string drawRegionStr;
    // append options here
    typedef limbo::programoptions::ProgramOptions po_type;
    using limbo::programoptions::Value;
    po_type desc (std::string("Available options"));
    desc.add_option(Value<bool>("--help", &help, "print help message").default_value(help).help(true))
        .add_option(Value<std::vector<std::string> >("--lef_input", &vLefInput, "input LEF files")) 
        .add_option(Value<std::string>("--def_input", &defInput, "input DEF file"))
        .add_option(Value<std::string>("--verilog_input", &verilogInput, "input Verilog file"))
        .add_option(Value<std::string>("--bookshelf_aux_input", &bookshelfAuxInput, "input Bookshelf aux file"))
        .add_option(Value<std::string>("--bookshelf_pl_input", &bookshelfPlInput, "additional input Bookshelf pl file"))
        .add_option(Value<std::string>("--def_size_input", &defSizeInput, "input def size file for benchmarks from CUHK"))
        .add_option(Value<std::string>("--def_output", &defOutput, "output DEF file"))
        .add_option(Value<std::string>("--rpt_output", &rptOutput, "output HTML report file"))
        .add_option(Value<double>("--target_util", &targetUtil, "target utilization").default_value(defaultParam.targetUtil))
        .add_option(Value<double>("--target_pin_util", &targetPinUtil, "target pin utilization per site").default_value(defaultParam.targetPinUtil))
        .add_option(Value<double>("--target_ppr", &targetPPR, "target pin pair ratio").default_value(defaultParam.targetPPR))
        .add_option(Value<double>("--max_displace", &maxDisplace, "maximum displacement in micron").default_value(defaultParam.maxDisplace))
        .add_option(Value<unsigned>("--bin_width", &binSize[kX], "bin width (#rows) in horizontal direction").default_value(defaultParam.binSize[0]))
        .add_option(Value<unsigned>("--bin_height", &binSize[kY], "bin height (#rows) in vertical direction").default_value(defaultParam.binSize[1]))
        .add_option(Value<unsigned>("--sbin_width", &binSize[2+kX], "sbin width (#rows) in horizontal direction").default_value(defaultParam.binSize[2]))
        .add_option(Value<unsigned>("--sbin_height", &binSize[2+kY], "sbin height (#rows) in vertical direction").default_value(defaultParam.binSize[3]))
        .add_option(Value<double>("--bin_space_threshold", &binSpaceThreshold,
         "when the capacity of a bin (exclude fixed macros) is smaller than a specific percentage of the bin area, do not take into the calculation for abu density").default_value(defaultParam.binSpaceThreshold))
        .add_option(Value<std::vector<std::string> >("--abu", &vAbuPercStr,
         "top percentage and weight pairs of bins for abu calculation").default_value(std::vector<std::string>(defaultAbuPercStr, defaultAbuPercStr+4), "2,10 5,5 10,2 20,1"))
        .add_option(Value<std::set<std::string> >("--def_ignore_cells", &sDefIgnoreCellType, "cells ignored in input DEF file"))
        .add_option(Value<std::set<std::string> >("--macro_obs_aware_layers", &sMacroObsAwareLayer, "layers of macro obstruction that are considered during placement"))
        .add_option(Value<bool>("--enable_place", &enablePlace, "enable placement").default_value(defaultParam.enablePlace))
        .add_option(Value<bool>("--enable_legalize", &enableLegalize, "enable legalization").default_value(defaultParam.enableLegalize))
        .add_option(Value<bool>("--evaluate_overlap", &evaluateOverlap, "evaluate overlapping pairs of cells").default_value(defaultParam.evaluateOverlap))
        .add_option(Value<bool>("--move_multi_row_cell", &moveMultiRowCell, "enable multi-row cell movement").default_value(defaultParam.moveMultiRowCell))
        .add_option(Value<bool>("--align_power_line", &alignPowerLine, "enable power line alignment for multi-row cell").default_value(defaultParam.alignPowerLine))
        .add_option(Value<bool>("--cluster_cell", &clusterCell, "enable cell clustering in chain global move").default_value(defaultParam.clusterCell))
        .add_option(Value<bool>("--sort_nets_by_degree", &sortNetsByDegree, "sort nets by degree").default_value(defaultParam.sortNetsByDegree))
        .add_option(Value<bool>("--draw_place_init", &drawPlaceInit, "draw initial placement").default_value(defaultParam.drawPlaceInit))
        .add_option(Value<bool>("--draw_place_final", &drawPlaceFinal, "draw final placement").default_value(defaultParam.drawPlaceFinal))
        .add_option(Value<bool>("--draw_place_anime", &drawPlaceAnime, "draw placement for animation").default_value(defaultParam.drawPlaceAnime))
        .add_option(Value<std::string>("--draw_region", &drawRegionStr, "draw placement region").default_value(defaultDrawRegionStr))
        .add_option(Value<std::string>("--file_format", &fileFormatStr, "file format to write placement solution <DEF | DEFSIMPLE | BOOKSHELF | BOOKSHELFALL>").default_value(toString(defaultParam.fileFormat)))
        .add_option(Value<unsigned>("--max_iters", &maxIters, "maximum optimization iterations").default_value(defaultParam.maxIters))
        ;
    helper.addOptions(desc); // extension

    try
    {
        desc.parse(argc, argv);

        // print help message
        if (help)
        {
            std::cout << desc << "\n";
            exit(1);
        }

        helper.processAhead(desc); // extension

        if (!desc.count("--bookshelf_aux_input")) // if specified Bookshelf input, LEF/DEF input is no longer required
        {
            // if not specified, must provide LEF/DEF
            dreamplaceAssertMsg(desc.count("--lef_input") && desc.count("--def_input"), "need either Bookshelf or LEF/DEF input files");
        }
        if (!desc.count("--def_output"))
        {
            // set default value
            defOutput = limbo::trim_file_suffix(limbo::get_file_name(defInput)) + "-out.def";
        }
        // post processing vAbuPerc
        vAbuPerc.clear();
        vAbuPerc.reserve(vAbuPercStr.size());
        for (std::vector<std::string>::const_iterator it = vAbuPercStr.begin(), ite = vAbuPercStr.end(); it != ite; ++it)
        {
            std::size_t found = it->find(',');
            vAbuPerc.push_back(std::make_pair(atoi(it->substr(0, found).c_str()), atoi(it->substr(found+1).c_str())));
        }
        std::sort(vAbuPerc.begin(), vAbuPerc.end(),
                boost::bind(&std::pair<int, int>::first, _1)
                <  boost::bind(&std::pair<int, int>::first, _2));

        // post processing fileFormat
        if (limbo::iequals(fileFormatStr, "DEF"))
            fileFormat = DEF;
        else if (limbo::iequals(fileFormatStr, "DEFSIMPLE"))
            fileFormat = DEFSIMPLE;
        else if (limbo::iequals(fileFormatStr, "BOOKSHELF"))
            fileFormat = BOOKSHELF;
        else if (limbo::iequals(fileFormatStr, "BOOKSHELFALL"))
            fileFormat = BOOKSHELFALL;
        // if specified Bookshelf input, fileFormat should also be Bookshelf
        if (defInput.empty() && (fileFormat == DEF || fileFormat == DEFSIMPLE))
        {
            fileFormat = BOOKSHELF;
            dreamplacePrint(kWARN, "DEF input file not specified, cannot output DEF file; set to DEFSIMPLE\n");
        }

        // post processing drawRegion
        std::vector<std::string> vToken;
        boost::trim(drawRegionStr);
        boost::split(vToken, drawRegionStr, boost::is_any_of(","));
        for (int i = 0; i < 4; ++i)
            drawRegion[i] = atoi(vToken.at(i).c_str());

        helper.processLater(desc); // extension
    }
    catch (std::exception& e)
    {
        // print help message and error message
        std::cout << desc << "\n";
        dreamplacePrint(kERROR, "%s\n", e.what());
        return false;
    }

    /// print parameters
    printParams();

    return true;
}

void UserParam::printParams() const 
{
    dreamplacePrint(kINFO, "lef_input = ");
    for (std::vector<std::string>::const_iterator it = vLefInput.begin(), ite = vLefInput.end(); it != ite; ++it)
        dreamplacePrint(kNONE, "%s ", it->c_str());
    dreamplacePrint(kNONE, "\n");
    dreamplacePrint(kINFO, "def_input = %s\n", defInput.c_str());
    dreamplacePrint(kINFO, "verilog_input = %s\n", verilogInput.c_str());
    dreamplacePrint(kINFO, "bookshelf_aux_input = %s\n", bookshelfAuxInput.c_str());
    dreamplacePrint(kINFO, "bookshelf_pl_input = %s\n", bookshelfPlInput.c_str());
    dreamplacePrint(kINFO, "def_size_input = %s\n", defSizeInput.c_str());
    dreamplacePrint(kINFO, "def_output = %s\n", defOutput.c_str());
    dreamplacePrint(kINFO, "rpt_output = %s\n", rptOutput.c_str());
    dreamplacePrint(kINFO, "target_util = %g\n", targetUtil);
    dreamplacePrint(kINFO, "max_displace = %g\n", maxDisplace);
    dreamplacePrint(kINFO, "bin size = (%u, %u) #rows\n", binSize[kX], binSize[kY]);
    dreamplacePrint(kINFO, "sbin size = (%u, %u) #rows\n", binSize[2+kX], binSize[2+kY]);
    dreamplacePrint(kINFO, "bin_space_threshold = %g\n", binSpaceThreshold);
    dreamplacePrint(kINFO, "abu = ");
    for (std::vector<std::pair<int, int> >::const_iterator it = vAbuPerc.begin(), ite = vAbuPerc.end(); it != ite; ++it)
        dreamplacePrint(kNONE, "%d,%d ", it->first, it->second);
    dreamplacePrint(kNONE, "\n");
    dreamplacePrint(kINFO, "def_ignore_cells = ");
    for (std::set<std::string>::const_iterator it = sDefIgnoreCellType.begin(), ite = sDefIgnoreCellType.end(); it != ite; ++it)
        dreamplacePrint(kNONE, "%s ", it->c_str());
    dreamplacePrint(kNONE, "\n");
    dreamplacePrint(kINFO, "macro_obs_aware_layers = ");
    for (std::set<std::string>::const_iterator it = sMacroObsAwareLayer.begin(), ite = sMacroObsAwareLayer.end(); it != ite; ++it)
        dreamplacePrint(kNONE, "%s ", it->c_str());
    dreamplacePrint(kNONE, "\n");
    dreamplacePrint(kINFO, "enable_place = %s\n", ((enablePlace)? "true" : "false"));
    dreamplacePrint(kINFO, "enable_legalize = %s\n", ((enableLegalize)? "true" : "false"));
    dreamplacePrint(kINFO, "evaluate_overlap = %s\n", ((evaluateOverlap)? "true" : "false"));
    dreamplacePrint(kINFO, "move_multi_row_cell = %s\n", ((moveMultiRowCell)? "true" : "false"));
    dreamplacePrint(kINFO, "align_power_line = %s\n", ((alignPowerLine)? "true" : "false"));
    dreamplacePrint(kINFO, "cluster_cell = %s\n", ((clusterCell)? "true" : "false"));
    dreamplacePrint(kINFO, "sort_nets_by_degree = %s\n", ((sortNetsByDegree)? "true" : "false"));
    dreamplacePrint(kINFO, "file_format = %s\n", toString(fileFormat).c_str());
    dreamplacePrint(kINFO, "max_iters = %u\n", maxIters);
}

void UserParam::printWelcome() const
{
}

DREAMPLACE_END_NAMESPACE
