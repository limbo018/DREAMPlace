/**
 * @file   DetailedPlaceDB.h
 * @author Yibo Lin
 * @date   Jan 2019
 */

#ifndef _DREAMPLACE_UTILITY_DETAILEDPLACEDB_H
#define _DREAMPLACE_UTILITY_DETAILEDPLACEDB_H

#include "utility/src/Msg.h"
#include "utility/src/Box.h"
#include "greedy_legalize/src/legality_check_cpu.h"
#include "draw_place/src/draw_place.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct Space 
{
    T xl; 
    T xh; 
};

struct BinMapIndex
{
    int bin_id; 
    int sub_id; 
};

struct RowMapIndex
{
    int row_id; 
    int sub_id; 
};

/// @brief a wrapper class of required data for detailed placement 
template <typename T>
struct DetailedPlaceDB
{
    typedef T type; 

    const T* init_x; 
    const T* init_y; 
    const T* node_size_x; 
    const T* node_size_y; 
    T* x; 
    T* y; 
    const int* flat_net2pin_map; 
    const int* flat_net2pin_start_map; 
    const int* pin2net_map; 
    const int* flat_node2pin_map; 
    const int* flat_node2pin_start_map; 
    const int* pin2node_map; 
    const T* pin_offset_x; 
    const T* pin_offset_y; 
    const unsigned char* net_mask; 
    T xl; 
    T yl; 
    T xh; 
    T yh; 
    T site_width; 
    T row_height; 
    T bin_size_x; 
    T bin_size_y; 
    int num_bins_x; 
    int num_bins_y; 
    int num_sites_x; 
    int num_sites_y; 
    int num_nodes; 
    int num_movable_nodes; 
    int num_nets;
    int num_pins; 

    inline int pos2site_x(T xx) const 
    {
        int sx = (xx-xl)/site_width; 
        sx = std::max(sx, 0); 
        sx = std::min(sx, num_sites_x-1); 
        return sx; 
    }
    inline int pos2site_y(T yy) const 
    {
        int sy = (yy-yl)/row_height; 
        sy = std::max(sy, 0); 
        sy = std::min(sy, num_sites_y-1); 
        return sy; 
    }
    /// @brief site index as an upper bound 
    inline int pos2site_ub_x(T xx) const  
    {
        int sx = ceil((xx-xl)/site_width); 
        sx = std::max(sx, 1); 
        sx = std::min(sx, num_sites_x); 
        return sx; 
    }
    /// @brief site index as an upper bound 
    inline int pos2site_ub_y(T yy) const  
    {
        int sy = ceil((yy-yl)/row_height); 
        sy = std::max(sy, 1); 
        sy = std::min(sy, num_sites_y); 
        return sy; 
    }
    inline int pos2bin_x(T xx) const  
    {
        int bx = (xx-xl)/bin_size_x; 
        bx = std::max(bx, 0); 
        bx = std::min(bx, num_bins_x-1); 
        return bx; 
    }
    inline int pos2bin_y(T yy) const  
    {
        int by = (yy-yl)/bin_size_y; 
        by = std::max(by, 0); 
        by = std::min(by, num_bins_y-1); 
        return by; 
    }
    inline void shift_box_to_layout(Box<T>& box) const  
    {
        box.xl = std::max(box.xl, xl);
        box.xl = std::min(box.xl, xh);
        box.xh = std::max(box.xh, xl);
        box.xh = std::min(box.xh, xh);
        box.yl = std::max(box.yl, yl);
        box.yl = std::min(box.yl, yh);
        box.yh = std::max(box.yh, yl);
        box.yh = std::min(box.yh, yh);
    }
    inline Box<int> box2sitebox(const Box<T>& box) const  
    {
        // xh, yh are exclusive 
        Box<int> sitebox (
                pos2site_x(box.xl), 
                pos2site_y(box.yl), 
                pos2site_ub_x(box.xh), 
                pos2site_ub_y(box.yh)
                ); 

        return sitebox; 
    }
    inline Box<int> box2binbox(const Box<T>& box) const
    {
        Box<int> binbox (
                pos2bin_x(box.xl), 
                pos2bin_y(box.yl),  
                pos2bin_x(box.xh), 
                pos2bin_y(box.yh) 
                );

        return binbox; 
    }
    /// @brief align x coordinate to site 
    inline T align2site(T xx) const 
    {
        return floor((xx-xl)/site_width)*site_width+xl; 
    }
    /// @brief compute optimal region for a cell 
    /// The method to compute optimal region ignores the pin offsets of the target cell. 
    /// If we want to consider the pin offsets, there may not be feasible box for the optimal region. 
    /// Thus, this is just an approximate optimal region. 
    /// When using the optimal region, one needs to refer to the center of the cell to the region, or the region completely covers the entire cell. 
    Box<T> compute_optimal_region(int node_id) const
    {
        Box<T> box (
                std::numeric_limits<T>::max(),
                std::numeric_limits<T>::max(),
                -std::numeric_limits<T>::max(),
                -std::numeric_limits<T>::max()
                ); 
        for (int node2pin_id = flat_node2pin_start_map[node_id]; node2pin_id < flat_node2pin_start_map[node_id+1]; ++node2pin_id)
        {
            int node_pin_id = flat_node2pin_map[node2pin_id];
            int net_id = pin2net_map[node_pin_id];
            if (net_mask[net_id])
            {
                for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id+1]; ++net2pin_id)
                {
                    int net_pin_id = flat_net2pin_map[net2pin_id];
                    int other_node_id = pin2node_map[net_pin_id];
                    if (node_id != other_node_id)
                    {
                        box.xl = std::min(box.xl, x[other_node_id]+pin_offset_x[net_pin_id]);
                        box.xh = std::max(box.xh, x[other_node_id]+pin_offset_x[net_pin_id]);
                        box.yl = std::min(box.yl, y[other_node_id]+pin_offset_y[net_pin_id]);
                        box.yh = std::max(box.yh, y[other_node_id]+pin_offset_y[net_pin_id]);
                    }
                }
            }
        }
        shift_box_to_layout(box);

        return box; 
    }
    /// @brief compute HPWL for a net 
    T compute_net_hpwl(int net_id) const
    {
        Box<T> box (
                std::numeric_limits<T>::max(),
                std::numeric_limits<T>::max(),
                -std::numeric_limits<T>::max(),
                -std::numeric_limits<T>::max()
                ); 
        for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id+1]; ++net2pin_id)
        {
            int net_pin_id = flat_net2pin_map[net2pin_id];
            int other_node_id = pin2node_map[net_pin_id];
            box.xl = std::min(box.xl, x[other_node_id]+pin_offset_x[net_pin_id]);
            box.xh = std::max(box.xh, x[other_node_id]+pin_offset_x[net_pin_id]);
            box.yl = std::min(box.yl, y[other_node_id]+pin_offset_y[net_pin_id]);
            box.yh = std::max(box.yh, y[other_node_id]+pin_offset_y[net_pin_id]);
        }
        if (box.xl == std::numeric_limits<T>::max() || box.yl == std::numeric_limits<T>::max())
        {
            return (T)0; 
        }
        return (box.xh-box.xl) + (box.yh-box.yl);
    }
    /// @brief compute HPWL for all nets 
    T compute_total_hpwl() const
    {
        //dreamplacePrint(kDEBUG, "start compute_total_hpwl\n");
        T total_hpwl = 0; 
        for (int net_id = 0; net_id < num_nets; ++net_id)
        {
            //if (net_mask[net_id])
            {
                total_hpwl += compute_net_hpwl(net_id);
            }
        }
        //dreamplacePrint(kDEBUG, "end compute_total_hpwl\n");
        return total_hpwl; 
    }
    /// @brief distribute cells to rows 
    void make_row2node_map(const T* vx, const T* vy, std::vector<std::vector<int> >& row2node_map) const 
    {
        // distribute cells to rows 
        for (int i = 0; i < num_nodes; ++i)
        {
            //T node_xl = vx[i]; 
            T node_yl = vy[i];
            //T node_xh = node_xl+node_size_x[i];
            T node_yh = node_yl+node_size_y[i];

            int row_idxl = (node_yl-yl)/row_height; 
            int row_idxh = ceil((node_yh-yl)/row_height)+1;
            row_idxl = std::max(row_idxl, 0); 
            row_idxh = std::min(row_idxh, num_sites_y); 

            for (int row_id = row_idxl; row_id < row_idxh; ++row_id)
            {
                T row_yl = yl+row_id*row_height; 
                T row_yh = row_yl+row_height; 

                if (node_yl < row_yh && node_yh > row_yl) // overlap with row 
                {
                    row2node_map[row_id].push_back(i); 
                }
            }
        }

        // sort cells within rows 
        for (int i = 0; i < num_sites_y; ++i)
        {
          // it is safer to sort by center 
          // sometimes there might be cells with 0 sizes 
            std::sort(row2node_map[i].begin(), row2node_map[i].end(), 
                    [&] (int node_id1, int node_id2) {return
                    vx[node_id1]+node_size_x[node_id1]/2 <
                    vx[node_id2]+node_size_x[node_id2]/2;}
                    );
        }
    }
    /// @brief distribute movable cells to bins 
    void make_bin2node_map(const T* host_x, const T* host_y, 
            const T* host_node_size_x, const T* host_node_size_y, 
            std::vector<std::vector<int> >& bin2node_map, std::vector<BinMapIndex>& node2bin_map) const 
    {
        // construct bin2node_map 
        for (int i = 0; i < num_movable_nodes; ++i)
        {
            int node_id = i; 
            T node_x = host_x[node_id] + host_node_size_x[node_id]/2; 
            T node_y = host_y[node_id] + host_node_size_y[node_id]/2;

            int bx = std::min(std::max((int)((node_x-xl)/bin_size_x), 0), num_bins_x-1);
            int by = std::min(std::max((int)((node_y-yl)/bin_size_y), 0), num_bins_y-1);
            int bin_id = bx*num_bins_y+by; 
            //int sub_id = bin2node_map.at(bin_id).size(); 
            bin2node_map.at(bin_id).push_back(node_id); 
        }
        // construct node2bin_map 
        for (unsigned int bin_id = 0; bin_id < bin2node_map.size(); ++bin_id)
        {
            for (unsigned int sub_id = 0; sub_id < bin2node_map[bin_id].size(); ++sub_id)
            {
                int node_id = bin2node_map[bin_id][sub_id];
                BinMapIndex& bm_idx = node2bin_map.at(node_id); 
                bm_idx.bin_id = bin_id; 
                bm_idx.sub_id = sub_id; 
            }
        }
#ifdef DEBUG
        int max_num_nodes_per_bin = 0; 
        for (unsigned int i = 0; i < bin2node_map.size(); ++i)
        {
            max_num_nodes_per_bin = std::max(max_num_nodes_per_bin, (int)bin2node_map[i].size());
        }
        printf("[D] max_num_nodes_per_bin = %d\n", max_num_nodes_per_bin);
#endif
    }
    /// @brief check whether placement is legal 
    bool check_legality() const 
    {
        return legalityCheckKernelCPU(
                init_x, init_y, 
                node_size_x, node_size_y, 
                x, y, 
                site_width, row_height, 
                xl, yl, xh, yh,
                num_nodes, 
                num_movable_nodes
                );
    }
    /// @brief draw placement 
    void draw_place(const char* filename) const 
    {
        drawPlaceLauncher<T>(
                x, y, 
                node_size_x, node_size_y, 
                pin_offset_x, pin_offset_y, 
                pin2node_map, 
                num_nodes, 
                num_movable_nodes, 
                0, 
                flat_net2pin_start_map[num_nets], 
                xl, yl, xh, yh, 
                site_width, row_height, 
                bin_size_x, bin_size_y, 
                filename
                );
    }
};

DREAMPLACE_END_NAMESPACE

#endif
