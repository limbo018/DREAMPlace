/**
 * @file   hannan_legalize.h
 * @author Yibo Lin
 * @date   Nov 2019
 */

#ifndef DREAMPLACE_MACRO_LEGALIZE_HANNAN_LEGALIZE_H
#define DREAMPLACE_MACRO_LEGALIZE_HANNAN_LEGALIZE_H

#include <vector>
#include <algorithm>
#include "utility/src/diamond_search.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief A class models Hannan grids. 
template <typename T>
class HannanGrids
{
    public:
        HannanGrids(const T* x, const T* y, const T* width, const T* height, std::size_t n, 
                const T xl, const T yl, const T xh, const T yh, 
                const T spacing_x, const T spacing_y)
        {
            build(x, y, width, height, n, xl, yl, xh, yh, spacing_x, spacing_y);
        }

        std::size_t dim_x() const 
        {
            return m_coordx.size(); 
        }
        std::size_t dim_y() const 
        {
            return m_coordy.size();
        }
        /// @brief query x index in log(n) time complexity
        std::size_t grid_x(T x) const 
        {
            auto it = std::lower_bound(m_coordx.begin(), m_coordx.end(), x);
            std::size_t ix = std::min((std::size_t)std::distance(m_coordx.begin(), it), dim_x()-1);
            T gxl = m_coordx[ix]; 
            if (gxl > x && ix)
            {
                ix -= 1; 
            }
            return ix;
        }
        /// @brief query y index in log(n) time complexity
        std::size_t grid_y(T y) const 
        {
            auto it = std::lower_bound(m_coordy.begin(), m_coordy.end(), y);
            std::size_t iy = std::min((std::size_t)std::distance(m_coordy.begin(), it), dim_y()-1);
            T gyl = m_coordy[iy]; 
            if (gyl > y && iy)
            {
                iy -= 1; 
            }
            return iy;
        }
        /// @brief get x coordinate of a grid 
        T coord_x(std::size_t ix) const 
        {
            return m_coordx[ix];
        }
        /// @brief get y coordinate of a grid 
        T coord_y(std::size_t iy) const 
        {
            return m_coordy[iy];
        }
        /// @brief check whether a grid overlaps with a rectangle. 
        /// Touching is not considered as overlap. 
        bool overlap(std::size_t ix, std::size_t iy, T xl, T yl, T xh, T yh) const 
        {
            T gxl = m_coordx[ix]; 
            T gxh = (ix+1 == dim_x())? std::numeric_limits<T>::max() : m_coordx[ix+1];
            T gyl = m_coordy[iy];
            T gyh = (iy+1 == dim_y())? std::numeric_limits<T>::max() : m_coordy[iy+1];

            return std::max(gxl, xl) < std::min(gxh, xh) 
                && std::max(gyl, yl) < std::min(gyh, yh);
        }
    protected:
        /// @brief build grids from rectangles and boundaries
        void build(const T* x, const T* y, const T* width, const T* height, std::size_t n, 
                const T xl, const T yl, const T xh, const T yh, 
                const T spacing_x, const T spacing_y)
        {
            // collect all scan lines
            m_coordx.reserve((n<<1)+2);
            m_coordy.reserve((n<<1)+2);
            m_coordx.push_back(xl); 
            m_coordx.push_back(xh); 
            m_coordy.push_back(yl);
            m_coordy.push_back(yh);
            for (std::size_t i = 0; i < n; ++i)
            {
                m_coordx.push_back(x[i]);
                m_coordx.push_back(x[i]+width[i]);
                m_coordy.push_back(y[i]);
                m_coordy.push_back(y[i]+height[i]);
            }

            // sort and make them unique 
            std::sort(m_coordx.begin(), m_coordx.end());
            std::sort(m_coordy.begin(), m_coordy.end());
            m_coordx.resize(std::distance(m_coordx.begin(), std::unique(m_coordx.begin(), m_coordx.end())));
            m_coordy.resize(std::distance(m_coordy.begin(), std::unique(m_coordy.begin(), m_coordy.end())));

            // in case some grids are too large 
            // add more scan lines with step size spacing_x and spacing_y 
            for (std::size_t i = 1, ie = m_coordx.size(); i < ie; ++i)
            {
                T gxl = m_coordx[i-1];
                T gxh = m_coordx[i];

                for (T xl = gxl + spacing_x; xl < gxh; xl += spacing_x)
                {
                    m_coordx.push_back(xl);
                }
            }
            for (std::size_t i = 1, ie = m_coordy.size(); i < ie; ++i)
            {
                T gyl = m_coordy[i-1];
                T gyh = m_coordy[i];

                for (T yl = gyl + spacing_y; yl < gyh; yl += spacing_y)
                {
                    m_coordy.push_back(yl);
                }
            }

            // they should already be unique 
            std::sort(m_coordx.begin(), m_coordx.end());
            std::sort(m_coordy.begin(), m_coordy.end());

#ifdef DEBUG
            for (auto cx : m_coordx)
            {
                dreamplacePrint(kNONE, "%g ", (double)cx);
            }
            dreamplacePrint(kNONE, "\n"); 
            for (auto cy : m_coordy)
            {
                dreamplacePrint(kNONE, "%g ", (double)cy);
            }
            dreamplacePrint(kNONE, "\n"); 
#endif
        }

        std::vector<T> m_coordx; ///< coordinates of grid lines in x direction 
        std::vector<T> m_coordy; ///< coordinates of grid lines in y direction
};

/// @brief A class models binary maps on Hannan grids. 
template <typename T>
class HannanGridMap : public HannanGrids<T>
{
    public:
        typedef HannanGrids<T> base_type;

        HannanGridMap(const T* x, const T* y, const T* width, const T* height, std::size_t n, 
                const T xl, const T yl, const T xh, const T yh, 
                const T spacing_x, const T spacing_y)
            : base_type(x, y, width, height, n, xl, yl, xh, yh, spacing_x, spacing_y)
        {
            // construct 2D binary map 
            m_map.assign(this->dim_x()*this->dim_y(), 0);
        }

        /// @brief set an entry in grid map 
        void set(std::size_t ix, std::size_t iy, bool value) 
        {
            m_map[ix*this->dim_y()+iy] = value; 
        }

        /// @brief get an entry in grid map 
        bool at(std::size_t ix, std::size_t iy) const 
        {
            return m_map[ix*this->dim_y()+iy];
        }

        /// @brief check whether a rectangle overlaps with any grid in the map 
        bool overlap(T xl, T yl, T xh, T yh) const 
        {
            std::size_t ixl = this->grid_x(xl); 
            std::size_t ixh = this->grid_x(xh) + 1; 
            std::size_t iyl = this->grid_y(yl); 
            std::size_t iyh = this->grid_y(yh) + 1; 

            for (std::size_t ix = ixl; ix < ixh; ++ix)
            {
                for (std::size_t iy = iyl; iy < iyh; ++iy)
                {
                    if (this->base_type::overlap(ix, iy, xl, yl, xh, yh) && this->at(ix, iy))
                    {
                        return true; 
                    }
                }
            }
            return false; 
        }

        /// @brief add a rectangle to the grid map 
        void add(T xl, T yl, T xh, T yh) 
        {
            std::size_t ixl = this->grid_x(xl); 
            std::size_t ixh = this->grid_x(xh) + 1; 
            std::size_t iyl = this->grid_y(yl); 
            std::size_t iyh = this->grid_y(yh) + 1; 

            for (std::size_t ix = ixl; ix < ixh; ++ix)
            {
                for (std::size_t iy = iyl; iy < iyh; ++iy)
                {
                    if (this->base_type::overlap(ix, iy, xl, yl, xh, yh))
                    {
                        this->set(ix, iy, 1); 
                    }
                }
            }
        }
        
    protected:
        std::vector<unsigned char> m_map; ///< 2D map indicating whether a grid is taken or not 
};

/// @brief A greedy macro legalization algorithm manipulating on Hannan grids. 
/// The procedure of the algorithm is as follows. 
/// For each macro: 
///     Perfrom spiral/diamond search to the locations; 
///     Find the first one with minimum displacement; 
///     Update the grid map; 
/// If the layout is very tight, it may not be able to find a solution. 
/// @return true if all macros legalized 
template <typename T>
bool hannanLegalizeLauncher(LegalizationDB<T> db, std::vector<int>& macros, const std::vector<int>& fixed_macros, int max_iters)
{
    dreamplacePrint(kINFO, "Legalize movable macros on Hannan grids\n");

    // count number of failures to control the order 
    std::vector<int> failure_counts (db.num_movable_nodes, 0);
    std::vector<T> x (db.num_movable_nodes, 0);
    std::vector<T> y (db.num_movable_nodes, 0);
    bool legal = true; 

    for (int iter = 0; iter < max_iters; ++iter)
    {
        dreamplacePrint(kINFO, "round %d\n", iter);
        // copy location to working array 
        for (auto node_id : macros)
        {
            x[node_id] = db.x[node_id];
            y[node_id] = db.y[node_id];
        }
        // sort from left to right, large to small 
        std::sort(macros.begin(), macros.end(), 
                [&](int node_id1, int node_id2){
                    int factor1 = (1 + failure_counts[node_id1]);
                    int factor2 = (1 + failure_counts[node_id2]);
                    T a1 = db.node_size_x[node_id1]*db.node_size_y[node_id1]; // * factor1;
                    T a2 = db.node_size_x[node_id2]*db.node_size_y[node_id2]; // * factor2;
                    T x1 = x[node_id1] / factor1;
                    T x2 = x[node_id2] / factor2; 
                    T y1 = y[node_id1] / factor1;
                    T y2 = y[node_id2] / factor2; 
                    //return a1 > a2 || (a1 == a2 && (x1 < x2 || (x1 == x2 && (y1 < y2 || (y1 == y2 && node_id1 < node_id2)))));
                    //return x1 < x2 || (x1 == x2 && (a1 > a2 || (a1 == a2 && (y1 < y2 || (y1 == y2 && node_id1 < node_id2)))));
                    return x1 < x2 || (x1 == x2 && (y1 < y2 || (y1 == y2 && (a1 > a2 || (a1 == a2 && node_id1 < node_id2)))));
                });

        T spacing_x = std::numeric_limits<T>::max();
        T spacing_y = std::numeric_limits<T>::max();
        for (auto node_id : macros)
        {
            spacing_x = std::min(spacing_x, db.node_size_x[node_id]);
            spacing_y = std::min(spacing_y, db.node_size_y[node_id]);
        }
        // make sure the grid is not too small 
        spacing_x = std::max(spacing_x, (db.xh-db.xl)/db.num_bins_x); 
        spacing_y = std::max(spacing_y, (db.yh-db.yl)/db.num_bins_y);
        dreamplacePrint(kDEBUG, "maximum grid spacing %gx%g, equivalent to %dx%d bins\n", 
                (double)spacing_x, (double)spacing_y, (int)((db.xh-db.xl)/spacing_x), (int)((db.yh-db.yl)/spacing_y));

        // construct hannan grid map for fixed macros 
        // collect fixed and dummy fixed nodes 
        std::vector<T> vx; 
        std::vector<T> vy; 
        std::vector<T> node_size_x; 
        std::vector<T> node_size_y; 
        vx.reserve(db.num_nodes); 
        vy.reserve(db.num_nodes); 
        node_size_x.reserve(db.num_nodes);
        node_size_y.reserve(db.num_nodes);
        for (auto node_id : fixed_macros)
        {
            vx.push_back(db.x[node_id]); 
            vy.push_back(db.y[node_id]); 
            node_size_x.push_back(db.node_size_x[node_id]); 
            node_size_y.push_back(db.node_size_y[node_id]); 
        }
        for (auto node_id : macros)
        {
            vx.push_back(x[node_id]); 
            vy.push_back(y[node_id]); 
            node_size_x.push_back(db.node_size_x[node_id]); 
            node_size_y.push_back(db.node_size_y[node_id]); 
        }

        HannanGridMap<T> grid_map (vx.data(), vy.data(), 
                node_size_x.data(), node_size_y.data(), vx.size(), 
                db.xl, db.yl, db.xh, db.yh, 
                spacing_x, spacing_y);

        // the right and top boundary should always be occupied 
        for (std::size_t ix = 0; ix < grid_map.dim_x(); ++ix)
        {
            grid_map.set(ix, grid_map.dim_y()-1, 1);
        }
        for (std::size_t iy = 0; iy < grid_map.dim_y(); ++iy)
        {
            grid_map.set(grid_map.dim_x()-1, iy, 1);
        }
        // set fixed nodes to occupy the grid map 
        for (auto node_id : fixed_macros)
        {
            T xl = db.init_x[node_id];
            T xh = xl + db.node_size_x[node_id]; 
            T yl = db.init_y[node_id];
            T yh = yl + db.node_size_y[node_id];
            std::size_t ixl = grid_map.grid_x(xl); 
            std::size_t ixh = grid_map.grid_x(xh); 
            std::size_t iyl = grid_map.grid_y(yl); 
            std::size_t iyh = grid_map.grid_y(yh); 

            for (std::size_t ix = ixl; ix <= ixh; ++ix)
            {
                for (std::size_t iy = iyl; iy <= iyh; ++iy)
                {
                    if (grid_map.HannanGrids<T>::overlap(ix, iy, xl, yl, xh, yh))
                    {
                        grid_map.set(ix, iy, 1); 
                    }
                }
            }
        }

        auto search_grids = diamond_search_sequence(grid_map.dim_y(), grid_map.dim_x()); 
        dreamplacePrint(kDEBUG, "Construct %lux%lu Hannan grids, diamond search sequence %lu\n", grid_map.dim_x(), grid_map.dim_y(), search_grids.size());

        legal = true; 
        for (auto node_id : macros)
        {
            T node_x = x[node_id];
            T node_y = y[node_id];
            T width = db.node_size_x[node_id];
            T height = db.node_size_y[node_id];
            std::size_t init_ix = grid_map.grid_x(node_x);
            std::size_t init_iy = grid_map.grid_y(node_y);

            bool found = false; 
            for (auto grid_offset : search_grids)
            {
                std::size_t ix = init_ix + grid_offset.ic;
                std::size_t iy = init_iy + grid_offset.ir;

                // valid grid 
                if (ix < grid_map.dim_x() && iy < grid_map.dim_y())
                {
                    T xl = grid_map.coord_x(ix);
                    T yl = grid_map.coord_y(iy);
                    if (grid_offset.ic == 0 && grid_offset.ir == 0)
                    {
                        dreamplaceAssertMsg(xl == node_x, "%g != %g", xl, node_x);
                        dreamplaceAssertMsg(yl == node_y, "%g != %g", yl, node_y);
                    }

                    // make sure the coordinates are aligned to row and site 
                    T aligned_xl = db.align2site(xl, width);
                    T aligned_yl = db.align2row(yl, height);
                    if (aligned_xl < xl)
                    {
                        xl = aligned_xl+db.site_width;
                    }
                    if (aligned_yl < yl)
                    {
                        yl = aligned_yl+db.row_height;
                    }
                    T xh = xl + width;
                    T yh = yl + height; 

                    if (!grid_map.overlap(xl, yl, xh, yh))
                    {
                        x[node_id] = xl; 
                        y[node_id] = yl; 
                        grid_map.add(xl, yl, xh, yh);
                        found = true; 
                        break; 
                    }
                }
            }
            if (!found)
            {
                dreamplacePrint(kERROR, "failed to find legal position for macro %d (%g, %g, %g, %g)\n", 
                        node_id, node_x, node_y, node_x + width, node_y + height
                        );
                failure_counts[node_id] += 1; 
                legal = false; 
            }
        }
        if (legal)
        {
            break; 
        }
    }

    // copy solutions back 
    for (auto node_id : macros)
    {
        db.x[node_id] = x[node_id];
        db.y[node_id] = y[node_id];
    }

    return legal; 
}

DREAMPLACE_END_NAMESPACE

#endif
