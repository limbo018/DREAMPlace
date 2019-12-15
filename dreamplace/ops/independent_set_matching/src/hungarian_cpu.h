/**
 * @file   hungarian_cpu.h
 * @author Yibo Lin
 * @date   Jan 2019
 */
#ifndef _DREAMPLACE_GLOBAL_MOVE_HUNGARIAN_CPU_H
#define _DREAMPLACE_GLOBAL_MOVE_HUNGARIAN_CPU_H

#include "munkres.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
class HungarianAlgorithmCPULauncher
{
    public:
        const char* name() const 
        {
            return "HungarianAlgorithmCPULauncher";
        }

        /// @brief solve assignment problem with Hungarian algorithm 
        /// @param cost a nxn row-major cost matrix 
        /// @param sol solution mapping from row to column 
        /// @param n dimension 
        /// @param skip_threshold if the weight is larger than the threshold, do not add the edge 
        T run(const T* cost, int* sol, int n, T skip_threshold = std::numeric_limits<T>::max())
        {
            m_matrix.resize(n, n);
            // Initialize matrix.
            for ( int row = 0 ; row < n ; row++ ) 
            {
                for ( int col = 0 ; col < n ; col++ ) 
                {
                    m_matrix(row,col) = cost[n*row+col];
                }
            }

            // Apply Munkres algorithm to matrix.
            Munkres<T> solver; 
            solver.solve(m_matrix);

            // Get solution and display objective.
            T total_cost = 0; 
            for ( int row = 0 ; row < n ; row++ ) 
            {
                for ( int col = 0 ; col < n ; col++ ) 
                {
                    if (m_matrix(row, col) == 0)
                    {
                        sol[row] = col; 
                        total_cost += cost[row*n+col]; 
                    }
                }
            }

            return total_cost; 
        }
    protected:
        Matrix<T> m_matrix; 
};

DREAMPLACE_END_NAMESPACE

#endif
