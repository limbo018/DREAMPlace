/**
 * @file   auction_cpu.h
 * @author Jiaqi Gu, Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_GLOBAL_MOVE_AUCTION_CPU_H
#define _DREAMPLACE_GLOBAL_MOVE_AUCTION_CPU_H

#include <iostream>
#include <cstring>
#include <cassert>

DREAMPLACE_BEGIN_NAMESPACE

#define AUCTION_MAX_EPS 10.0 // Larger values mean solution is more approximate
#define AUCTION_MIN_EPS 1.0
#define AUCTION_FACTOR  0.1
#define AUCTION_MAX_ITERS 9999
#define BIG_NEGATIVE -9999999

/// @return 1 if found a solution 
template <typename T>
int run_auction(
    int    num_nodes,
    T* data_ptr,      // data, num_nodes*num_nodes in row-major  
    int*   person2item_ptr, // results
    
    float auction_max_eps,
    float auction_min_eps,
    float auction_factor, 
    int auction_max_iters, 
    int* item2person_ptr=nullptr, 
    T* bids_ptr=nullptr, 
    T* prices_ptr=nullptr, 
    int* sbids_ptr=nullptr
)
{
    
    // --
    // Declare variables
    
    bool allocate_flag = false; 
    if (!item2person_ptr)
    {
        item2person_ptr = (int*)malloc(num_nodes * sizeof(int));
        bids_ptr      = (T*)malloc(num_nodes * num_nodes * sizeof(T));
        prices_ptr    = (T*)malloc(num_nodes * sizeof(T));
        //bidders_ptr     = (int*)malloc(num_nodes * num_nodes * sizeof(int)); // unused
        sbids_ptr       = (int*)malloc(num_nodes * sizeof(int));
        allocate_flag = true; 
    }

    int   *data           = data_ptr;
    int   *person2item    = person2item_ptr;
    int   *item2person    = item2person_ptr;
    int   *prices         = prices_ptr;
    int   *sbids          = sbids_ptr;
    int   *bids           = bids_ptr;
    int  num_assigned = 0; 

    for(int i = 0; i < num_nodes; i++) {
        prices[i] = 0.0;
        person2item[i] = -1;
    }

    // Start timer


    float auction_eps = auction_max_eps;
    int counter = 0;
    while(auction_eps >= auction_min_eps && counter < auction_max_iters) {
        for(int i = 0; i < num_nodes; i++) {
            person2item[i] = -1;
            item2person[i] = -1;
        }
        num_assigned = 0;


        while(num_assigned < num_nodes && counter < auction_max_iters){
            counter += 1;

            std::memset(bids, 0, num_nodes * num_nodes * sizeof(T));
            std::memset(sbids, 0, num_nodes * sizeof(int));

            // #pragma omp parallel for num_threads(1)
            for(int i = 0; i < num_nodes; i++) {
                if(person2item[i] == -1) {
                    T top1_val = BIG_NEGATIVE; 
                    T top2_val = BIG_NEGATIVE; 
                    int top1_col = BIG_NEGATIVE; 
                    T tmp_val = BIG_NEGATIVE;

                    for (int col = 0; col < num_nodes; col++)
                    {
                        tmp_val = data[i * num_nodes + col]; 
                        if (tmp_val < 0)
                        {
                            continue;
                        }
                        tmp_val = tmp_val - prices[col];
                        if (tmp_val >= top1_val)
                        {
                            top2_val = top1_val;
                            top1_col = col;
                            top1_val = tmp_val;
                        }
                        else if (tmp_val > top2_val)
                        {
                            top2_val = tmp_val;
                        }
                    }
                    if (top2_val == BIG_NEGATIVE)
                    {
                        top2_val = top1_val;
                    }
                    T bid = top1_val - top2_val + auction_eps;
                    bids[num_nodes * top1_col + i] = bid;
                    sbids[top1_col] = 1; 
                }
            }

            // #pragma omp parallel for num_threads(1)
            for(int j = 0; j < num_nodes; j++) {
                if(sbids[j] != 0) {
                    T high_bid  = 0.0;
                    int high_bidder = -1;

                    T tmp_bid = -1;
                    for(int i = 0; i < num_nodes; i++){
                        tmp_bid = bids[num_nodes * j + i]; 
                        if(tmp_bid > high_bid){
                            high_bid    = tmp_bid;
                            high_bidder = i;
                        }
                    }
                    int current_person = item2person[j];
                    if(current_person >= 0){
                        person2item[current_person] = -1; 
                    } else {
                        // #pragma omp atomic
                        num_assigned++;
                    }

                    prices[j]                += high_bid;
                    person2item[high_bidder] = j;
                    item2person[j]           = high_bidder;
                }
            }
        }

        auction_eps *= auction_factor;
    } 

    // //Print results
    // int score = 0;
    // for (int i = 0; i < num_nodes; i++) {
    //     std::cout << i << " " << person2item[i] << std::endl;
    //     score += data[i * num_nodes + person2item[i]];
    // }
    // std::cerr << "score=" <<score << std::endl;   

    if (allocate_flag)
    {
        free(item2person_ptr); 
        free(bids_ptr);
        free(prices_ptr);  
        //free(bidders_ptr); 
        free(sbids_ptr); 
    }

    return (num_assigned >= num_nodes);
} // end run_auction

template <typename T>
class AuctionAlgorithmCPULauncher
{
    public:
        const char* name() const 
        {
            return "AuctionAlgorithmCPULauncher";
        }
        /// @brief solve assignment problem with auction algorithm 
        /// The matrix is converted to non-negative matrix with maximization 
        /// Skipped edges are assigned with BIG_NEGATIVE
        /// @param cost a nxn row-major cost matrix 
        /// @param sol solution mapping from row to column 
        /// @param n dimension 
        /// @param skip_threshold if the weight is larger than the threshold, do not add the edge 
        /// @param minimize_flag true for minimization problem and false or maximization 
        T run(const T* cost, int* sol, int n, T skip_threshold = std::numeric_limits<T>::max(), 
                int minimize_flag = 1, 
                T auction_max_eps=AUCTION_MAX_EPS,
                T auction_min_eps=AUCTION_MIN_EPS, 
                float auction_factor=AUCTION_FACTOR, 
                int auction_max_iters=AUCTION_MAX_ITERS
                )
        {
            int nn = n*n; 

            m_matrix.resize(nn); 
            m_item2person.resize(n); 
            m_bids.resize(nn); 
            m_prices.resize(n);
            //m_bidders.resize(nn); 
            m_sbids.resize(n); 

            if (minimize_flag)
            {
                T max_cost = 0; 
                for (int i = 0; i < nn; ++i)
                {
                    T c = cost[i];
                    if (c < skip_threshold)
                    {
                        max_cost = std::max(max_cost, c); 
                    }
                }
                for ( int row = 0 ; row < n ; row++ ) 
                {
                    for ( int col = 0 ; col < n ; col++ ) 
                    {
                        int idx = n*row+col;
                        T c = cost[idx]; 
                        m_matrix[idx] = (c < skip_threshold)? max_cost-c : BIG_NEGATIVE;
                    }
                }
            }
            else 
            {
                std::copy(cost, cost+nn, m_matrix.data()); 
            }

            int ret = run_auction<T>(
                    n,
                    m_matrix.data(),
                    sol,
                    auction_max_eps,
                    auction_min_eps,
                    auction_factor, 
                    auction_max_iters, 
                    m_item2person.data(), 
                    m_bids.data(), 
                    m_prices.data(), 
                    m_sbids.data()
                    );

            // Get solution and display objective.
            T total_cost = 0; 
            if (ret)
            {
                for ( int row = 0 ; row < n ; row++ ) 
                {
                    int col = sol[row]; 
                    total_cost += cost[n*row+col]; 
                }
            }
            else // not found a solution and terminated early 
            {
                total_cost = std::numeric_limits<T>::max(); 
                for (int row = 0; row < n; ++row)
                {
                    sol[row] = row; 
                }
            }

            return total_cost; 
        }

    protected:

        std::vector<T> m_matrix; 
        std::vector<int> m_item2person; 
        std::vector<T> m_bids;
        std::vector<T> m_prices;
        std::vector<int> m_sbids;
};

DREAMPLACE_END_NAMESPACE

#endif
