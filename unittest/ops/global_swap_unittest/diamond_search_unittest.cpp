/**
 * @file   diamond_search_unitest.cpp
 * @author Yibo Lin
 * @date   Jan 2019
 */
#include <iostream>
#include "global_swap/src/diamond_search.h"

template <typename T>
void test()
{
    T num_rows = 21; 
    T num_cols = 21; 
    auto sorted_grids = DreamPlace::diamond_search_sequence(num_rows, num_cols); 
    DreamPlace::diamond_search_print(sorted_grids);
}

int main()
{
    test<unsigned int>();
}
