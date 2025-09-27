/**
 * File              : adj_matrix.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 12.18.2024
 * Last Modified Date: 12.18.2024
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void adjMatrixLauncher(const int* flat_net2pin_map, const int* flat_net2pin_start_map, 
    const int* pin2node_map, const T* net_weights, const unsigned char* net_mask, 
    int num_nodes, int num_nets, 
    std::vector<int>& row, std::vector<int>& col, std::vector<T>& data);

/// @brief Given netlist and net weights, build adjacency matrix in COO format
//std::tuple<std::vector<float>, std::vector<int>, std::vector<int>> 
std::vector<at::Tensor> 
adj_matrix_forward(at::Tensor flat_net2pin_map,
                           at::Tensor flat_net2pin_start_map, 
                           at::Tensor pin2node_map, 
                           at::Tensor net_weights, 
                           at::Tensor net_mask, 
                           int num_nodes) {
  CHECK_FLAT_CPU(flat_net2pin_map);
  CHECK_CONTIGUOUS(flat_net2pin_map);
  CHECK_FLAT_CPU(flat_net2pin_start_map);
  CHECK_CONTIGUOUS(flat_net2pin_start_map);
  CHECK_FLAT_CPU(pin2node_map);
  CHECK_CONTIGUOUS(pin2node_map);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);

  int num_nets = flat_net2pin_start_map.numel() - 1; 
  at::Tensor data_tensor;
  at::Tensor row_tensor;
  at::Tensor col_tensor;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      net_weights, "adjMatrixLauncher", [&] {
      // COO format of sparse matrix 
      std::vector<scalar_t> data; 
      std::vector<int> row; 
      std::vector<int> col; 

      adjMatrixLauncher<scalar_t>(
          DREAMPLACE_TENSOR_DATA_PTR(flat_net2pin_map, int),
          DREAMPLACE_TENSOR_DATA_PTR(flat_net2pin_start_map, int),
          DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
          DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
          DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
          num_nodes, num_nets, 
          row, col, data
          );

      data_tensor = torch::from_blob(data.data(), {(long)data.size()}, net_weights.options()).clone();
      row_tensor = torch::from_blob(row.data(), {(long)row.size()}, flat_net2pin_map.options()).clone();
      col_tensor = torch::from_blob(col.data(), {(long)col.size()}, flat_net2pin_map.options()).clone();

      //return std::make_tuple(data, row, col);
      });

  std::vector<at::Tensor> ret ({data_tensor, row_tensor, col_tensor});

  return ret; 
}


/// @brief Build adjacency matrix using clique model 
template <typename T>
void adjMatrixLauncher(const int* flat_net2pin_map, const int* flat_net2pin_start_map, 
    const int* pin2node_map, const T* net_weights, const unsigned char* net_mask, 
    int num_nodes, int num_nets, 
    std::vector<int>& row, std::vector<int>& col, std::vector<T>& data) {

  for (int i = 0; i < num_nets; ++i) {
    if (net_mask && !net_mask[i]) 
      continue; 
    int bgn = flat_net2pin_start_map[i]; 
    int end = flat_net2pin_start_map[i+1]; 
    int degree = end - bgn; 
    T weight = 1; 
    if (net_weights) 
      weight = net_weights[i]; 
    weight /= std::max(degree - 1, 1);

    for (int j = bgn; j < end; ++j) { // clique model 
      int p1 = flat_net2pin_map[j]; 
      int node1 = pin2node_map[p1]; 
      for (int k = j+1; k < end; ++k) {
        int p2 = flat_net2pin_map[k]; 
        int node2 = pin2node_map[p2]; 
        data.push_back(weight);
        row.push_back(node2);
        col.push_back(node1); 
      }
    }
  }
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adj_matrix_forward", &DREAMPLACE_NAMESPACE::adj_matrix_forward, "AdjMatrix forward");
}
