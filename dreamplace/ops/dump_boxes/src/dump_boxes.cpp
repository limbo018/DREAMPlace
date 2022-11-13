/**
 * @file   dump_boxes.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute half-perimeter wirelength
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

static unsigned int magic_count = 0; 

void dump_boxes_forward(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y, 
    at::Tensor pin_pos, at::Tensor flat_netpin,
    at::Tensor netpin_start, at::Tensor net_weights, at::Tensor net_mask, 
    double xl, double yl, double xh, double yh, 
    int num_bins_x, int num_bins_y, 
    int num_movable_nodes, int num_terminals) {

  int num_nodes = pos.numel() / 2; 
  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin_pos.numel() / 2; 
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeDumpBoxesLauncher", [&] {
      auto host_node_x = DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t);
      auto host_node_y = host_node_x + num_nodes; 
      auto host_node_size_x = DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t);
      auto host_node_size_y = DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t);
      auto host_pin_x = DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t); 
      auto host_pin_y = host_pin_x + num_pins; 
      auto host_flat_netpin = DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int);
      auto host_netpin_start = DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int); 
      auto host_net_mask = DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char); 
      //auto host_net_weights = DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t); 

      if ((magic_count < 100 && magic_count % 10 == 0) || (magic_count >= 100 && magic_count % 100 == 0)) {
          char filename[64]; 

          {
              sprintf(filename, "cell_boxes%u.txt", magic_count); 
              std::ofstream out (filename); 

              std::cout << "!!!!!!!! write to " << filename << " !!!!!!!!\n";

              out << "diearea " << xl << " " << yl << " " << xh << " " << yh << "\n";
              out << "#bins " << num_bins_x << " " << num_bins_y << "\n";
              out << "#movable " << num_movable_nodes << " #fixed " << num_terminals << " #fillers " << num_nodes - num_movable_nodes - num_terminals << "\n";
              auto bin_size_x = (xh - xl) / num_bins_x; 
              auto bin_size_y = (yh - yl) / num_bins_y; 
              scalar_t sqrt2 = std::sqrt(2.0);
              for (int i = 0; i < num_nodes; ++i) {
                auto cx = host_node_x[i] + host_node_size_x[i] / 2; 
                auto cy = host_node_y[i] + host_node_size_y[i] / 2; 
                auto sx = std::max(host_node_size_x[i], (scalar_t)bin_size_x * sqrt2); 
                auto sy = std::max(host_node_size_y[i], (scalar_t)bin_size_y * sqrt2); 
                auto weight = host_node_size_x[i] * host_node_size_y[i] / (sx * sy);
                out << cx - sx / 2 << " " << cy - sy / 2 << " "
                  << cx + sx / 2 << " " << cy + sy / 2 << " " << weight << "\n";
              }
              out.close();
          }
          {
              sprintf(filename, "net_boxes%u.txt", magic_count); 
              std::ofstream out (filename); 

              std::cout << "!!!!!!!! write to " << filename << " !!!!!!!!\n";

              out << "diearea " << xl << " " << yl << " " << xh << " " << yh << "\n";
              out << "#bins " << num_bins_x << " " << num_bins_y << "\n";
              out << "#nets " << num_nets << "\n"; 
              for (int i = 0; i < num_nets; ++i) {
                auto bxl = std::numeric_limits<scalar_t>::max(); 
                auto byl = std::numeric_limits<scalar_t>::max(); 
                auto bxh = std::numeric_limits<scalar_t>::lowest(); 
                auto byh = std::numeric_limits<scalar_t>::lowest(); 
                for (int bgn = host_netpin_start[i], end = host_netpin_start[i + 1]; bgn < end; ++bgn) {
                  auto pin_id = host_flat_netpin[bgn]; 
                  auto px = host_pin_x[pin_id]; 
                  auto py = host_pin_y[pin_id]; 
                  bxl = std::min(bxl, px); 
                  byl = std::min(byl, py); 
                  bxh = std::max(bxh, px); 
                  byh = std::max(byh, py); 
                }
                out << bxl << " " << byl << " " << bxh << " " << byh << " " << int(host_net_mask[i]) << "\n";
              }
              out.close();
          }

      }
      magic_count += 1; 
  });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::dump_boxes_forward, "DumpBoxes forward");
}
