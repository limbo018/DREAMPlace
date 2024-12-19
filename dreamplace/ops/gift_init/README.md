## Flow
1. Transform the netlist to a sparse adjacency matrix and save it to .npz file.
   1. The paper use clique model. In fact, other models may also work.
2. Generate random initial locations of cells.
3. Use GiFt to produce optimized cell locations based on multi-resolution graph signals.
4. Save the locations produced by GiFt to def file
5. Feed the def file to DREAMPlace to generate the final legalized placement result.

## Acknowledgement 
This operator is adapted from Dr. Yiting Liu's implementation. 
```
Yiting Liu, Hai Zhou, Jia Wang, Fan Yang, Xuan Zeng, Li Shang, The Power of Graph Signal Processing for Chip Placement Acceleration, ICCAD 2024. 
```
We sincerely appreciate the contribution from the authors. 
