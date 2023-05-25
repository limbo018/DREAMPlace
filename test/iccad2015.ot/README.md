* Caution about TNS and WNS reported by DREAMPlace

DREAMPlace leverages OpenTimer for static timing analysis, while the official evaluation script of [ICCAD 2015 contest](http://iccad-contest.org/2015/CAD-contest-at-ICCAD2015/index.html) adopts UI-Timer (the former development of OpenTimer). OpenTimer and UI-Timer have slightly different ways to compute TNS and WNS in early/late modes, so the TNS and WNS reported by DREAMPlace are slightly different from that reported by the official evaluation script. In the DREAMPlace 4.0 papers, we used the official evaluation script to report TNS and WNS in the result tables. 

* Peiyu Liao, Siting Liu, Zhitang Chen, Wenlong Lv, [Yibo Lin](http://yibolin.com) and [Bei Yu](https://www.cse.cuhk.edu.hk/~byu/), 
  "**DREAMPlace 4.0: Timing-driven Global Placement with Momentum-based Net Weighting**", 
  IEEE/ACM Proceedings Design, Automation and Test in Eurpoe (DATE), Antwerp, Belgium, Mar 14-23, 2022
  ([preprint](https://yibolin.com/publications/papers/PLACE_DATE2022_Liao.pdf))

* Peiyu Liao, Dawei Guo, [Zizheng Guo](https://guozz.cn/), Siting Liu, Zhitang Chen, Wenlong Lv, [Yibo Lin](http://yibolin.com) and [Bei Yu](https://www.cse.cuhk.edu.hk/~byu/), 
  "**DREAMPlace 4.0: Timing-driven Placement with Momentum-based Net Weighting and Lagrangian-based Refinement**", 
  IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2023
