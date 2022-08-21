# HiSA: Hierarchically Semantic Associating for Video Temporal Grounding

This is the pytorch implementation of HiSA that is published in IEEE TIP.

## Introduction
Video Temporal Grounding (VTG) aims to locate the time interval in a video that is semantically relevant to a language query. Existing VTG methods interact the query with entangled video features and treat the instances in a dataset independently. However, intra-video entanglement and inter-video connection are rarely considered in these methods, leading to mismatches between the video and language. To this end, we propose a novel method, dubbed Hierarchically Semantic Associating (HiSA), which aims to precisely align the video with language and obtain discriminative representation for further location regression. Specifically, the action factors and background factors are disentangled from adjacent video segments, enforcing precise multimodal interaction and alleviating the intra-video entanglement.In addition, cross-guided contrast is elaborately framed to capture the inter-video connection, which benefits the multimodal understanding to locate the time interval. Extensive experiments on three benchmark datasets demonstrate that our approach significantly outperforms the state-of-the-art methods. <https://ieeexplore.ieee.org/abstract/document/9846867>

## Requirments
Python 3.8.12  
Pytorch 1.4.0

## Acknowledgements
Our work is inspired from many recent efforts. They are:  
[ACRM](https://github.com/tanghaoyu258/ACRM-for-moment-retrieval)  
[TMLGA](https://github.com/crodriguezo/TMLGA)  
Many thanks for their work

## Citation
If you find our work helpful, please consider citing our [paper](https://ieeexplore.ieee.org/abstract/document/9846867):  
@ARTICLE{9846867,  
    author={Xu, Zhe and Chen, Da and Wei, Kun and Deng, Cheng and Xue, Hui},  
    journal={IEEE Transactions on Image Processing},   
  title={HiSA: Hierarchically Semantic Associating for Video Temporal Grounding},   
  year={2022},  
  volume={31},  
  number={},  
  pages={5178-5188},  
  doi={10.1109/TIP.2022.3191841}}  

