# eXtreme
## Fast-memory Efficient Extreme Events Prediction in Complex Time series [PDF](https://doi.org/10.1145/3402597.3402609 "Downdoald the paper from here")

## Abstract
_This paper proposes a generic memory-efficient framework for realtime stochastic extreme events prediction in complex time series systems such as intrusion detection, Internet of Things (IoT), social networks, stock markets etc. Ideally we exploit the expressiveness of deep neural networks and temporal nature of sequence-to-sequence structures (parallel Convolutional and recurrent neural networks) glued on Convolutional Quantile Loss and memory network to model explicitly extreme events. Convolutional Quantile Loss is used to predict future extreme events, while memory network is used to memorize extreme events in future observations. We show that the approach can capture long and short-term temporal effects as well as other non-linear dynamic patterns across multiple probabilistic time series with reliable principled uncertainty estimates. We demonstrate and validate empirically the effectiveness of the proposed framework via extensive experiments and rigorous evaluation on large-scale real world datasets. The experimental results showcase that the proposed method is fast, robust, accurate and has superior performance compared to the well-known prediction methods._

## baseTest.py 
This file contains extreme implemetation algorithm together with other algos 

## Datasets
The dataset description is contained in dataset directory

## Dependencies
1. Tensoflow 2
2. Keras 2.3.1
3. Python 3.6
4. Matplotlib 3.1.2
5. tqdm 4.44.1


# Reference
If you find this code useful in your research, please, consider citing our paper:

```
  @inproceedings{wamburaicrsa2020,
	AUTHOR = "Stephen Wambura and He Li and Eyou Niggusie", 
	TITLE = "Fast-memory Efficient Extreme Events Prediction in Complex Time series",
	publisher = {Association for Computing Machinery},
	booktitle  = {ICRSA 2020: Proceedings of the 2020 3rd International Conference on Robot Systems and Applications},
	address   = {Chengdu, China},
	month  = {14-16 June},
	pages  = {60-69},
	YEAR = {2020},
}
```
# License
OFA is distributed under Apache 2.0 license.

Contact: Stephen Wambura (stephen.wambura@dit.ac.tz)
