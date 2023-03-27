# MAMG
This repository contains code for ''Gradient Surgery for Multi-Task Learning".



## Datasets

Our experiments in the paper were based on the following repositories.

CIFAR-100-MTL: [RoutingNetworks](https://github.com/cle-ros/RoutingNetworks)

MultiMNIST: [MultiObjectiveOptimization](https://github.com/intel-isl/MultiObjectiveOptimization)

`CityScapes`、`NYUv2`、Taskonomy: [MTAN](https://github.com/sunxm2357/AdaShare)

MT10/MT50/goal-conditioned pushing in [MetaWorld](https://meta-world.github.io/): [softlearning](https://github.com/rail-berkeley/softlearning) with modifications (per-task temperature and per-task replay buffers). We will release modified multi-task softlearning code soon.


## Reference

If you find our approach useful in your research, please consider citing:

```
@article{chai2022model,
  title={A model-agnostic approach to mitigate gradient interference for multi-task learning},
  author={Chai, Heyan and Yin, Zhe and Ding, Ye and Liu, Li and Fang, Binxing and Liao, Qing},
  journal={IEEE Transactions on Cybernetics},
  year={2022},
  publisher={IEEE}
}
```
