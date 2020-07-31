# NOTEARS-Tensorflow

This repository is a Tensorflow reimplementation of NOTEARS [1].

## 1. Setup

```
pip install -r requirements.txt
```

## 2. Training
To run `NoTears`, for example, run:

```
python main.py  --seed 1230 \
                --d 20 \
                --n 1000 \
                --degree 4 \
                --graph_thres 0.3 \
                --l1_graph_penalty 0.1
```

### Remark
- Some of the code implementation is referred from https://github.com/xunzheng/notears

## References
[1] Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. DAGs with NO TEARS: Continuous optimization for structure learning. In Advances in Neural Information Processing Systems, 2018.