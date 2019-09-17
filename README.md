# NOTEARS-Tensorflow

This repository is a Tensorflow reimplementation of NOTEARS [1].

## 1. Setup

```
conda create -n notears python=3.6
source activate notears
pip install -r requirements.txt
```

## 2. Training
To run `NoTears`, for example, run:

```
cd src
python main.py  --seed 1230 \
                --d 20 \
                --n 3000 \
                --degree 3 \
                --dataset_type linear \
                --graph_thres 0.3 \
                --l1_graph_penalty 0.0 \
                --max_iter 20 \
                --iter_step 500 \
                --init_iter 3 \
                --learning_rate 1e-3 \
                --h_tol 1e-10 \
                --rho_thres 1e12 \
                --rho_multiply 10 \
                --h_thres 0.25 \
                --use_float64 True 
```

### Remark
- Some of the code implementation is referred from https://github.com/xunzheng/notears

## References
[1] Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. Dags with no tears: Continuous optimization for structure learning. In Advances in Neural Information Processing Systems 31, pages 9472–9483. Curran Associates, Inc., 2018.

