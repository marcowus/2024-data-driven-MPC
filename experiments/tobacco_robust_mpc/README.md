# Tobacco Robust MPC (Data-driven Linear + Disturbance-tightened MPC)

这个实验把你给的数据预处理流程（Excel 多批次、中文列名、时间列处理、插值清洗）落成一个可直接运行的鲁棒 MPC 脚本。

## 方法概要

1. 从你的批次数据拟合一阶线性模型：
   \(x_{k+1} = A x_k + B u_k + c + w_k\)
2. 用加权岭回归（forgetting factor）估计 \(A,B,c\)。
3. 用残差分位数估计扰动上界 \(\bar w\)。
4. 在 MPC 中做 tube-style 约束收紧：
   \(r_{t+1}=|A|r_t+\bar w\)，并用 \(x\in[x_{min}+r_t, x_{max}-r_t]\) 的收紧区间。
5. 使用 `cvxpy` + `OSQP` 在线滚动优化。

## 运行

```bash
TRAIN_XLS_PATH='/your/path/train.xls' python experiments/tobacco_robust_mpc/run_tobacco_robust_mpc.py
```

## 输出

- `experiments/tobacco_robust_mpc/results/tobacco_robust_mpc_summary.png`
- `experiments/tobacco_robust_mpc/results/tobacco_robust_mpc_metrics.npz`

## 依赖

```bash
pip install pandas numpy matplotlib cvxpy openpyxl seaborn
```
