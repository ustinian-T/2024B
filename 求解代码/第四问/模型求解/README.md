# 2024 高教社杯 B题：问题四 Python 完整实现

> 技术路线：抽样统计不确定性 → Q2/Q3 精确评价器 → 90%/95% 同时置信上界 → 稳健重优化 → 退化/覆盖/交叉/单调性/Bootstrap 检验 → 灵敏度分析。

## 1. 模型口径

本项目严格按《问题四建模手册》实现，不用人为 `n=100/10000`、Beta(1,1)、统一 ±5% 等无来源参数充当题目数据。

1. **统计层**
   - 单件质量结果按 Bernoulli 处理，累计次品数按 Binomial 处理。
   - `fixed`：Clopper–Pearson 精确单侧上置信界。
   - `sequential`：反演问题一相同的单侧混合似然比 e-process/confidence sequence；随机停止记录不会误当固定样本。
2. **系统同时置信**
   - Q2 一个情形含 3 个质量参数；Q3 含 12 个质量参数。
   - 用 Bonferroni/union bound：每个参数置信度 `1-(1-gamma)/d`，构造 90%/95% 系统同时上界。
3. **Q2 内层模型**
   - 信息保持型 Bellman–Markov 暂态系统；显式保存 K/UG/UB 状态。
   - 拆解后已知合格件保持 K；未知回收件按回收检测规则处理。
   - 用 `rho(Q)<1` 检查吸收性。
   - 静态策略完整搜索；无拆解时与几何分布闭式公式交叉核验。
4. **Q3 内层模型**
   - BOM 分层状态压缩递推；父节点失败后用条件概率更新未知子节点质量风险。
   - 题目具体实例通过 NumPy 根节点向量化，精确覆盖 `2^16=65536` 个静态策略。
   - 名义、90%、95% 关键向量可再用独立平坦 65536 枚举交叉核验。
5. **稳健层**
   - 先对所有静态策略做质量参数单调性审计（zero-others 与 upper-others 两类锚点）。
   - 审计通过：按理论单调性使用同时上界向量作为最坏点，并对全部策略重新优化。
   - 审计失败：Q2 自动退回 8 个盒角点搜索；Q3 自动退回 4096 个盒角点 × 65536 策略的端点稳健搜索，禁止静默沿用上界捷径。
6. **模型检验**
   - Q2 六情形名义回归。
   - Q2 `z=0` 的 Bellman 与闭式几何期望一致性。
   - Q3 分层递推与独立 65536 平坦枚举一致性。
   - Q3 最优策略成本分解手算回归。
   - 固定样本 Clopper–Pearson **精确覆盖率**。
   - Q1 序贯停止边界回归 + 完整 Bernoulli 流的 sequential confidence-sequence 覆盖率 Monte Carlo 检验。
   - 最坏点单调性审计。
   - 同原抽样设计 parametric bootstrap 稳定性检验。
7. **灵敏度分析**
   - 使用题目自身出现的范围：次品率 5%–20%、零件检测费 1–8 元、调换损失 6–40 元、拆解费 5–40 元。
   - 输出最优/次优利润间隔 `Delta`，不只报告“最优方案”。
   - 有真实固定样本后做信息充分性投影：`n,2n,4n,...` 只作为实验设计变量，并明确假定未来样本比例保持当前 `p_hat`。
   - 有 Q3 真实抽样后输出 `p_F: p_hat -> U95` × `L:6->40` 的 Q4 专属二维相图数据。

## 2. 为什么默认不生成“唯一 Q4 稳健数值”

题目只告诉我们表中的次品率由抽样得到，却没有给每个质量参数真实的 `sample_design,n,k`。因此默认运行会：

- 完整跑通 Q2/Q3 名义回归与模型检验；
- 输出不依赖真实 `n,k` 的覆盖率和经济参数灵敏度；
- 自动生成 `data/sampling_records_template.csv`；
- 若没有 `data/sampling_records.csv`，明确跳过 Q4 的 90%/95% 唯一稳健数值。

这不是代码缺失，而是避免凭空补数据。把真实抽样记录填入模板后，同一程序会自动进入完整 Q4 稳健重优化。

## 3. 安装与运行

```bash
pip install -r requirements.txt
python main.py
```

填入真实抽样记录：

```bash
cp data/sampling_records_template.csv data/sampling_records.csv
# 填 sample_design / n / k / batch_id；若是 sequential，还填 seq_reference_p0
python main.py --sampling data/sampling_records.csv
```

常用选项：

```bash
# 只关闭 Bootstrap；其它检验仍执行
python main.py --skip-bootstrap

# Bootstrap 按 64→128→256→512 自适应检查稳定；这两个数只是数值控制，不是模型参数
python main.py --bootstrap-start 64 --bootstrap-max 512

# 调试时可关闭独立平坦 65536 交叉检查；正式结果不建议关闭
python main.py --no-flat-cross-check
```

## 4. sampling_records 字段

- `quality_id`：唯一质量参数 ID，如 `Q2-S1-C1`、`Q3-H2`。
- `quality_type`：`raw_component` / `process`。
- `sample_design`：`fixed` / `sequential`。
- `n,k`：真实停止时的样本量和次品数。
- `input_certified`：工序次品率必须为 True，即抽样时所有直接输入都已确认合格。
- `batch_id`：批次标识，便于审计。
- `seq_reference_p0`：只有 sequential 的同设计 Bootstrap 需要，必须与问题一实际停止规则一致。
- `reported_rate`：题目表中的点估计，仅用于核查；不会覆盖 `k/n`。

## 5. 主要输出

无真实 sampling_records 时：

- `regression_q2.csv`
- `validation_q2_closed_form.csv`
- `regression_q3.csv`
- `validation_q3_handcheck.csv`
- `validation_cp_coverage.csv`
- `validation_q1_sequential_boundaries.csv`
- `validation_q1_sequential_coverage.csv`
- `sensitivity_q2_nominal.csv`
- `sensitivity_q3_nominal.csv`
- `q4_phase_map_nominal.csv`
- `run_summary.json`

有真实 sampling_records 后进一步生成：

- `quality_intervals.csv`
- `q4_q2_policy.csv`
- `q4_q3_policy.csv`
- `q4_topk.csv`
- `q4_bootstrap.csv`
- `q4_information_curve.csv`
- `q4_phase_map.csv`
- `validation_q2_monotonicity.csv`
- `validation_q3_monotonicity.csv`

MATLAB 后续只读这些 CSV 画图，不重新实现求解公式。

## 6. 已执行的程序级验收

项目打包前已执行：

```bash
python -m compileall -q .
pytest -q
python main.py
```

并额外用**仅用于软件路径测试、绝不作为题目结果**的临时 fixed-sample fixture 跑通完整 Q4 分支（包括90%/95%稳健重优化、单调性审计、信息充分性和 Bootstrap），确认所有输出文件可正常产生。临时 fixture 不包含在项目数据目录中。
