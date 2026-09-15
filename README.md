# 光谱配色复核 API (Spectral Colour-Match Review)

一块涂料色板在日光（D65）下合格、在商场灯光（A / F11 窄带荧光）下明显偏色，
这是典型的**同色异谱 (metamerism)**：只看一组 Lab 数值无法发现，必须比较
反射率光谱在多个光源下的表现。

本服务基于 CIE 1931 2° 标准观察者，对目标样与试样的波长—反射率序列进行：

1. **共同波长网格插值**（默认 380–780 nm，步长 5–20 nm 可选）；
2. 逐光源计算 **XYZ / Lab / CIEDE2000 (ΔE00)**；
3. 内置 **D65 / A / F11**，也支持上传任意相对功率分布 (SPD)；
4. 当某光源 ΔE00 ≤ 阈值而另一光源超差时，标记 **同色异谱风险**；
5. 列出对 XYZ 色差贡献最大的**波长波段**（可配置 20–100 nm 聚合宽度）；
6. 批量接口对多个试样**按阈值筛选、排序候选**；
7. `/review/uncertainty` 接收目标样/试样各 2–50 条**重复扫描**，用可复现的
   自助法（bootstrap）蒙特卡洛给出 ΔE00 的中位数、2.5%/97.5% 分位数、
   超阈概率，并判定“稳定合格 / 临界 / 稳定超差”。

## 色度学口径

- 积分：公共波长网格上的**梯形积分**；
- 三刺激值：`X = 100 · ∫S(λ)x̄(λ)R(λ)dλ / ∫S(λ)ȳ(λ)dλ`（Y、Z 同理），
  完美反射体 Y = 100；
- **每个光源使用其自身的适应白点**（该 SPD 下完美反射体的 XYZ）。
  否则中性灰在 A 光源下会被错误地算成明显偏黄，A/D65 之间的 Lab 也不可比；
- CIEDE2000 采用 Sharma–Wu–Dalal (2005) 公式，kL = kC = kH = 1；
- 内置数据为 CIE 15:2004 的 5 nm 标准表，已打包在 `app/data/`，运行时无需联网。

整套 XYZ/Lab/ΔE00 管线已与 [colour-science](https://www.colour-science.org/)
独立实现对拍：XYZ 最大偏差 < 0.02，ΔE00 偏差 < 0.002；CIEDE2000 另通过
Sharma 32 组标准数据（见 `tests/de2000_cases.py`）。

## 运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000
# 交互式文档: http://127.0.0.1:8000/docs
```

开发/测试：

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

重新生成内置 CIE 数据（仅在需要刷新标准表时，需联网）：

```bash
.venv/bin/python scripts/build_data.py
```

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 存活检查 |
| GET  | `/illuminants` | 内置光源目录与覆盖范围 |
| POST | `/api/v1/review` | 单对目标样/试样，多光源复核 |
| POST | `/api/v1/review/batch` | 一个目标样 + 多个试样，筛选/排序 |
| POST | `/api/v1/review/uncertainty` | 重复扫描自助法：ΔE00 分布、超阈概率、稳定/临界判定 |

### 请求体关键字段

- `target` / `sample`：`{"wavelengths": [nm...], "values": [反射率 0–1...]}`，
  两个数组等长；波长可乱序（内部排序），但不可重复；
- `illuminants`：`{"name": "D65" | "A" | "F11"}` 或
  `{"custom": {"wavelengths": ..., "values": ...}, "label": "商场LED"}`；
  缺省同时计算 D65、A、F11；
- `tolerance_de00`：合格阈值（ΔE00 ≤ 阈值判合格，默认 2.0）；
- `grid_step_nm`：共同网格步长，5–20 nm（默认 10）；
- `band_width_nm` / `top_bands`：波段归因的聚合宽度与返回条数；
- 批量额外支持 `sort_by`（`max_de00`/`mean_de00`/`name`）、
  `only_passing_all`、`only_metameric`。

### 示例：日光合格、A 光源超差（同色异谱）

```bash
curl -s http://127.0.0.1:8000/api/v1/review \
  -H 'Content-Type: application/json' \
  -d @examples/metameric_pair.json | python3 -m json.tool
```

响应摘要（完整字段见 OpenAPI）：

```jsonc
{
  "results_by_illuminant": [
    {"illuminant": "D65", "delta_e00": 0.0,   "pass": true,
     "lab_target": {"L": 62.3, "a": 17.4, "b": 24.5},
     "top_bands": [{"wavelength_range_nm": [580.0, 610.0], "magnitude": 1.798, ...}]},
    {"illuminant": "A",   "delta_e00": 3.163, "pass": false, ...},
    {"illuminant": "F11", "delta_e00": 1.019, "pass": true,  ...}
  ],
  "metamerism_risk": {
    "flag": true,
    "passes_under": ["D65", "F11"],
    "fails_under":  ["A"],
    "worst_illuminant": "A", "worst_de00": 3.163,
    "best_illuminant": "D65", "best_de00": 0.0,
    "spread_de00": 3.163
  }
}
```

`top_bands[*].magnitude` 是该波段对目标—试样 ΔXYZ 向量的梯形积分贡献范数；
`signed_projection` 为正表示该波段沿总色差方向（主因），为负表示在抵消色差。

### 批量

```bash
curl -s http://127.0.0.1:8000/api/v1/review/batch \
  -H 'Content-Type: application/json' \
  -d @examples/batch_request.json | python3 -m json.tool
```

候选按 `max_de00`（所有光源中最差表现）升序返回；可用
`only_metameric=true` 只保留有同色异谱风险的试样，或
`only_passing_all=true` 只保留全光源合格者。

### 重复测量不确定性：`/api/v1/review/uncertainty`

ΔE00 靠近容差线时，重复测量的波动会让同一批次时而合格、时而超差。
本接口要求目标样与试样各提交 **2–50 条重复反射率扫描**（每条允许使用
不同波长网格），计算流程：

1. 取所有扫描（及自定义光源）波长支撑的**共同区间**，插值到统一网格；
2. 每次蒙特卡洛抽样分别从目标样、试样的完整扫描中**有放回抽取整条
   光谱**——整条抽取保留了同一次扫描内各波长之间的相关性；
3. 同一组配对抽样在所有光源下复用，因此“任一光源超差”的概率是在
   联合事件上统计的，光源间的超差事件保持相关；
4. 逐光源计算每次抽样的 XYZ（沿用梯形积分、各光源自适应白点、
   CIEDE2000），得到 ΔE00 分布。

请求参数：

- `target.scans` / `sample.scans`：`SpectrumInput` 数组，长度 2–50；
- `seed`（0–2³¹−1，默认 42）与 `draws`（100–200000，默认 2000）：
  相同 `seed` + 相同输入得到逐位一致的可复现结果；
- `critical_probability_bound`：判定概率界限 α（0 < α < 0.5，默认 0.05）。

判定规则（设 P = P(ΔE00 > 阈值)）：

| 分类 | 条件 | 含义 |
|------|------|------|
| `stable_pass`（稳定合格） | P ≤ α | 95% 区间整体在容差线内侧 |
| `critical`（临界） | α < P < 1−α | 结果跨容差线，建议复测 |
| `stable_fail`（稳定超差） | P ≥ 1−α | 95% 区间整体在容差线外侧 |

```bash
curl -s http://127.0.0.1:8000/api/v1/review/uncertainty \
  -H 'Content-Type: application/json' \
  -d @examples/uncertainty_request.json | python3 -m json.tool
```

响应逐光源返回 `median_de00`、`q025_de00`、`q975_de00`、
`ci95_width_de00`、`probability_over_tolerance`、`classification` 与
`effective_draws`（实际抽到的不同“目标扫描内容×试样扫描内容”配对数；
提交完全相同的重复扫描时该数会下降）。`summary` 汇总：

```jsonc
{
  "replicates": {"target_scans": 6, "sample_scans": 5,
                 "possible_pairs": 30, "unique_pairs_sampled": 30},
  "results_by_illuminant": [
    {"illuminant": "D65", "median_de00": 0.6381,
     "q025_de00": 0.1158, "q975_de00": 1.6191,
     "probability_over_tolerance": 0.2372, "classification": "critical", ...},
    {"illuminant": "A",   "median_de00": 2.8675,
     "probability_over_tolerance": 1.0, "classification": "stable_fail", ...},
    {"illuminant": "F11", "median_de00": 1.4333,
     "probability_over_tolerance": 0.8972, "classification": "critical", ...}
  ],
  "summary": {
    "probability_over_tolerance_any_illuminant": 1.0,
    "classification_any": "stable_fail",
    "least_stable_illuminant": "F11",
    "least_stable_ci95_width_de00": 1.7161
  }
}
```

“最不稳定光源”取 95% 区间宽度最大的光源；分块计算保证
`draws=200000` 时内存占用仍有上界（约数十 MB）。

## 错误处理（HTTP 422）

所有数据问题一次性收集返回，每条错误都带 `field` / `code` / `reason`，
必要时附定位明细：

| code | 触发条件 | 附加字段 |
|------|----------|----------|
| `length_mismatch` | 波长/数值数组长度不一致 | 两个长度 |
| `non_numeric` / `non_finite` | 非数字、NaN、Inf | 违例值 |
| `duplicate_wavelength` | 重复波长 | `wavelengths_nm` |
| `wavelength_gap` | 测量区间内缺口 > 20 nm | `gap_start_nm`/`gap_end_nm`/`gap_nm` |
| `insufficient_coverage` | 反射率未覆盖 390–770 nm（自定义光源允许部分可见范围，跨度 ≥80 nm） | `range_nm`/`required_range_nm` |
| `too_few_points` | 少于 10 个测点 | `points` |
| `reflectance_out_of_range` | 反射率超出 [0, 1] | `violations`（波长+值） |
| `negative_power` | 自定义光源出现负功率 | `negative_count` |
| `zero_illuminant_power` | 自定义 SPD 全零 | — |
| `zero_normalisation_denominator` | 归一化分母 ∫S(λ)ȳ(λ)dλ ≤ 0（SPD 在可见区无能量） | 指明光源 |
| `unknown_illuminant` | 内置光源名不存在 | `allowed` |
| `illuminant_ambiguous` | 同时/均未提供 `name` 与 `custom` | — |
| `duplicate_illuminant` | 光源重复 | — |
| `invalid_grid_step` / `invalid_wavelength_range` / `range_step_mismatch` | 网格参数非法 | `value`/`allowed_range` |

## 目录结构

```
app/
  colorimetry.py     # CMF/SPD 数据、插值、XYZ、Lab、CIEDE2000
  validation.py      # 带字段定位的输入校验
  engine.py          # 网格对齐、逐光源计算、同色异谱判定、波段归因
  schemas.py         # Pydantic 请求/响应模型
  main.py            # FastAPI 路由与 422 归一化
  data/              # 打包的 CIE 1931 CMF 与 A/D65/F11 SPD（离线）
scripts/build_data.py
tests/               # CIEDE2000 标准数据 + 单元/接口/对拍/不确定性测试
examples/            # curl 示例请求（含 uncertainty_request.json）
```
