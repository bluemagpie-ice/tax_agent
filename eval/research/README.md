# Research 评测及数据导出说明

## 运行评测

在项目根目录打开 Windows PowerShell，使用已有的 Python 虚拟环境。分析命令中的 `<experiment-directory>` 需要换成实际生成的实验目录名。

```powershell
if (-not (Test-Path eval/research/config.json)) {
    Copy-Item eval/research/config.example.json eval/research/config.json
}
.\.venv\Scripts\python.exe -m eval.research_eval --config eval/research/config.json --smoke --limit 1
.\.venv\Scripts\python.exe -m eval.research_eval --config eval/research/config.json --live --allow-unverified
.\.venv\Scripts\python.exe -m eval.research_analysis data/research_eval/<experiment-directory>
```

- `--smoke` 运行快速流程检查。脚本模拟模型响应，用于检查数据结构、流程和导出。上面的冒烟测试只选一题。评估税务质量需要真实报告和人工核验。
- `--live` 调用配置好的外部模型，需要 API Key。政策检索使用**固定保存的公开节选**，按关键词检索并回放已保存的官网内容。实验数据与正式 Qdrant、聊天记忆和个人案例库隔离。搜索引擎和官网访问是否正常，需要通过实际 Research 网页另行验收。

政策快照是 2026 年 11 月 5 日读取的部分官方节选。审核时应分别判断来源真实性、当前效力和对具体业务的适用性，并根据需要查阅完整原文。[《中华人民共和国增值税法》](https://fgk.chinatax.gov.cn/zcfgk/c100009/c5237365/content.html) · [财政部 税务总局公告 2023 年第 12 号](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5210453/content.html)

## 分组与实验控制

- **A 组**运行 Research 工作流，关闭历史案例检索。
- **B 组**运行同一工作流，开启历史案例检索与差异比较，并把相关案例线索提供给研究规划。
- **C 组**按固定步骤理解问题、检索政策、按需计算和生成报告，作为标准操作流程（SOP）基线。它与 A、B 组使用相同的模型、工具和政策资料，由独立的固定流程执行器运行。

三组共用报告数据结构、来源及结论检查规则和执行预算。实验输出供独立评测，实验案例只保存在隔离 SQLite 中。测试样例的保存确认由脚本模拟，专业复核状态以实际记录为准。

配置文件可设置实验分组、数据划分、重复次数、执行顺序的随机种子、预算和可选的模型服务单价。要观察重复运行的波动，可将 `repeats` 设为 3 或更多。`schedule_seed` 只控制任务执行顺序。评测未向服务提供方发送模型随机种子，因此 `random_seed` 留空。模型别名指向的版本可能变化，系统会记录服务方实际返回的模型 ID，固定版本需要服务方支持。

每次实验保存数据集、政策节选和案例样例的副本及哈希值，并记录去除敏感信息后的模型参数、研究源码哈希、Python 版本和相关依赖版本。实验配置请保留非敏感参数，API Key 通过项目现有配置提供。有条件时使用对应固定版本的模型 ID。

## 数据状态与泄漏检查

`dataset.json` 包含 18 道评测候选题和 6 道开发调试题。**全部仍待人工税务核验。** 核验状态 `human_validation.status` 为 `pending`，参考判断尚未填写。核验前如需调用真实模型试跑，须显式加上 `--allow-unverified`，将结果作为探索性记录。核验后再更新状态并附上核验依据。

案例家族指业务结构相近的一组案例。当前分组是暂定的，需要检查是否有换了措辞、实质上接近同一道题的样本。模拟案例中还包含故意错误的历史结论、已撤回案例及属于其他用户的案例，用于观察错误继承和访问范围。

公开法规是允许检索的资料。后续制作快照时，应排除或标注可直接泄漏参考答案的公开问答和近似重复材料。导出文件会记录潜在泄漏及案例家族重叠标记。解释涉及时效的实验前，请先核对快照时间、案例可用日期和任务截止日期。当前快照覆盖有限，遇到证据不足时可按实际情况记录。

## 导出文件

- `runs.csv` 每行对应一个任务、一个分组和一次重复运行。使用带 BOM 的 UTF-8 编码，便于用 Pandas、Excel 或 Stata 打开。缺失值留空。
- `runs.jsonl` 每行保存一条完整记录，包含运行信息、任务状态、证据、计划、案例引用、工具输入与结果、模型输出、已返回的用量、错误及执行轨迹。
- `manifest.json` 和输入文件副本保存实验控制条件、哈希值和复现所需的输入，凭据由项目配置单独管理。
- `quality_scores_template.csv` 是独立的人工评分表，评分栏初始为空。
- `summary.md` 是供阅读的探索性运行摘要。
- `business.sqlite` 和 `checkpoints.sqlite` 是本次实验专用的业务与检查点数据库。

输入、输出和总 token 数以模型服务方的 `usage_metadata` 为准。只要有一次调用缺少用量，汇总用量及费用就会留空。按字节估算的执行预算单独记录。填写 `pricing.input_per_million`、`pricing.output_per_million` 和 `pricing.currency` 后，程序才会按每百万 token 的单价估算模型费用。工具侧记录调用次数，外部工具费用需单独核算。模型客户端关闭自动重试，失败调用会保留。单次结构化调用可在总预算内额外调用一次模型修复格式，相关重试次数、调用量、token 和时间都计入记录。

## 人工质量评分

请对照原始来源及事先独立核验的参考判断审核报告。填写评分人 ID、评分时间、`scoring_method=human`、评分规则版本，并在 `basis` 中记录理由和依据。

二元指标填 1 或 0，分别对应是或否。无法判断或不适用的项目留空。`major_tax_error=1` 表示存在重大税务错误，`inherited_historical_error=1` 表示继承了历史错误。`report_completeness` 填 0 到 1 之间的值。`human_edit_count` 按提前约定且一致的计数规则记录实质性修改次数。

评分范围包括结论正确性、重大错误、引用真实性、引用能否支持结论、时间及地区适用性、缺失事实、案例差异、历史错误继承、报告完整性和修改工作量。模型自评可供参考，标准答案由人工独立核验。多位评分人应分别保存评分和理由，需要统一判断时另行复核分歧。

```powershell
.\.venv\Scripts\python.exe -m eval.research_analysis data/research_eval/<experiment-directory> --scores path/to/human_scores.csv
```

将命令中的实验目录和评分文件路径替换为实际路径。程序生成 `descriptive.csv`、`paired_task_differences.csv`、`paired_summary.csv` 和分析说明。同一任务的重复运行会先按组汇总，再以案例家族为单位做 2,000 次 Bootstrap 重抽样，用来估计区间。少于两个案例家族时，区间留空。评分人均值用于描述各方评分，如需正式裁定须另行复核。当前的小样本和暂定分组适合探索差异，判断差异是否显著或哪组更优仍需进一步证据。

也可用 Python 或 Stata 读取原始数据，将示例中的文件路径改成实际路径。

```python
import pandas as pd

runs = pd.read_csv('path/to/runs.csv')
raw = pd.read_json('path/to/runs.jsonl', lines=True)
```

```stata
import delimited "path/to/runs.csv", clear encoding(utf8)
```
