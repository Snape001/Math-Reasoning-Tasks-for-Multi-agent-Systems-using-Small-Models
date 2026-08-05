# Paired collaboration analysis

| Metric | Value |
|---|---:|
| Problems | 200 |
| Baseline accuracy | 93.50% |
| Collaboration accuracy | 94.00% |
| Accuracy delta | +0.50% |
| Rescued / regressed | 3 / 2 |
| McNemar exact p | 1.000000 |
| Avg baseline tokens | 6813.03 |
| Avg collaboration tokens | 6648.48 |
| Token multiplier | 0.9758470166724644 |

`rescued` 表示单智能体错误而协作正确；`regressed` 表示单智能体正确而协作错误。
只有在配对 200 题上 rescued 稳定多于 regressed，才能声称协作机制带来净收益。
