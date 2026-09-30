"""更新 04 对比表。"""
import json

with open('04_超预期成长.ipynb', 'r', encoding='utf-8') as fp:
    nb = json.load(fp)

# 找对比表 cell
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'markdown':
        continue
    src = cell.get('source', '')
    if isinstance(src, list): src = ''.join(src)
    if '| SUE |' in src and '研报 IC |' in src and '多空年化' in src:
        target_idx = i
        break

src = nb['cells'][target_idx]['source']
if isinstance(src, list): src = ''.join(src)

old_table = '''| **SUE** | **10.60%** ⭐ | **2.87%** | **41.42%** | **13.80%** |
| **SUR** | **6.44%** ⭐ | 2.40% | **19.04%** | 11.35% |
| JOR(MAX 40d) | -0.24% | 2.71% | 2.21% | 12.01% |
| JOR(LAST 40d) | 1.32% | **3.17%** | 19.30% | **12.48%** |
| 财报超预告 | (跳过) | 1.84% | - | 6.84% |
| 单季超预期 | (跳过) | 3.23% | - | 9.12% |
| **综合（3 因子）** | 见上 | 4.17% | 见上 | **18.10%** |'''

new_table = '''| **SUE**（lag=1）| **3.14%** ⭐ | **2.87%** | 11.79% | **13.80%** |
| **SUR**（lag=1）| **2.56%** ⭐ | 2.40% | 8.90% | 11.35% |
| JOR(MAX 40d) | -0.24% | 2.71% | 2.21% | 12.01% |
| JOR(LAST 40d) | 1.32% | **3.17%** | 19.30% | **12.48%** |
| 财报超预告 | (跳过) | 1.84% | - | 6.84% |
| 单季超预期 | (跳过) | 3.23% | - | 9.12% |
| **综合（3 因子）** | 见上 | 4.17% | 见上 | **18.10%** |'''

if old_table in src:
    src = src.replace(old_table, new_table)
    nb['cells'][target_idx]['source'] = src
    with open('04_超预期成长.ipynb', 'w', encoding='utf-8') as fp:
        json.dump(nb, fp, ensure_ascii=False, indent=1)
    print('✅')
else:
    print('NOT FOUND')