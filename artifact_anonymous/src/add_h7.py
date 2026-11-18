s = open('README.md').read()
add = '''
### 四组数据汇总（NextGen 2 AM vs 7 AM，Train vs Test）

| 数据集 | 空间划分 | 活跃车辆 | v_ref (km/h) | Mode B 可行率（K>=3） |
|---|---|---|---|---|
| highway_2 (2 AM) | Train | 1.4 | 110 | 70/108（65%） |
| highway_2 (2 AM) | Test | 1.9 | 76 | 43/108（40%） |
| highway_7 (7 AM) | Train | 26.6 | 81 | 0-3/108（0-3%） |
| highway_7 (7 AM) | Test | 36.5 | 53 | 0/108（0%） |

结论：可行率随活跃车辆数**单调下降**，在 7 AM 密度下攻击关闭。
Test 划分是空间不相交区域，排序与 Train 一致。
'''
s = s.rstrip() + '\n' + add
open('README.md', 'w').write(s)
print('added', '四组数据汇总' in s)
