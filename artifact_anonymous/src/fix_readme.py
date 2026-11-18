p = 'README.md'
s = open(p).read()
s = s.replace('- （19 MB', '- `InTAS_highway_2.zip`（19 MB')
s = s.replace('- 高密度 （832 MB', '- 高密度 `InTAS_highway_7.zip`（832 MB')
s = s.replace('每个接收车辆一个 ；按  去重后',
              '每个接收车辆一个 `veh_XXXX.json`；按 `(sender_id, sendTime)` 去重后')
s = s.replace('-  是报告值（含真实 V2X 噪声），真值 = 报告值 − 。',
              '- `sender.pos/spd` 是报告值（含真实 V2X 噪声），真值 = 报告值 − `pos_noise/spd_noise`。')
s = s.replace('：解压内层包', '`src/nextgen_load.py`：解压内层包')
s = s.replace('1 s 分箱 → 。', '1 s 分箱 → `X/V/X_rep/V_rep/AL/ain/aout`。')
s = s.replace(' 直接复用攻击/检测器。',
              '`run_one(..., sim_override=sim)` 直接复用攻击/检测器。')
open(p, 'w').write(s)
print('fixed', 'InTAS_highway_2.zip' in s, 'sender_id' in s, 'sim_override' in s)
