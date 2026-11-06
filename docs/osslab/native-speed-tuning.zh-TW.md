# 現役速度調校與每台 1 TB 磁碟估算

現役維持 EAQ v1.1＋Keys Abliterated overlay、TensorFold `fbbc971`＋`07d9bcbe5743`、
1M window、3 streams、KV 分片、32-token copy。**本輪未找到穩定加速，已還原原 pair；沒有更換權重。**
數據見 [本輪紀錄](native-speed-tuning.json)。

## 舊版為何看起來比較快

| 配方 | 一般短題 decode tok/s |
| --- | ---: |
| 更早 EAQ v1／DSML25 | 105.84 |
| 本次引擎升級前，EAQ v1.1 | 93.61 |
| 本次引擎升級後，EAQ v1.1 | 93.34 |

本次更新前後只差約 0.3%，測試輸出相同，沒有顯著降速證據。
較早的 105.84 與現役不是同一套部署：權重從 EAQ v1 改為含 27 個 BF16 linears 的 v1.1，
KV 分片、streams、視窗及引擎也有變更。高精度層會增加部分運算／資料成本，但歷史沒有逐項隔離，
**不能把全部降幅或主要原因直接歸給 BF16、去限制 overlay 或某個補丁**。
另一個更早的 106.13 是 text-stream／JSON schema 驗收失敗的候選，不是現役前一版。
來源見 [升級前後](performance-memory.zh-TW.md)及 [EAQ 改版紀錄](performance-memory.zh-TW.md)。

舊版 214.74 decode tok/s 是可複製既有程式碼的工作量，不能和約 93 tok/s 的從零生成直接相比。

## 本輪 planner A/B

先量原設定三次，再逐一切換；每組暖機後量三次，最後還原原設定重測。
同四題、temperature 0、thinking off、cached tokens 0；候選逐題 prompt／output hash 均與原設定相同。
以下數字為三輪的中位數；short 每輪另含三題取中位數。

| 設定 | short decode tok/s | context decode tok/s | 三路 end-to-end tok/s |
| --- | ---: | ---: | ---: |
| 原設定，切換前 | 93.40 | 87.50 | 126.94 |
| calibration 0.5 | 88.27 | 84.90 | 124.03 |
| calibration 0.1 | 92.22 | 87.53 | 123.21 |
| solo overhead | 91.59 | 86.82 | 121.65 |
| 原設定，還原後 | 92.08 | 86.90 | 未重跑 |

基準前後亦有約 1.4% 波動；沒有候選達到穩定約 3% 的 short 提升，全部不採用。
三路合計為總 completion tokens／整輪耗時，含 prefill 與等待；不能視為各路 decode 數字相加。
另有輸出 2,081 tokens 的合成程式碼複製題，AST 正確且輸出 hash 相同；本輪只保留 wall-clock 計時，
沒有取得該題 runtime decode 欄位，因此不與歷史 214.74 decode tok/s 比較。

## 引擎耗時採樣

用同映像包裝 Nsight Systems，取得 head 的 10 秒 CUDA node trace。
kernel 累計時間主要在量化矩陣／MoE，另見 L2 prefetch、RDMA collect；不同 stream 會重疊，
表中占比不是總回覆耗時比例，也不是可直接省下的時間。
這是後續原始碼調查線索，尚未形成實測通過的引擎修改。

插樁過程前 12 題通過一致性檢查，第 13 題的 matched／hash 合併 assertion 未通過；失敗 response 未保留，
無法分辨是哪個子條件。該候選未採用，且不得當作數值等價或加速驗收。
已還原原設定，health 正常；未插樁的四題重送均核對原 output hash 一致。

## 每台各 1 TB SSD

**依現役檔案實測占用估算，每台各 1 TB 可容納這份單版本配方。**
目前實機 SSD 較大，這是容量盤點，並非在實體 1 TB 磁碟重裝驗收。
以下使用資料檔 allocated blocks、inode 去重與十進位 GB，spill 以完整 32 GiB 額度計入；
目錄 metadata 未計入，低於表格 0.01 GB 精度。逐 byte 盤點見 JSON 的 `capacity`。

| 項目 | head GB | worker GB |
| --- | ---: | ---: |
| 保留的原始 EAQ v1.1 權重 | 210.80 | — |
| overlay 新寫 shard | 127.46 | — |
| overlay sidecar | 0.68 | — |
| worker metadata／vision | — | 5.86 |
| 單端 rank cache | 106.28 | 106.28 |
| Engram | 203.08 | 203.08 |
| kit | 0.67 | 0.67 |
| 現役 Docker image | 25.17 | 25.17 |
| spill 額度 32 GiB | 34.36 | 34.36 |
| **合計** | **708.49** | **375.41** |

head 的模型來源與 stock 權重為相同 inode，served 未修改 shard 也與 stock 共用，不能重複加總；來源另有約 0.2 MB 獨立 metadata，已計入總額。
此估算已保留原始權重，沒有刪除資料；其他模型、舊映像、Docker build cache、OS 與更新暫存不在表內。
head 建議留約 150–200 GB 給 OS 與更新，避免同時保留多套完整版本；在 head 暫建另一端 rank cache
還會多約 106 GB。兩台合計只有 1 TB 則不適用，上表合計已超過 1 TB。

「ABLIT 多約 300 GB」是可能保留重寫權重與額外 rank cache 的磁碟空間情境，並非額外 300 GB RAM。
本配方已完成 graft；本輪更新與調校重用現有權重，不會重新下載另一套模型。
