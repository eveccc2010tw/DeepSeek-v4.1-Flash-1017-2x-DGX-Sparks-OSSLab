# 引擎預取與 48-row 複製視窗實測

本輪通過效益驗證的是 **48-row 複製視窗：合成程式碼複製題約快 16.38%**。
一般生成仍約 92 tok/s。另做了三項引擎原始碼預取修改，未找到穩定收益，沒有採用該新映像。
逐輪數據、usage、hash 與計時見 [JSON 紀錄](native-engine-optimization.json)。

權重維持 EAQ v1.1 `aee640e2`＋Keys Abliterated overlay `d6681ac7`。
引擎維持 `fbbc971`＋52 補丁 `07d9bcbe5743`；1M 視窗、2,101,248 pool、3 streams、
KV 分片、32 GiB spill、原 memguard 與雙機 supervisor 均保持。

## 原始碼預取實驗

以現役 52 補丁為基底，增加預設關閉的第 53 補丁，兩端實際重編並測試：

- 提早預取：取消主模型預取前的固定 3 微秒等待。
- 草稿預取：為 DSpark 草稿 pass 加上獨立 CUDA stream／events，提前讀取後續層權重。
- BF16 預取：將 plain BF16 linears 納入有效 byte 範圍計算；與前兩項合併測量。

同四題、temperature 0、thinking off、cached tokens 0；每組暖機後三輪，每輪 short 為三題中位數，
context 是 1,735 input tokens。表中再取三輪中位數。逐題 prompt／output hashes 均一致。

| 設定 | short SSE decode tok/s | context SSE decode tok/s |
| --- | ---: | ---: |
| 原服務，之前 | 92.43 | 86.55 |
| 新映像，全部關閉 | 92.06 | 86.31 |
| 提早預取 | 91.57 | 86.81 |
| 草稿預取 | 92.15 | 86.76 |
| 三項合併 | 92.43 | 86.87 |
| 原服務，還原後 | 91.65 | 85.92 |

各組另通過三路輸出與複製題核對，合併組通過完整 API smoke。
基準本身亦有波動，沒有候選達到約 3% 穩定提升，因此不保留預取映像。
公開的 52 補丁配方未加入這個實驗補丁；JSON 保留實驗 identity 與 patch SHA256 供追蹤。

## 32 → 48 → 32 的複製題比較

既有引擎與寬複製視窗補丁已支援 48 rows，設定工具原本只允許 16／32。
這次只開放明確選用 48；工具未設定時仍為 32。兩端 `round_rows` 及 head `copy_rows` 一起設為 48。

測試要求複製 80 個 Python 函式，只修改其中一個常數，輸出 2,081 tokens。
各階段先暖機，再各測三次；九次 AST 正確、完整 output hash 相同、cached tokens 0。

| 設定 | 一般 short SSE decode tok/s | context SSE decode tok/s | 複製 native decode tok/s | 複製整題秒數 |
| --- | ---: | ---: | ---: | ---: |
| 32，之前 | 91.65 | 85.92 | 215.16 | 11.04 |
| 48 | 91.55 | 86.13 | 250.39 | 9.69 |
| 32，還原後 | 91.52 | 86.11 | 215.15 | 11.05 |

48 的三次複製 decode 分別為 249.99／250.39／250.66 tok/s；相對前後 32 基準中位數約快 16.38%。
整題耗時也下降。這是容易利用既有文字的合成複製／單處修改工作量，不能推廣為所有寫程式或一般生成都快 16%。
三路一般題的總 completion tokens／整輪耗時為 121.32 → 120.76 tok/s，差約 0.46%，未見明顯退步。

SSE decode 使用 `(completion_tokens - 1) / (最後文字事件時間 - 第一文字事件時間)`；
複製題取原生回應的 `tensorfold.tokens_per_second`，整題秒數取 client wall clock。
不同計時口徑分欄呈現。三路合計包含等待與 prefill，不能當成各路 decode 相加。

## 長提示、功能與記憶體

將同一份程式碼放到 **32,999-token、未命中 cache** 的提示後，分別在 32／48 各測一次：
AST、完整輸出 hash、usage 一致。處理中第 3 秒送入短請求，分別在 4.18／4.20 秒完成，
早於長請求的首 token（16.01／15.97 秒）。這是功能與排程驗收，不是多輪長上下文效能結論。

48 通過 authentication、一般文字 SSE、含特殊 DSML 字元的逐段工具參數、Responses include、
Anthropic thinking 與紅色圖片辨識。兩次回退均核對原設定與健康狀態。

啟動紀錄的 worker arena 使用量由 5,030,672,200 → 5,296,714,392 bytes，增加約 **0.25 GiB**；
reserved 由 5,307,026,432 → 5,508,746,240 bytes，增加約 0.19 GiB。
head 啟動 device 計數亦增加 201,719,808 bytes（約 0.19 GiB）；arena 使用量與 allocator reserved 是不同指標。
48 未增加權重或 rank cache 的磁碟副本。

複製 A/B 與 33K 驗收期間每 5 秒採樣，共 85 組，最低 MemAvailable 為 head **6.77 GiB**、worker **7.23 GiB**；
SwapFree 相較起點最多減少各 4 KiB。既有 soft 2 GiB／hard 1 GiB guard 保留。
這是本輪工作量的採樣，不能當成 1M 長上下文最壞記憶體上限；本輪沒有重跑完整 1M 壓力或多長會話重啟恢復。

兩端現有 kit 相同。32／48 audit 均有兩項相同的低 row findings，源於 audit 展開了 caller 不會發出的組合；
audit 本身仍 exit 1，不能稱全面通過。48 新增的寬 `_rowmm_parts` 由既有小段 fallback 覆蓋，
較寬 KV fetch 也可能改走非 ring 交換；實際效益以上述 GPU 測量為準。

## 使用已驗證的設定

在兩端 hostspec 明確填入：

```json
{"round_rows": 48, "copy_rows": 48}
```

其餘權重、pool、streams、kit、spill 及 guards 沿用原配方。
`copy_rows` 由 head 使用；worker 的 `round_rows` 必須一致。省略這兩欄仍保留 32 的預設。

## 其他方向的判斷

- 草稿模型每輪接受更多 token，可減少每個輸出 token 要讀的權重，但需要相符的 drafter／訓練與品質驗證；本輪沒有訓練或更換它。
- 跨機交換仍可研究排程與 pinned-memory 路徑。一般 GPUDirect RDMA 在 Spark 不受支援，不能靠開啟 nvidia-peermem／GDRCopy 直接加速；目前路徑符合 NVIDIA 的 pinned-host-buffer 建議。
- 現有 DanTup dashboard 是有 client 才採樣、預設每 5 秒一次，不能套用上游密集 nvidia-smi 輪詢的損失數字；本輪保留監控。
- 上游測過預取 budgets、CPU 綁核、減少草稿專家等負收益方向；沒有把這些方法當成已驗證的本機加速。

來源：[Tuned kernel 研究](https://github.com/sehoon123/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned/blob/71052a7d40745ba43f1d9fcb01cb3c2643d93923/research/KERNEL-ROUNDS.md)、
[NVIDIA Spark GPUDirect 說明](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/related/1)、
[DanTup dashboard](https://github.com/DanTup/dgx_dashboard)。
