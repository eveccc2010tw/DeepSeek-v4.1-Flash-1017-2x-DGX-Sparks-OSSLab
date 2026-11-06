# OSSLab 去限制（Abliterated / ABLIT）版：Mia 新引擎與 Tuned 客製配方

**本配方使用去限制權重：EAQ v1.1 混合精度 EXL3，套用 drowzeys Keys Abliterated attention overlay。**
這是已準備好的 attention 輸出投影權重替換，不使用執行期 GLP steering。
本 repo 記錄權重身分與 API／效能驗收；未進行拒答率或通用模型品質評測。
模型來源：[EAQ v1.1](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw)、
[Keys Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated)。

**[完整歷史測速、記憶體優化與雙機監控](performance-memory.zh-TW.md)**

這份 fork 以 Mia 官方 recipe `31457d05f8917301bbf609c3d12c7f364af8a4a6` 為底，
保留官方的 48 個引擎補丁與 2 個 inputs 補丁，加入 4 個已在雙 DGX Spark 上驗證的移植。
TensorFold 固定在 `fbbc97132cd004237d9887b8d3185896efaedc5c`。

## 四個補丁

| 補丁 | 作用 |
| --- | --- |
| `0100-dsv41-wide-copy-windows.patch` | 32-token 寬複製視窗；保留新引擎的 calibration、copy_proposals、no_copies |
| `0101-prompt-progress-states.patch` | RAM-only 64K prefill 進度點，支援分段 prefill 的取消重送 |
| `0102-responses-keepalive-ping.patch` | Responses／Anthropic SSE keepalive |
| `0103-responses-buffer-growth.patch` | Responses 文字／工具參數緩衝改為幾何成長，避免逐段重新複製 |

前三項由既有 OSSLab 客製移植；0103 對照
[sehoon123 Tuned](https://github.com/sehoon123/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned/tree/171052a7d40745ba43f1d9fcb01cb3c2643d93923)
的 0060 修復。這不是整包 Tuned 補丁覆蓋。
DSML 解析及 CUDA context 修復已由基底／官方補丁涵蓋，不重複套用。

## 已驗證設定

1M 視窗、2,101,248 分片池、3 streams、32-token copy、graph cap 128、NCCL channels 8、
0.5 GiB prompt cache、每台 32 GiB spill、64K RAM 進度點；分段 prefill 為
`TF_FILL_CHUNKS=1`、`TF_FILL_FIRST=1`、`TF_FILL_CUT_IN=8192`。
外部 memguard 使用 soft 2.0 GiB、hard 1.0 GiB、hold 4 秒。

實測權重為 EAQ v1.1 `aee640e2afba93be683bb5b9b420115fe7fbac20`，搭配已準備好的
drowzeys overlay `d6681ac778ecbd7b422382ff6af88fb3d2debcb7`；Engram
`1bae0e1088993b9b82de08f59de5be5a34e42170`，kit
`abe07c7d509a0199cc7fec15263be2aa52ac652f`。權重與 gated overlay 不隨本 repo 散布。

本輪實際設定由 `tools/prepare-mia-zig-container.py` 產生。
`examples/osslab-rank0.json`、`examples/osslab-rank1.json` 是該工具的可編輯範例，
主機位址使用文件專用的 `192.0.2.1`；所有路徑、介面、GID 及 RDMA 裝置需填入自己的值。
模型／overlay 身分必須與實際準備的權重一致，不能只修改標籤。

上游 `start.sh` 保留原本的 16-row 檢查及 EAQ v1 overlay 限制。
上述 32-row、EAQ v1.1＋overlay 設定使用本頁的候選工具與已準備好的權重目錄；
JSON 範例不是 `scripts/local.sh`。本 repo 沒有變更上游的模型準備限制。

## 建置及建立候選

完整建置沿用上游 `zig/Dockerfile`／`scripts/prepare.sh`，所有 52 個引擎補丁會依檔名排序套用。
已有相容的 Mia 原生底圖時，可用本 repo 的 engine-only helper 重建，限制編譯容器的 RAM／swap 與 CPU：

```bash
python3 tools/rebuild-mia-zig-engine.py \
  --base-image YOUR_PREPARED_MIA_NATIVE_IMAGE \
  --patch-dir zig/patches --inputs-patch-dir zig/patches-inputs \
  --work-dir /srv/build/osslab-current \
  --tag tensorfold-dsv41:osslab-current \
  --memory 3584m --cpus 0-3 --allow-capabilities-diff
```

`YOUR_PREPARED_MIA_NATIVE_IMAGE` 須含對應的 Zig、CUDA 工具及 inputs；能力差異仍需人工核對。
本輪在 aarch64 sm121 編譯一次，兩台使用相同產物，三個執行檔 SHA256 一致。
編譯產物、Docker 映像及權重不附在這份原始碼配方中。

先準備模型、rank cache、kit、Engram、token map 及私有 API key 檔，再在各主機建立候選：

```bash
python3 tools/prepare-mia-zig-container.py \
  --spec examples/osslab-rank0.json --image tensorfold-dsv41:osslab-current \
  --original YOUR_EXISTING_RANK0 --name osslab-current-r0 \
  --state-dir /srv/private/osslab-current-r0
```

rank 1 改用另一份 JSON 與容器名稱。工具只建立未啟動容器，保留原容器 inspect 與 private rollback 紀錄。
設定範例明確填入本輪數值；工具的歷史預設值不是本輪部署值。
啟動／監督可用隨附的 `tools/supervise-native-pair.py`、通用 pair／systemd 範例及上游 `scripts/memguard.sh`；
監督程式需要已配置的 `tmssh` gateway，主機認證與 gateway 實作不隨 repo 發布。
安裝方式、門檻與監控指令見[監控文件](performance-memory.zh-TW.md#雙機監督與-memguard)。
升級使用新的 spill 目錄；兩個 rank 的 engine、patch、模型與 Engram 身分必須一致。

## 測速與實機驗收

同權重、同提示、temperature 0、關閉 thinking、cached tokens 0：

| 場景 | 升級前 | 升級後驗收 | 上線後重測 | 本次去限制版重測 |
| --- | ---: | ---: | ---: | ---: |
| 短提示，3 次中位數 | 93.61 tok/s | 93.34 tok/s | 92.47 tok/s | **93.28 tok/s** |
| 1,735-token 提示，1 次 | 86.94 tok/s | 86.33 tok/s | 86.11 tok/s | **87.67 tok/s** |

上線後重測首字延遲為 0.30／1.09 秒；三路同時生成各 44.7–50.2 tok/s，
合計 126.8 tok/s（總輸出除以整輪時間，含等待及 prefill）。前後答案 hash 一致；普通生成沒有宣稱加速。
逐筆證據：[升級前](mia-current-baseline-speed.json)、[升級後](mia-current-speed.json)、
[上線後重測](live-speed-retest.json)、[本次重測](live-speed-ablit-retest.json)。本次首字延遲為 0.30／1.10 秒。

歷史一般生成曾有 EAQ v1 的 **106.13 tok/s**；舊 EAQ v1.1／768K 的程式碼複製測試為
**214.74 tok/s**、雙路合計 **231.59 tok/s**。它們的權重、設定與工作負載不同，
不能當作這次新引擎的一般生成速度。[歷史表格與原始數據](performance-memory.zh-TW.md#歷史測速)。

記憶體優化包含 KV 分片、NCCL 8 channels、Graph cap 128、0.5 GiB prompt cache、
每台 32 GiB SSD spill 及 64K RAM 進度點。本次測前／測後可用記憶體約 7.0／7.5 GiB，
雙端 memguard、監督服務與 HTTP health 均正常；這兩次快照不代表峰值。
[連續採樣及優化來源](performance-memory.zh-TW.md#記憶體優化與採樣)。

實機另外通過：API 權限、SSE、含 XML／Unicode DSML 的工具串流參數、Responses、Anthropic thinking、圖片；
兩個約 299K-token 對話交替均 cached 299,008、約 1.1 秒恢復；長 prefill 插入短請求 0.52 秒完成。
雙 rank 重啟後 SSD 實際 reads／recovered points 命中，無 read／CRC／size errors；
未生成即取消的唯一前綴請求重送 cached 131,072，disk reads 未增加，證明 RAM 進度恢復。

CPU regression：Zig 23/23 build steps，審查後 332 fresh tests，其餘成功快取；
server 120＋engine API 114、取消進度 5/5 通過。
host spill fault-injection 使用上游 `TF_SPILL_TEST_DIR=/dev/shm`，真 SSD 另做上述實機驗收。
未驗單一請求塞滿 1M、超過本輪約 300K 的長度及 24 小時 soak。

重跑測速時，以環境提供 `MODEL_BASE_URL`／`MODEL_API_KEY`，輸出放私有目錄：

```bash
python3 tools/benchmark-matched.py --label your-pair --output /private/speed.json
# API／長對話驗收另需 requests：python3 -m pip install requests
timeout 900 python3 tools/accept-mia-current.py smoke --output /private/smoke.json
timeout 1800 python3 tools/accept-mia-current.py long --output /private/long.json
timeout 900 python3 tools/accept-mia-current.py progress --cancel-after 75 --output /private/progress.json
```

取消時間依 prefill 速度調整，須已建立至少一個 64K 進度點並早於完整 prefill；
本輪指定 72.47 秒取消，服務端於 74.38 秒結束原請求。SSD resume phase 必須結合雙 rank 重啟及實際 I/O 指標。

## 來源與授權

上游 [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks)、
[TensorFold](https://github.com/bertholomus/TensorFold)、Tuned 及原作者的署名與授權均保留。
本 fork 的新增內容與來源雜湊見 [manifest](manifest.json) 及根目錄 `NOTICE`。

## 公開封裝查核

本輪公開封裝沿用已部署的引擎與工具原始碼，沒有修改其執行行為。
四個補丁及六個 helper 逐位元組比對一致；52＋2 補丁以 engine-only helper 重算身份為 `07d9bcbe5743`。
兩份 profile 實際送入 `candidate_config` 驗證通過，所有相對文件連結均存在。

```bash
python3 -m unittest discover -s tests -p 'test_mia_zig_container.py' -v
python3 -m unittest discover -s tests -p 'test_native_pair.py' -v
python3 -m py_compile tools/prepare-mia-zig-container.py tools/prepare-prefix-container.py \
  tools/rebuild-mia-zig-engine.py tools/benchmark-matched.py tools/accept-mia-current.py \
  tools/supervise-native-pair.py
bash -n scripts/memguard.sh
systemd-analyze --user verify examples/osslab-native-pair.service
git diff --check -- . ':(exclude)zig/patches/**'
```

候選 regression 9/9、雙機監督 regression 9/9、語法、service 與差異檢查通過；主工具的 `--help` 均可執行。
引擎／來源與工具／公開資料兩份唯讀審查均為 Standards 0、Spec 0、Verification pass。
本輪沒有重編 GPU 或重新部署；上述 GPU 結果來自同一份引擎來源先前的實機驗收與上線後重測。

## 每台 1 TB 與本輪調校

[容量盤點與調校紀錄](native-speed-tuning.zh-TW.md)：每台各 1 TB 的單版本容量估算為 head 約 708.49 GB、worker 約 375.41 GB，含現役映像與完整 spill 額度，OS／更新另留空間。本輪三組 planner 沒有穩定收益，已還原原設定；未更換去限制權重，未宣稱引擎加速。
