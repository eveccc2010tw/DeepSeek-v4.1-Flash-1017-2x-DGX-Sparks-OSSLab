# 去限制版測速、歷史紀錄與記憶體監控

本配方的現役權重為 **EAQ v1.1 + drowzeys Keys Abliterated overlay（去限制／ABLIT 版）**。
固定版本見 [manifest](manifest.json)。權重以 attention 輸出投影替換方式準備，不使用執行期 GLP steering；
未做拒答率或通用模型品質評測。

## 現役重測

台灣時間 **2026-11-07 07:06**，TensorFold `fbbc971`、1M window、3 streams、KV 分片、
32-token copy、EAQ v1.1 混合精度權重；temperature 0、thinking 關閉、cached tokens 0。
用同一組 synthetic prompts，短題 103 in／256 out 跑三題取中位數；長提示 1,735 in／128 out 跑一題。
解碼速度為 `(completion_tokens - 1) / (最後文字事件時間 - 第一文字事件時間)`，不含首字等待。

| 場景 | decode tok/s | 首字延遲 |
| --- | ---: | ---: |
| 短提示，中位數 | **93.28** | **0.298 秒** |
| 長提示 | **87.67** | **1.098 秒** |

[逐筆 JSON](live-speed-ablit-retest.json) 保留 prompt／output hash、usage 與時間；
四題都符合 matched gate，且與升級後驗收的四題輸出 hash 一致。
同一現役引擎較早的[三路併發重測](live-speed-retest.json)為各 44.7–50.2 tok/s，
合計 **126.81 tok/s**：合計值是全部輸出 tokens 除以整輪耗時，包含 prefill 與等待，並非三路 decode 數字相加。

測速工具與提示隨 repo 提供。環境提供 `MODEL_BASE_URL`、`MODEL_API_KEY` 後執行：

```bash
python3 tools/benchmark-matched.py --label your-ablit-pair --output /private/speed.json
```

## 歷史測速

以下是 Git 留存的完整部署觀測；不同列可能更換權重、引擎、context、streams 與 draft 策略。
它們不是控制單一變因的實驗，也不表示最新版本達到歷史最高值。
一般生成使用相同的 103 in／256 out 短題及 1,735 in／128 out 長提示；
跨權重答案不要求相同，現役前後比較另有輸出 hash 核對。

| 時間（台灣）／配方 | 視窗與條件 | 短題 tok/s | 長提示 tok/s |
| --- | --- | ---: | ---: |
| 2026-10-19，vLLM 舊基準 | 384K | 24.03 | 23.98 |
| 2026-10-19，LibertAIDAI／TensorFold + GLP | 768K、drafts 5 | 72.90 | 65.46 |
| 2026-10-21，easthambuilds + drow 舊配方 | 1M | 85.17 | 78.07 |
| 2026-10-21，Mia EAQ v1 ABLIT 候選 `31eea460` | 1M、1 stream | **106.13** | 90.13 |
| 2026-10-22，EAQ v1／DSML25 `370daf0c` | 摘要未記 matched run 視窗 | 105.84 | 90.37 |
| 2026-10-30，EAQ v1.1 ABLIT Tuned `d8e5b28` | 768K、3 streams、27 BF16 linears | 94.57 | 87.51 |
| 2026-11-07，升級前 EAQ v1.1 ABLIT | 1M、3 streams | 93.61 | 86.94 |
| 2026-11-07，新引擎 `fbbc971` 驗收 | 1M、3 streams | 93.34 | 86.33 |
| 2026-11-07 07:06，新引擎本次重測 | 1M、3 streams | **93.28** | **87.67** |

EAQ v1 的 106.13 tok/s 是早期全 2.9bpw 候選；當時 text-stream／JSON schema 功能驗收未過，未保留上線。
現役 EAQ v1.1 包含 27 個 BF16 linears，並啟用 KV 分片及三路服務，不能把差異全部歸因於某個補丁。
DSML25 的 105.84 tok/s 為另一筆後續紀錄，並非將 106.13 四捨五入。

原始數據整理在 [historical-benchmarks.json](historical-benchmarks.json)，保留各來源的 Git commit、檔案路徑與 SHA256，
以及可公開的逐題 benchmark／摘要。[升級前](mia-current-baseline-speed.json)、[升級後](mia-current-speed.json)另附完整四題結果。

### 歷史程式碼複製工作負載

舊 `d8e5b28`／EAQ v1.1 ABLIT／768K 配方的 **32-token 寬複製視窗**，
在既有程式碼逐字改版工作負載測得：

| 場景 | 結果 |
| --- | ---: |
| 單路，2,788 prompt／2,736 completion tokens | **214.74 decode tok/s**，首字 1.666 秒 |
| 雙路，各自 decode | 128.55／125.61 tok/s |
| 雙路合計吞吐 | **231.59 tok/s** |

紀錄顯示雙路彼此、以及雙路與單路的程式碼輸出逐字一致。
這是可複製既有程式碼的工作負載；本次沒有在新引擎重跑這項測試。
歷史摘要只保留雙路 aggregate 數值，未保留其計時公式，不能與現役三路 end-to-end 值直接比較。
來源為歷史 JSON 的 `eaq_v11_tuned_768k.wide_copy_and_progress_acceptance.wide_copy`。

## 記憶體優化與採樣

DGX Spark 的 GPU 與主機共用記憶體，因此觀察 `/proc/meminfo` 的 `MemAvailable`，同時檢查 swap 與容器狀態。

| 機制 | 現役設定與作用 |
| --- | --- |
| KV 分片 | `KV_SHARD=1`；雙 rank 分攤壓縮 KV。歷史相同 2,101,248 pool 的壓縮平面由每台 1,783 降至 1,062 MiB，約省 721 MiB／台；不是本輪隔離重測 |
| NCCL channels | `NCCL_MAX_NCHANNELS=8`；歷史相較 16 channels 約少 1.0 GiB pinned host memory／台，並非本輪獨立 A/B |
| Graph 上限 | `GRAPH_CAP=128`；限制保留的 graph 數量，淘汰後可重新捕獲，不宣稱固定省幾 GiB |
| Prompt cache | `--prompt-cache-gib 0.5`；限制 prompt cache 預算 |
| SSD spill | 每台 32 GiB 本機 SSD 額度，淘汰的 cache 可持久化及重啟後恢復；這不是額外的 32 GiB RAM／VRAM |
| Prefill 進度 | 每 65,536 tokens 保留 rolling RAM 進度點，這類進度點本身不 enqueue SSD；一般 history／shared mark 仍依 spill 策略持久化 |

KV／NCCL 的歷史觀測摘錄與來源雜湊見歷史 JSON 的 `memory_optimization_history`。
這些優化與新引擎共同構成現役配置；不把歷史節省量相加當作本輪實測。
舊版 116,821-token 取消案例觀察到取消後 SSD writes 未增加、重送命中 65,536 tokens，完成後 writes 增加 1；
這是該案例的觀測，不代表所有 prefill 途中都不會寫入 SSD。

現役引擎實機驗收曾以目標間隔 5 秒採樣、共 **142 筆**，兩台最低 `MemAvailable` 為 **7.27／7.79 GiB**，
swap free 均未改變。採樣時段為台灣時間 **2026-11-07 02:56:48～03:09:04**，
不涵蓋稍後 03:10 的三路生成；整體驗收另有約 299K-token 雙對話、SSD 恢復、取消重送等場景，不能將本段採樣當作全程最低值。
歷史 JSON 的 `current_acceptance_memory` 保留採樣起訖及原始採樣檔 SHA256。
採樣間隔無法捕捉所有瞬間峰值，且該值不是 1M 滿載保證。

本次短測另記錄測前、測後各一筆[記憶體快照](memory-monitoring-retest.json)：

| rank | 測前可用 GiB | 測後可用 GiB | guard／容器 |
| --- | ---: | ---: | --- |
| 0 | 7.014 | 7.027 | 正常，OOMKilled=false |
| 1 | 7.513 | 7.491 | 正常，OOMKilled=false |

本次 swap free 未改變、監督服務 active、HTTP health 200。快照與先前連續採樣來自不同時間，不能混當同一輪最低值。
尚未在本版測單一請求塞滿 1M 或 24 小時 soak。

## 雙機監督與 memguard

隨附 `tools/supervise-native-pair.py` 與 `tests/test_native_pair.py`，與已部署來源逐位元組一致。
上游 `scripts/memguard.sh` 保留原樣。兩層作用如下：

- 每台 memguard 每 **0.25 秒**讀取可用記憶體：低於 **2.0 GiB 持續 4 秒**，或低於 **1.0 GiB 連續兩次**，就 `docker kill` 該 rank；短暫回升只記錄不 kill。觸發時記錄高 RSS 程序快照。
- 控制端 supervisor 每 **30 秒**檢查雙容器；其中一端退出會停止雙端，再由 systemd 有上限地重試。連續三次主機查詢失敗會走清理／失敗流程，SSH timeout 與清理也需要時間。
- 冷啟動前要求每台至少 **110 GiB** 可用；先開兩端 guard、再開 worker、等 2 秒開 head。既有雙端皆運行時可接管，不要求再次空出權重所占記憶體。
- 設定檔旁的 `.lock` 防止兩個 supervisor 同時操作；停止某端失敗時保留該端 guard。systemd 限制每小時 3 次啟動嘗試、失敗後等 5 秒重試，並非無限循環。

**依賴：**Linux、Python 3、systemd user service、控制端可用的 `tmssh HOST COMMAND` gateway；
各 DGX 上須有 Docker 存取權與 Bash、GNU coreutils、awk、procps。
`tmssh` 與其認證是部署環境的前置條件，本 repo 不提供 gateway，不能直接當普通 SSH 的獨立工具使用。
既有上游 `start.sh` 有自己的管理方式；同一對容器應選定一個管理流程。

準備方式：

1. 依[配方](README.zh-TW.md)建立兩個未啟動候選容器，確認 deployment labels、模型與 engine 身分一致。
2. 將 `scripts/memguard.sh` 放到兩台自己的 operations 目錄，設為可執行，確保該目錄與 log／pid 可由遠端操作帳號寫入。
3. 複製 `examples/osslab-native-pair.json` 為控制端私有 `pair.json`，填入自己的 tmssh host、容器、絕對 operations 路徑。範例明寫 **2.0／1.0 GiB、4 秒**；不要省略而落回工具的歷史預設 soft 2.5 GiB。
4. 控制端安裝 supervisor 與 service 範例：

```bash
install -Dm755 tools/supervise-native-pair.py "$HOME/.local/share/osslab-deepseek/supervise-native-pair.py"
install -Dm600 /private/pair.json "$HOME/.config/osslab-deepseek/pair.json"
install -Dm644 examples/osslab-native-pair.service "$HOME/.config/systemd/user/osslab-native-pair.service"
systemd-analyze --user verify "$HOME/.config/systemd/user/osslab-native-pair.service"
systemctl --user daemon-reload
# 確認已由舊管理流程交接後才啟用；此指令會管理並啟動指定的雙容器：
systemctl --user enable --now osslab-native-pair.service
```

常用唯讀查詢（主機與路徑請換成自己的值）：

```bash
systemctl --user status osslab-native-pair.service
journalctl --user -u osslab-native-pair.service -n 60 --no-pager
tmssh spark-head 'awk "/MemAvailable|SwapFree/" /proc/meminfo; tail -n 20 /srv/deepseek/ops/memguard.log'
tmssh spark-worker 'awk "/MemAvailable|SwapFree/" /proc/meminfo; tail -n 20 /srv/deepseek/ops/memguard.log'
```

停止這個 service 會停止雙容器；memguard 則是在記憶體壓力下直接 kill，兩者不是相同的關閉方式。
本次發布增加原始碼與文件，沒有在現役主機重裝或調整門檻。
