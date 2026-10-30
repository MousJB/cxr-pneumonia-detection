# cxr-pneumonia-detection

[English](README.md)

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
の実際の胸部 X 線画像（DICOM）を用いた肺野陰影の**検出**。学習**前**に固定したプロトコル
（[`PROTOCOL.ja.md`](PROTOCOL.ja.md)）に従って 2 つのモデルバージョンを比較し、
[MedVision QA](https://github.com/MousJB/medvision-qa) で検証する。

> **モデルはまだ一度も学習しておらず、本リポジトリは検出結果を一切報告していない。** 結果は、
> 本リポジトリのコードで実際に得られた時点で初めて記載する。研究目的のプロジェクトであり、
> **医療機器ではなく**、臨床的な検証も行っていない。

## 進捗

| ステージ | 状態 |
|---|---|
| 1. 環境構築、ダウンロード、データマニフェスト | 完了 |
| 2. 前処理（DICOM → PNG、患者単位の分割、重複監査、コンペティション指標） | 完了（commit `abfc675`） |
| 3. 計算ベンチマークと事前登録プロトコル（[`PROTOCOL.ja.md`](PROTOCOL.ja.md)、[`config.toml`](config.toml)） | 完了 |
| 4. A（512 px）と B（384 px）の最終学習、検証セットでの選択 | 未着手 |
| 5. テスト分割での 1 回限りの評価、MedVision QA、誤り分析 | 未着手 |
| 6. 結果、限界、公開 | 未着手 |

## 事前登録された計画（`PROTOCOL.ja.md` の要約）

- **モデル：** torchvision Faster R-CNN ResNet-50 FPN（v1）、COCO 重み、2 クラスヘッド。
  **A** は 512 px（基準）、**B** は 384 px（唯一の変更点）。FPN v2、MobileNetV3、バッチサイズ 8
  はベンチマークで測定したが、学習は行わない（理由は §4.3）。
- **学習：** バランス済みエポック（陽性すべて ＋ 同数の陰性、8,308 枚）、SGD lr 0.005、
  バッチ 4、float32、最大 10 エポック、検証セットのコンペティション指標による早期終了
  （patience 3）、シード 20261030。
- **選択：** 検証セットのみで行う。B が A に対して非劣性（コンペティション指標マージン 0.02、
  感度マージン 0.05、小さな陰影の再現率）であれば B を選ぶ。
- **テスト：** すべての決定を固定した後に 1 回だけ評価する。コンペティション指標（B の非劣性、
  マージン 0.02）と MedVision QA の基準 C1–C3（画像単位の AUROC、感度、特異度）。
- **GPU 予算：** float32 で測定したスループットから、GTX 1650（4 GB）で最大約 12–13 時間。

### 計算ベンチマーク（学習用画像、性能の測定ではない）

float16 混合精度はこの GPU では**動作しない**：ResNet-50 のモデルでは最初のステップから損失が
NaN になり（すべての最適化ステップがスキップされ、ボックスは予測されない）、試したすべての
モデルで float32 より 2.4–2.5 倍遅い。GTX 1650 はテンソルコアを持たない。そのため学習は
float32 で行う。学習していない計算の時間を測った以前の float16 の表は、`PROTOCOL.ja.md`
§5.2 に無効と明記して残している。

| アーキテクチャ | 入力 | バッチ | ピーク GB | 学習 枚/秒（float32） | 実行回数 |
|---|---:|---:|---:|---:|---:|
| R50 FPN v1（実行 A） | 512 | 4 | 1.63 | 4.75 ± 0.19 | 3 |
| R50 FPN v1（実行 B） | 384 | 4 | 1.19 | 6.57 ± 0.04 | 3 |
| R50 FPN v2 | 512 | 4 | 3.54 | 2.75 | 1 |
| MobileNetV3-L FPN | 512 | 4 | 0.75 | 22.40 | 1 |
| R50 FPN v1 | 512 | 8 | 3.43 | 3.64 | 1 |

方法、正しさの確認（有限の損失、パラメータの更新）、cuDNN 自動調整（+6 % でメモリ 2 倍：
オフ）、channels_last、データ読み込み、限界：`PROTOCOL.ja.md` §5。

## データ

コンペティションの規約により画像の再配布は禁止されているため、画像は本リポジトリに
**含まれない**。公開するのはコード、メタデータ、ファイルのハッシュ、予測結果のみである。
再現するには、Kaggle でコンペティションの規約に同意し、API トークン
（`~/.kaggle/kaggle.json`）を作成したうえで、以下を実行する。

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/download_rsna.py
.venv\Scripts\python scripts/prepare_rsna.py
```

`scripts/download_rsna.py` はファイル数を確認し、`metadata/raw_manifest.csv`（ダウンロード
した全ファイルのパス、サイズ、SHA-256）と `metadata/raw_summary.json` を出力する。
`scripts/prepare_rsna.py` は 512 px の PNG とボックス・画像の表を `data/`（git の管理対象外）
に、`metadata/split.csv` と `metadata/data_audit.json`（バージョン管理対象）を出力する。

### ステップ 2 の概要（commit `abfc675`）

26,684 枚の DICOM 画像（1024 px、8 ビット）を 2 × 2 の面積平均で 512 px PNG に縮小。
RSNA→NIH 対応表を用いた患者単位の分割（DICOM の `SOPInstanceUID` で画像ごとに確認済み）。
どの患者も複数のセットに属さない。

| セット | 画像数 | 陽性 | 患者数 |
|---|---:|---:|---:|
| train | 18,575 | 4,154 | 8,017 |
| val | 2,760 | 651 | 1,145 |
| test | 5,349 | 1,207 | 2,290 |

**重複監査：** 完全一致（画素の SHA-256）および近似重複（32 px サムネイルの 64 ビット pHash、
ハミング距離 ≤ 4）のペアを `metadata/data_audit.json` に記録。891 組の近似重複ペアが分割
境界をまたぐ。最も近い 4 組を目視で確認し、すべて異なる患者であった。32 px では pHash は
胸部 X 線画像に対して誤った近似重複ペアを生成する。残りの 887 組は確認していない。

## テスト

```
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest
```

`tests/test_metric.py`：コンペティション指標。`tests/test_data.py`：すべてのデータ拡張で
ボックスが画像に追従すること（合成矩形）。`tests/test_config.py`：`config.toml` が
MedVision QA の検証を通り、プロトコルと一致すること。
`tests/test_training.py`：バランス済みサンプラー、永続的なデータワーカーへのエポック番号の
伝達、ウォームアップ、早期終了、τ の選択。`tests/test_evaluation.py`：ボックスの対応付け、
画像単位・ボックス単位の指標、AUROC、対応のある患者単位ブートストラップ、選択規則。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `cxr/data.py` | データの読み込み、データ拡張、torchvision データセット |
| `cxr/models.py` | Faster R-CNN の構築（COCO 重み、2 クラスヘッド） |
| `cxr/metric.py` | コンペティション指標 |
| `cxr/training.py`、`cxr/evaluation.py` | サンプラー、早期終了、τ の選択。評価とブートストラップ |
| `scripts/` | ダウンロード、前処理、GPU ベンチマーク（`benchmark.py`、`precision_benchmark.py`）、学習（`train.py`）、評価（`evaluate.py`） |
| `metadata/` | マニフェスト、分割、重複監査（バイト単位で固定、ハッシュ記録済み） |
| `PROTOCOL.md`、`config.toml` | 事前登録プロトコルと MedVision QA の設定 |

## ライセンス

コード：`LICENSE` を参照。データ：RSNA Pneumonia Detection Challenge のコンペティション規約
（非商用利用に限る、画像の再配布禁止）。
