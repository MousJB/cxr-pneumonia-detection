# cxr-pneumonia-detection

[English](README.md)

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
の実際の胸部X線画像（DICOM）を用いた肺野陰影の**検出**。学習**前**に固定したプロトコルに
従って 2 つのモデルバージョン（v1、v2）を比較し、
[MedVision QA](https://github.com/MousJB/medvision-qa) で評価する。

> 作業中。本リポジトリのコードで実際に得られた結果のみを記載する。研究目的のプロジェクトで
> あり、**医療機器ではなく**、臨床的な検証も行っていない。

## 進捗

| ステップ | 状態 |
|---|---|
| 1. 環境構築、ダウンロード、データマニフェスト | 作業中 |
| 2. 前処理（DICOM → PNG、患者単位の分割、重複監査） | — |
| 3. 事前登録プロトコル（`PROTOCOL.md`） | — |
| 4. v1 ベースライン | — |
| 5. v2 | — |
| 6. テスト分割での 1 回限りの評価 | — |
| 7. 結果と限界 | — |

## データ

コンペティションの規約により画像の再配布は禁止されているため、画像は本リポジトリに
**含まれない**。公開するのはコード、メタデータ、ファイルのハッシュ、予測結果のみである。
再現するには、Kaggle でコンペティションの規約に同意し、API トークン
（`~/.kaggle/kaggle.json`）を作成したうえで、以下を実行する。

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/download_rsna.py
```

`scripts/download_rsna.py` はファイル数を確認し、`metadata/raw_manifest.csv`（ダウンロード
した全ファイルのパス、サイズ、SHA-256）と `metadata/raw_summary.json` を出力する。

## ライセンス

コード：`LICENSE` を参照。データ：RSNA Pneumonia Detection Challenge のコンペティション規約
（非商用利用に限る、画像の再配布禁止）。
