# Physics-Informed Machine Learning

このリポジトリは、University of Washington の Composite Group の教材・サンプルコードを集めたものです。

## 免責事項 / Disclaimer

**For Educational Use Only**

本リポジトリに含まれるコードおよび教材は、教育目的のみで公開されています。
本リポジトリの内容は、以下の公式サイトから提供されている教材に基づいています：

- **Source:** [University of Washington Composites Group - AI](https://composites.uw.edu/AI/)
- **Author:** Navid Zobeiry (navidz@uw.edu)
- **Institution:** University of Washington, Seattle, WA

教授により公開されているコースウェアを参考にしており、商用利用は禁止です。

## 内容
- `3Dprint/` : 3D印刷に関するデータと解析スクリプト（`3Dprint.py`）
- `heat/` : 熱伝導のサンプルデータとスクリプト
- `adhesive/` : 接着に関するデータ
- `intro/` : 各種機械学習導入ノート、GPR、NNなど
- `SINDy/` : SINDy に関する簡単な例

クイックスタート
1. 仮想環境を作成・有効化（Windows PowerShell の例）:

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
```

2. 依存関係をインストール:

```powershell
pip install -r requirements.txt
```

3. 3Dprint スクリプトを実行:

```powershell
& ".venv\Scripts\python.exe" "3Dprint\3Dprint.py"
```

備考
- 大きなPDFおよびデータファイルが含まれます。
- PDF ファイルはリポジトリから削除されており、必要に応じて公式ソース（https://composites.uw.edu/AI/）から入手してください。
- 依存関係は `requirements.txt` で管理されています。

ライセンス
- 本リポジトリのコードは、教育目的での使用を想定しています。
- 商用利用や再配布の際は、元の著者（Navid Zobeiry）および University of Washington に確認してください。
