# Physics-Informed Machine Learning

このリポジトリは、University of Washington の Composite Group の教材・サンプルコードを集めたものです。

内容:
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
- 大きなPDFやデータファイルが含まれます。必要に応じて `.gitignore` を調整してください。
- リポジトリの説明は GitHub 上のリポジトリ設定で編集できます（Settings → Repository name & description）。

ライセンス
- 必要であれば追加してください。
