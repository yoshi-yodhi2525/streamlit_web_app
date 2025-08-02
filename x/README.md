# 共起ネットワーク分析アプリ

このアプリは、CSVファイルからテキストデータを読み込み、共起ネットワークをインタラクティブに可視化するStreamlitアプリケーションです。

## 機能

- 📊 CSVファイルのアップロードと読み込み
- 🕸️ 共起ネットワークの可視化
- 📈 ネットワーク統計情報の表示
- 💾 分析結果のエクスポート
- 🎛️ パラメータの調整機能

## 使用方法

### ローカルでの実行

1. 必要なライブラリをインストール：
```bash
pip install -r requirements.txt
```

2. アプリを実行：
```bash
streamlit run cooccurrence_network.py
```

### Streamlit Cloudでのデプロイ

1. GitHubリポジトリにコードをプッシュ
2. [Streamlit Cloud](https://share.streamlit.io/)にアクセス
3. リポジトリを接続
4. メインファイルとして `cooccurrence_network.py` を指定
5. デプロイ

## ファイル構成

```
x/
├── cooccurrence_network.py    # メインアプリケーション
├── requirements.txt           # 依存関係
├── .streamlit/
│   └── config.toml          # Streamlit設定
├── sample.csv                # サンプルデータ
├── text.csv                  # サンプルデータ
└── README.md                 # このファイル
```

## パラメータ設定

- **最小出現回数**: 単語が含まれる文書の最小数
- **最大単語数**: 分析対象とする単語の最大数
- **最小共起重み**: エッジを表示する最小の共起回数

## 出力

- インタラクティブな共起ネットワーク図
- ネットワーク統計（ノード数、エッジ数、密度、平均次数）
- 次数ランキング
- 共起強度ランキング
- ネットワークデータのCSVエクスポート

## 技術仕様

- **フレームワーク**: Streamlit
- **可視化**: Plotly
- **ネットワーク分析**: NetworkX
- **テキスト処理**: scikit-learn
- **データ処理**: pandas, numpy 