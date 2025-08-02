# イベント用Webアプリ

Streamlitを使用して作成されたイベント用Webアプリケーションです。

## 機能

### 📅 イベントタイムテーブル
- イベントのスケジュールを時系列で表示
- 現在時刻に基づいて進行中のイベントをハイライト表示
- 各イベントの会場・講演者情報を表示

### 🗺️ 会場案内図
- インタラクティブな会場平面図
- 各会場の詳細情報（収容人数、設備など）
- 色分けされた会場表示

### 🔗 リンク集
- カテゴリ別に整理されたリンク集
- 公式サイト、SNS、資料ダウンロードなどのリンク
- 各リンクの説明付き

### 👤 マイページ
- プロフィール画像のアップロード機能
- 名前、会社名、役職、メールアドレス、自己紹介の設定
- セッション間でのプロフィール情報の保持

## セットアップ

### ローカル環境での実行

#### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

#### 2. アプリケーションの実行
```bash
streamlit run app.py
```

#### 3. ブラウザでアクセス
アプリケーションが起動したら、ブラウザで `http://localhost:8501` にアクセスしてください。

### StreamlitCloudへのデプロイ

#### 1. GitHubにリポジトリをプッシュ
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main
```

#### 2. StreamlitCloudでデプロイ
1. [StreamlitCloud](https://share.streamlit.io/)にアクセス
2. GitHubアカウントでログイン
3. 「New app」をクリック
4. リポジトリ、ブランチ、メインファイル（app.py）を選択
5. 「Deploy!」をクリック

#### 3. デプロイ完了
数分後にアプリが公開され、URLが生成されます。

## カスタマイズ

### タイムテーブルの編集
`app.py` の `get_timetable_data()` 関数内のデータを編集することで、イベントスケジュールを変更できます。

### 会場情報の編集
`get_venue_info()` 関数内の辞書を編集することで、会場の詳細情報を変更できます。

### リンク集の編集
`get_links_data()` 関数内のリストを編集することで、リンク集をカスタマイズできます。

## 技術仕様

- **フレームワーク**: Streamlit
- **データ可視化**: Plotly
- **画像処理**: Pillow
- **データ処理**: Pandas

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 