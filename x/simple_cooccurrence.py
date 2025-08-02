import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import io

# ページ設定
st.set_page_config(
    page_title="共起ネットワーク分析（簡易版）",
    page_icon="🕸️",
    layout="wide"
)

# タイトル
st.title("🕸️ 共起ネットワーク分析アプリ（簡易版）")
st.markdown("CSVファイルをアップロードして、テキストデータの共起ネットワークを可視化します")

# サイドバー
st.sidebar.header("設定")
upload_option = st.sidebar.radio(
    "データの選択方法",
    ["サンプルファイルを使用", "ファイルをアップロード"]
)

# データ読み込み
@st.cache_data
def load_data(file_path):
    """CSVファイルを読み込む"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"ファイルの読み込みエラー: {e}")
        return None

# テキストデータの前処理
def preprocess_text(text):
    """テキストの前処理"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # 特殊文字を除去
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 共起行列の作成
def create_cooccurrence_matrix(texts, min_freq=2, max_features=50):
    """共起行列を作成"""
    # テキストの前処理
    processed_texts = [preprocess_text(text) for text in texts]
    
    # CountVectorizerで単語を抽出
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_freq,
        stop_words=None,
        token_pattern=r'\b\w+\b'
    )
    
    try:
        # 単語-文書行列を作成
        word_doc_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # 共起行列を計算
        cooccurrence_matrix = word_doc_matrix.T @ word_doc_matrix
        
        return cooccurrence_matrix.toarray(), feature_names
    except Exception as e:
        st.error(f"共起行列の作成エラー: {e}")
        return None, None

# 簡易ネットワークグラフの作成
def create_simple_network(cooccurrence_matrix, feature_names, min_weight=1):
    """簡易ネットワークデータを作成"""
    nodes = []
    edges = []
    
    # ノード情報
    for i, word in enumerate(feature_names):
        nodes.append({
            'id': word,
            'size': cooccurrence_matrix[i, i],
            'degree': 0
        })
    
    # エッジ情報
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            weight = cooccurrence_matrix[i, j]
            if weight >= min_weight:
                edges.append({
                    'source': feature_names[i],
                    'target': feature_names[j],
                    'weight': weight
                })
                # 次数を更新
                for node in nodes:
                    if node['id'] in [feature_names[i], feature_names[j]]:
                        node['degree'] += 1
    
    return nodes, edges

# Plotlyでネットワークを可視化
def plot_simple_network(nodes, edges, title="共起ネットワーク"):
    """Plotlyで簡易ネットワークを可視化"""
    if not nodes:
        st.warning("ネットワークにノードがありません。設定を調整してください。")
        return None
    
    # ノードの座標を計算（円形配置）
    n_nodes = len(nodes)
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    radius = 1.0
    
    node_x = radius * np.cos(angles)
    node_y = radius * np.sin(angles)
    
    # ノードサイズの正規化
    max_size = max([node['size'] for node in nodes]) if nodes else 1
    node_sizes = [node['size'] / max_size * 30 + 10 for node in nodes]
    
    # プロット作成
    fig = go.Figure()
    
    # エッジを追加
    for edge in edges:
        source_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['source'])
        target_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['target'])
        
        x0, y0 = node_x[source_idx], node_y[source_idx]
        x1, y1 = node_x[target_idx], node_y[target_idx]
        
        # エッジの太さを正規化
        edge_width = edge['weight'] / max([e['weight'] for e in edges]) * 3 + 1 if edges else 1
        
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=edge_width, color='gray'),
            opacity=0.6,
            hoverinfo='skip',
            showlegend=False
        ))
    
    # ノードを追加
    node_text = [f"{node['id']}<br>出現回数: {node['size']}<br>次数: {node['degree']}" for node in nodes]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        text=[node['id'][:10] for node in nodes],
        textposition="middle center",
        hovertext=node_text,
        hoverinfo='text',
        name='ノード'
    ))
    
    # レイアウト設定
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-1.5, 1.5]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-1.5, 1.5]
        ),
        plot_bgcolor='white'
    )
    
    return fig

# メイン処理
def main():
    # データ読み込み
    if upload_option == "サンプルファイルを使用":
        # サンプルファイルを使用
        sample_files = ["sample.csv", "text.csv"]
        selected_file = st.sidebar.selectbox("サンプルファイルを選択", sample_files)
        file_path = f"{selected_file}"
        df = load_data(file_path)
    else:
        # ファイルアップロード
        uploaded_file = st.sidebar.file_uploader(
            "CSVファイルをアップロード",
            type=['csv']
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("ファイルをアップロードしてください")
            return
    
    if df is None or df.empty:
        st.error("データが読み込めませんでした")
        return
    
    # データの表示
    st.subheader("📊 データプレビュー")
    st.dataframe(df.head())
    
    # テキスト列の選択
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    if not text_columns:
        st.error("テキストデータを含む列が見つかりません")
        return
    
    selected_column = st.sidebar.selectbox("分析する列を選択", text_columns)
    
    # パラメータ設定
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        min_freq = st.number_input("最小出現回数", min_value=1, value=2)
    with col2:
        max_features = st.number_input("最大単語数", min_value=10, value=30)
    with col3:
        min_weight = st.number_input("最小共起重み", min_value=1, value=1)
    
    # 分析実行
    if st.sidebar.button("分析実行", type="primary"):
        with st.spinner("共起ネットワークを分析中..."):
            # テキストデータを取得
            texts = df[selected_column].dropna().tolist()
            
            if not texts:
                st.error("テキストデータがありません")
                return
            
            # 共起行列を作成
            cooccurrence_matrix, feature_names = create_cooccurrence_matrix(
                texts, min_freq, max_features
            )
            
            if cooccurrence_matrix is None:
                return
            
            # 簡易ネットワークを作成
            nodes, edges = create_simple_network(cooccurrence_matrix, feature_names, min_weight)
            
            if not nodes:
                st.warning("条件に合うノードがありません。パラメータを調整してください。")
                return
            
            # 結果表示
            st.subheader("🕸️ 共起ネットワーク")
            
            # ネットワーク図を表示
            fig = plot_simple_network(nodes, edges, f"{selected_column}の共起ネットワーク")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # 統計情報
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ノード数", len(nodes))
            with col2:
                st.metric("エッジ数", len(edges))
            with col3:
                if len(nodes) > 1:
                    total_edges = len(edges)
                    max_possible_edges = len(nodes) * (len(nodes) - 1) // 2
                    density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
                    st.metric("密度", f"{density:.3f}")
            with col4:
                if len(nodes) > 1:
                    avg_degree = sum(node['degree'] for node in nodes) / len(nodes)
                    st.metric("平均次数", f"{avg_degree:.1f}")
            
            # 詳細情報
            st.subheader("📈 詳細情報")
            
            # ノードの次数ランキング
            nodes_df = pd.DataFrame(nodes)
            nodes_df = nodes_df.sort_values('degree', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**次数ランキング（上位10位）**")
                st.dataframe(nodes_df[['id', 'degree', 'size']].head(10).rename(
                    columns={'id': '単語', 'degree': '次数', 'size': '出現回数'}
                ))
            
            # エッジの重みランキング
            edges_df = pd.DataFrame(edges)
            edges_df = edges_df.sort_values('weight', ascending=False)
            
            with col2:
                st.write("**共起強度ランキング（上位10位）**")
                st.dataframe(edges_df.head(10).rename(
                    columns={'source': '単語1', 'target': '単語2', 'weight': '共起回数'}
                ))
            
            # ダウンロード機能
            st.subheader("💾 データエクスポート")
            
            # ネットワークデータをCSVでダウンロード
            csv = nodes_df.to_csv(index=False)
            st.download_button(
                label="ネットワークデータをダウンロード",
                data=csv,
                file_name="network_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 