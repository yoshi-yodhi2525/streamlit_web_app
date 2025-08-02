import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import io

# ページ設定
st.set_page_config(
    page_title="共起ネットワーク分析",
    page_icon="🕸️",
    layout="wide"
)

# タイトル
st.title("🕸️ 共起ネットワーク分析アプリ")
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

# ネットワークグラフの作成
def create_network_graph(cooccurrence_matrix, feature_names, min_weight=1):
    """NetworkXグラフを作成"""
    G = nx.Graph()
    
    # ノードを追加
    for i, word in enumerate(feature_names):
        G.add_node(word, size=cooccurrence_matrix[i, i])
    
    # エッジを追加
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            weight = cooccurrence_matrix[i, j]
            if weight >= min_weight:
                G.add_edge(feature_names[i], feature_names[j], weight=weight)
    
    return G

# Plotlyでネットワークを可視化
def plot_network(G, title="共起ネットワーク"):
    """Plotlyでネットワークを可視化"""
    if len(G.nodes()) == 0:
        st.warning("ネットワークにノードがありません。設定を調整してください。")
        return None
    
    # レイアウトの計算
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # ノードの座標
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>出現回数: {G.nodes[node]['size']}")
        node_size.append(G.nodes[node]['size'])
    
    # エッジの座標
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G.edges[edge]['weight'])
    
    # ノードサイズの正規化
    max_size = max(node_size) if node_size else 1
    node_size = [size / max_size * 20 + 10 for size in node_size]
    
    # エッジの太さの正規化
    max_weight = max(edge_weights) if edge_weights else 1
    edge_width = [weight / max_weight * 3 + 1 for weight in edge_weights]
    
    # プロット作成
    fig = go.Figure()
    
    # エッジを追加
    for i in range(0, len(edge_x), 3):
        if i + 2 < len(edge_x):
            fig.add_trace(go.Scatter(
                x=[edge_x[i], edge_x[i+1]],
                y=[edge_y[i], edge_y[i+1]],
                mode='lines',
                line=dict(width=edge_width[i//3], color='gray'),
                opacity=0.6,
                hoverinfo='skip',
                showlegend=False
            ))
    
    # ノードを追加
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        text=[name[:10] for name in G.nodes()],
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
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
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
        file_path = f"x/{selected_file}"
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
            
            # ネットワークグラフを作成
            G = create_network_graph(cooccurrence_matrix, feature_names, min_weight)
            
            if len(G.nodes()) == 0:
                st.warning("条件に合うノードがありません。パラメータを調整してください。")
                return
            
            # 結果表示
            st.subheader("🕸️ 共起ネットワーク")
            
            # ネットワーク図を表示
            fig = plot_network(G, f"{selected_column}の共起ネットワーク")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # 統計情報
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ノード数", len(G.nodes()))
            with col2:
                st.metric("エッジ数", len(G.edges()))
            with col3:
                density = nx.density(G)
                st.metric("密度", f"{density:.3f}")
            with col4:
                if len(G.nodes()) > 1:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                    st.metric("平均次数", f"{avg_degree:.1f}")
            
            # 詳細情報
            st.subheader("📈 詳細情報")
            
            # ノードの次数ランキング
            degree_ranking = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            degree_df = pd.DataFrame(degree_ranking, columns=['単語', '次数'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**次数ランキング（上位10位）**")
                st.dataframe(degree_df.head(10))
            
            # エッジの重みランキング
            edge_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
            edge_weights.sort(key=lambda x: x[2], reverse=True)
            edge_df = pd.DataFrame(edge_weights, columns=['単語1', '単語2', '共起回数'])
            
            with col2:
                st.write("**共起強度ランキング（上位10位）**")
                st.dataframe(edge_df.head(10))
            
            # ダウンロード機能
            st.subheader("💾 データエクスポート")
            
            # ネットワークデータをCSVでダウンロード
            network_data = []
            for node in G.nodes():
                network_data.append({
                    'node': node,
                    'degree': G.degree(node),
                    'size': G.nodes[node]['size']
                })
            
            network_df = pd.DataFrame(network_data)
            csv = network_df.to_csv(index=False)
            st.download_button(
                label="ネットワークデータをダウンロード",
                data=csv,
                file_name="network_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 