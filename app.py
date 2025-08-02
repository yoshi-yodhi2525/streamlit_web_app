import streamlit as st
import pandas as pd
from datetime import datetime, time
import plotly.graph_objects as go
from PIL import Image
import os

# ページ設定
st.set_page_config(
    page_title="イベントアプリ",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .event-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .venue-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .link-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #ffeaa7;
    }
    .profile-section {
        background-color: #f1f2f6;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# サンプルデータ
def get_timetable_data():
    return pd.DataFrame({
        '時間': ['09:00', '09:30', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00'],
        'イベント': [
            '受付開始',
            '開会式',
            '基調講演「AIの未来」',
            'パネルディスカッション',
            '昼食・ネットワーキング',
            'ワークショップA「技術セッション」',
            'ワークショップB「ビジネスセッション」',
            '休憩・交流タイム',
            'クロージング',
            '懇親会'
        ],
        '会場': [
            'ロビー',
            'メインホール',
            'メインホール',
            'メインホール',
            'レストラン',
            '会議室A',
            '会議室B',
            'ロビー',
            'メインホール',
            'レストラン'
        ],
        '講演者': [
            '-',
            '主催者',
            '田中教授',
            'パネリスト4名',
            '-',
            '山田技術者',
            '佐藤経営者',
            '-',
            '主催者',
            '-'
        ]
    })

def get_venue_info():
    return {
        'メインホール': '1階、収容人数200名、プロジェクター・音響設備完備',
        '会議室A': '2階、収容人数50名、ホワイトボード・プロジェクター完備',
        '会議室B': '2階、収容人数30名、ホワイトボード完備',
        'レストラン': '1階、昼食・懇親会会場',
        'ロビー': '1階、受付・休憩スペース'
    }

def get_links_data():
    return [
        {'カテゴリ': '公式', 'タイトル': 'イベント公式サイト', 'URL': 'https://example.com', '説明': 'イベントの詳細情報'},
        {'カテゴリ': '公式', 'タイトル': '参加者専用ページ', 'URL': 'https://participants.example.com', '説明': '参加者向け資料ダウンロード'},
        {'カテゴリ': 'SNS', 'タイトル': 'Twitter公式アカウント', 'URL': 'https://twitter.com/event', '説明': '最新情報・写真投稿'},
        {'カテゴリ': 'SNS', 'タイトル': 'Instagram', 'URL': 'https://instagram.com/event', '説明': 'イベント写真・ストーリー'},
        {'カテゴリ': '資料', 'タイトル': '発表資料ダウンロード', 'URL': 'https://docs.example.com', '説明': '講演資料のダウンロード'},
        {'カテゴリ': 'アンケート', 'タイトル': '参加者アンケート', 'URL': 'https://survey.example.com', '説明': 'イベント後のアンケート'}
    ]

def create_venue_map():
    # 簡易的な会場図を作成
    fig = go.Figure()
    
    # フロア平面図の簡易表現
    fig.add_trace(go.Scatter(
        x=[0, 10, 10, 0, 0],
        y=[0, 0, 8, 8, 0],
        mode='lines',
        line=dict(color='black', width=2),
        name='建物外壁'
    ))
    
    # 各会場の位置
    venues = {
        'メインホール': [2, 4, 6, 3],
        'レストラン': [7, 1, 9, 2],
        'ロビー': [1, 6, 4, 7],
        '会議室A': [6, 5, 8, 6],
        '会議室B': [6, 7, 8, 8]
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (venue, coords) in enumerate(venues.items()):
        # 透明度を含む色をrgba形式で指定
        base_color = colors[i]
        rgba_color = f'rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, 0.3)'
        
        fig.add_trace(go.Scatter(
            x=[coords[0], coords[2], coords[2], coords[0], coords[0]],
            y=[coords[1], coords[1], coords[3], coords[3], coords[1]],
            mode='lines+text',
            line=dict(color=colors[i], width=3),
            text=[venue, '', '', '', ''],
            textposition='middle center',
            name=venue,
            fill='toself',
            fillcolor=rgba_color
        ))
    
    fig.update_layout(
        title='会場案内図',
        xaxis=dict(range=[0, 10], showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 8], showgrid=False, zeroline=False),
        width=600,
        height=400,
        showlegend=True
    )
    
    return fig

def show_timetable():
    st.markdown('<h2 class="section-header">📅 イベントタイムテーブル</h2>', unsafe_allow_html=True)
    
    timetable = get_timetable_data()
    
    # 現在時刻に基づいてハイライト
    now = datetime.now().time()
    
    for idx, row in timetable.iterrows():
        event_time = datetime.strptime(row['時間'], '%H:%M').time()
        
        if event_time <= now:
            st.markdown(f"""
            <div class="event-card" style="background-color: #d4edda; border-left-color: #28a745;">
                <strong>{row['時間']}</strong> - {row['イベント']}<br>
                <small>会場: {row['会場']} | 講演者: {row['講演者']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="event-card">
                <strong>{row['時間']}</strong> - {row['イベント']}<br>
                <small>会場: {row['会場']} | 講演者: {row['講演者']}</small>
            </div>
            """, unsafe_allow_html=True)

def show_venue_guide():
    st.markdown('<h2 class="section-header">🗺️ 会場案内図</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_venue_map()
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        venue_info = get_venue_info()
        st.markdown('<h3>会場詳細</h3>', unsafe_allow_html=True)
        
        for venue, info in venue_info.items():
            st.markdown(f"""
            <div class="venue-info">
                <strong>{venue}</strong><br>
                {info}
            </div>
            """, unsafe_allow_html=True)

def show_links():
    st.markdown('<h2 class="section-header">🔗 リンク集</h2>', unsafe_allow_html=True)
    
    links_data = get_links_data()
    
    # カテゴリ別にグループ化
    categories = {}
    for link in links_data:
        cat = link['カテゴリ']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(link)
    
    for category, links in categories.items():
        st.markdown(f'<h3>{category}</h3>', unsafe_allow_html=True)
        
        for link in links:
            st.markdown(f"""
            <div class="link-card">
                <strong>{link['タイトル']}</strong><br>
                <a href="{link['URL']}" target="_blank">{link['URL']}</a><br>
                <small>{link['説明']}</small>
            </div>
            """, unsafe_allow_html=True)

def show_mypage():
    st.markdown('<h2 class="section-header">👤 マイページ</h2>', unsafe_allow_html=True)
    
    # ユーザー情報の入力フォーム
    with st.form("user_profile"):
        st.markdown('<h3>プロフィール設定</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # プロフィール画像のアップロード
            uploaded_file = st.file_uploader("プロフィール画像", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, width=150)
        
        with col2:
            name = st.text_input("お名前", value=st.session_state.get('user_name', ''))
            company = st.text_input("会社名", value=st.session_state.get('user_company', ''))
            position = st.text_input("役職", value=st.session_state.get('user_position', ''))
            email = st.text_input("メールアドレス", value=st.session_state.get('user_email', ''))
            bio = st.text_area("自己紹介", value=st.session_state.get('user_bio', ''), height=100)
        
        submitted = st.form_submit_button("プロフィールを保存")
        
        if submitted:
            st.session_state.user_name = name
            st.session_state.user_company = company
            st.session_state.user_position = position
            st.session_state.user_email = email
            st.session_state.user_bio = bio
            st.success("プロフィールが保存されました！")
    
    # 保存されたプロフィールの表示
    if st.session_state.get('user_name'):
        st.markdown('<h3>現在のプロフィール</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="profile-section">
            <h4>{st.session_state.user_name}</h4>
            <p><strong>会社:</strong> {st.session_state.user_company}</p>
            <p><strong>役職:</strong> {st.session_state.user_position}</p>
            <p><strong>メール:</strong> {st.session_state.user_email}</p>
            <p><strong>自己紹介:</strong><br>{st.session_state.user_bio}</p>
        </div>
        """, unsafe_allow_html=True)

# メインアプリケーション
def main():
    st.markdown('<h1 class="main-header">🎉 イベントアプリ</h1>', unsafe_allow_html=True)
    
    # サイドバーナビゲーション
    st.sidebar.title("📱 ナビゲーション")
    
    page = st.sidebar.selectbox(
        "ページを選択",
        ["タイムテーブル", "会場案内", "リンク集", "マイページ"]
    )
    
    # ページ表示
    if page == "タイムテーブル":
        show_timetable()
    elif page == "会場案内":
        show_venue_guide()
    elif page == "リンク集":
        show_links()
    elif page == "マイページ":
        show_mypage()
    
    # フッター
    st.markdown("---")
    st.markdown("© 2024 イベントアプリ - Streamlitで作成")

if __name__ == "__main__":
    main() 