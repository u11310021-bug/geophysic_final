import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
import os

# 1. 檢查字型檔是否存在，沒有的話就從網路下載 (使用 Noto Sans TC)
font_path = 'NotoSansTC-Regular.otf'
if not os.path.exists(font_path):
    # 顯示下載訊息，避免以為當機
    print(f"正在下載中文字型至 {font_path}，請稍候...") 
    os.system(f'wget "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansTC-Regular.otf" -O {font_path}')

# 2. 告訴 Matplotlib 使用這個字型
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = ['Noto Sans TC']
plt.rcParams['axes.unicode_minus'] = False # 解決負號 '-' 顯示成方塊的問題
# -----------------------------------------------------------
# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="地球物理大冒險：從地表到深部",
    page_icon="🌍",
    layout="wide"
)

# --- 側邊欄：導航與外部連結 ---
st.sidebar.title("🧭 導航地圖")
page = st.sidebar.radio("前往關卡：", ["首頁：板塊構造", "任務一：捕捉地震波", "任務二：重力計算", "任務三：透視地底"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 延伸閱讀與工具")
st.sidebar.info("這些是研究室提供的強大工具，必存！")
st.sidebar.markdown("[1. 作業繳交區 (HuggingFace)](https://huggingface.co/spaces/u11310021/homework1)")
st.sidebar.markdown("[2. 重力異常計算工具](https://huggingface.co/spaces/u11310021/freeair_gravity)")
st.sidebar.markdown("[3. PyGMT 板塊繪圖庫](https://github.com/u11310021-bug/plot_plate_boundary_pygmt)")
st.sidebar.markdown("[4. PyGMT 學習筆記](https://u11310021-bug.github.io/learn_pygmt/)")
st.sidebar.markdown("[5. 網誌教學](https://dichiowooly.blogspot.com/2025/09/body-font-family-arial-sans-serif-line.html)")

# --- 共用函式 ---
def local_css(file_name):
    # 這裡可以加入 CSS 美化，暫時略過保持簡單
    pass

# ==========================================
# 首頁：板塊構造 (Nazca Plate)
# ==========================================
if page == "首頁：板塊構造":
    st.title("🌍 醒醒吧同學！我們在納斯卡板塊上！")
    st.markdown("### 如果你現在覺得頭暈，可能是因為下面這個東西在動...")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.success("#### 納斯卡板塊 (Nazca Plate) 的地質身分證")
        st.markdown("""
        **這傢伙夾在太平洋和南美洲中間，造就了壯觀的安地斯山脈。**
        
        它有三種主要的邊界類型（期末考會考，請畫螢光筆）：
        
        1.  **➡️ 東部：聚合型邊界 (Convergent)**
            *   **發生什麼事？** 納斯卡板塊(海) 鑽到 南美板塊(陸) 下面。這叫 **隱沒帶 (Subduction Zone)**。
            *   **後果：** 超大地震 (如 1960 智利大地震)、火山爆發、形成秘魯-智利海溝。
        
        2.  **⬅️ 西部：分離型邊界 (Divergent)**
            *   **發生什麼事？** 跟太平洋板塊分手快樂。
            *   **地點：** 東太平洋海隆 (East Pacific Rise)。會有岩漿冒出來形成新地殼！
            
        3.  **⬇️ 南部：分離型邊界**
            *   **地點：** 智利海隆 (Chile Rise)。
        """)
        
    with col2:
        # 這裡用 PyGMT 的概念圖，但為了互動網頁不當機，我們用 matplotlib 模擬示意
        st.info("#### 💡 構造示意圖")
        st.write("想像一下，左邊是海，右邊是山...")
        
        # 簡單繪製隱沒帶示意圖
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        land = np.ones_like(x) * 5
        ocean_plate = -0.5 * x + 5
        
        ax.plot(x, land, 'g-', linewidth=3, label='南美板塊 (陸)')
        ax.plot(x, ocean_plate, 'b-', linewidth=3, label='納斯卡板塊 (海)')
        ax.fill_between(x, -1, ocean_plate, color='lightblue', alpha=0.5)
        ax.fill_between(x, ocean_plate, -5, color='gray', alpha=0.3, label='地函')
        ax.text(8, 5.5, "安地斯山脈", fontsize=12, color='green')
        ax.text(2, 2, "海溝", fontsize=12, color='blue')
        
        ax.set_ylim(-5, 8)
        ax.legend()
        ax.axis('off')
        ax.set_title("隱沒帶：板塊去哪兒了？")
        st.pyplot(fig)

    st.warning("👉 想畫出專業級的板塊地圖？去側邊欄點擊 **PyGMT 板塊繪圖庫**！")

# ==========================================
# 任務一：Obspy 地震波
# ==========================================
elif page == "任務一：捕捉地震波":
    st.title("📡 任務一：用 Python 抓地震波")
    st.markdown("不要再手動下載資料了，讓 AI 和 Python 幫你做苦工。")
    
    st.subheader("🛠️ 你的武器：Obspy")
    st.markdown("這是我們在 GitHub Codespace 上跑的程式碼，直接複製貼上就能用！")
    
    # 顯示程式碼 (根據圖片內容重製)
    code = """
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt

# 1. 設定客戶端
client = Client("IRIS")

# 2. 設定時間 (2025年了，假設我們回顧或是預測)
starttime = UTCDateTime("2025-10-07T23:52:12")
duration = 120 # 秒
endtime = starttime + duration

# 3. 選擇台站 (台灣常見台站範例)
network = "TW"
station = "NACB"
location = ""
channel = "BHZ"

# 4. 下載資料
st = client.get_waveforms(network, station, location, channel, starttime, endtime)
print("下載完成！")

# 5. 畫圖並存檔
fig = plt.figure(figsize=(10, 4))
st.plot(outfile="waveform.png", fig=fig)
plt.show()
    """
    st.code(code, language='python')
    
    st.markdown("---")
    st.subheader("📊 結果預覽 (模擬)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("參數調整模擬：")
        noise_level = st.slider("背景雜訊程度", 0.0, 5.0, 1.0)
        amp = st.slider("地震波震幅", 1, 10, 5)
    
    with col2:
        # 模擬產生一個地震波圖 (因為不想在 demo 依賴外部網路連線抓資料)
        t = np.linspace(0, 120, 1000)
        # 模擬 P 波和 S 波到達
        signal = np.zeros_like(t)
        p_arrival = 30
        s_arrival = 60
        
        # 簡單的衰減正弦波模擬地震
        signal[t >= p_arrival] += amp * np.sin(2 * np.pi * 5 * (t[t>=p_arrival]-p_arrival)) * np.exp(-0.1 * (t[t>=p_arrival]-p_arrival))
        signal[t >= s_arrival] += (amp*1.5) * np.sin(2 * np.pi * 3 * (t[t>=s_arrival]-s_arrival)) * np.exp(-0.05 * (t[t>=s_arrival]-s_arrival))
        
        # 加入雜訊
        noise = np.random.normal(0, noise_level * 0.2, len(t))
        data = signal + noise
        
        fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
        ax_wave.plot(t, data, 'k-', linewidth=0.8)
        ax_wave.set_title(f"TW.NACB..BHZ - 模擬波形")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Counts")
        ax_wave.grid(True, alpha=0.3)
        st.pyplot(fig_wave)
        st.caption("☝️ 這是模擬圖。執行上面的 Python code 可以抓到真實數據喔！")

# ==========================================
# 任務二：重力計算
# ==========================================
elif page == "任務二：重力計算":
    st.title("🍎 任務二：地心引力抓不住你？")
    st.markdown("這裡充滿了數學公式，但別怕，我們有 `numpy`。")
    
    st.markdown("### 自由空氣修正 (Free-air Correction)")
    st.latex(r"FAC = 0.308 \times h")
    st.write("隨著高度 $h$ (公尺) 增加，重力會變小，所以我們要修正回來。")
    
    st.markdown("---")
    st.subheader("🧮 互動計算機")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 根據提供的截圖內容設計輸入
        lat = st.number_input("緯度 (Latitude)", value=48.1195)
        ele = st.number_input("海拔高度 (Elevation, m)", value=487.9)
        observed_g = st.number_input("觀測重力值 (mGal)", value=980717.39)
        
    with col2:
        # 計算邏輯 (參照圖片中的 numpy 計算)
        # 1. 將緯度轉弧度
        lat_r = lat * np.pi / 180
        
        # 2. 理論重力公式 (Somigliana equation approximation from screenshot context)
        # Gt = Ge * (1 + 0.005278895 * sin(lat)^2 + 0.000023462 * sin(lat)^4)
        Ge = 978031.85
        Gt = Ge * (1 + 0.005278895 * np.sin(lat_r)**2 + 0.000023462 * np.sin(lat_r)**4)
        
        # 3. 自由空氣修正
        FAC = ele * 0.308
        
        # 4. 布給修正 (Bouguer Correction) - 圖片中有 BC = ele * 0.112
        BC = ele * 0.112
        
        # 5. 自由空氣異常 (Free-air Anomaly)
        # 通常 FAA = G_obs - G_theoretical + FAC
        # 但圖片中的算式似乎是: (Go + FAC) - Gt
        faa = (observed_g + FAC) - Gt
        
        # 6. 布給異常 (Bouguer Anomaly)
        # BA = FAA - BC
        ba = faa - BC
        
        st.write(f"**理論重力值 (Gt):** `{Gt:.4f}` mGal")
        st.write(f"**自由空氣修正 (FAC):** `+{FAC:.4f}` mGal")
        st.write(f"**布給修正 (BC):** `-{BC:.4f}` mGal")
        
        st.success(f"### 🎯 自由空氣異常 (FAA): {faa:.4f} mGal")
        st.info(f"### ⛰️ 布給異常 (BA): {ba:.4f} mGal")
        
    st.markdown("---")
    st.markdown("**程式碼小抄 (Numpy):**")
    st.code("""
import numpy as np
lat_r = lat * np.pi / 180
Gt = 978031.85 * (1 + 0.005278895*np.sin(lat_r)**2 + ...)
FAC = ele * 0.308
BC = ele * 0.112
FAA = (observed_g + FAC) - Gt
    """, language='python')

# ==========================================
# 任務三：震測折射 (Refraction)
# ==========================================
elif page == "任務三：透視地底":
    st.title("🔦 任務三：透視地底的秘密")
    st.markdown("只要在地表敲一下，我們就能知道地底下有多深。這是**折射震測**的魔術。")
    
    st.markdown("### 關鍵公式：交錯距離 (Crossover Distance)")
    st.markdown("當**折射波**跑得比**直達波**快的那一瞬間，那個距離就是 $X_{cr}$。")
    
    # PDF 中的公式
    st.latex(r"T = \frac{2h \cos\theta}{V_1} + \frac{X}{V_2}")
    st.write("經過一番推導 (司乃爾定律代入...)，我們得到交錯距離公式：")
    st.latex(r"X_{cr} = 2h \sqrt{\frac{V_2 + V_1}{V_2 - V_1}}")
    
    st.markdown("---")
    st.subheader("🎮 實驗室：調整參數看看 Xcr 怎麼變")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        h = st.slider("地層厚度 h (m)", 10, 100, 30)
        v1 = st.slider("第一層速度 V1 (m/s)", 300, 1000, 500)
        v2 = st.slider("第二層速度 V2 (m/s)", 1001, 3000, 1500)
        
        if v1 >= v2:
            st.error("⚠️ 錯誤：折射震測要求 V2 > V1 才能發生全反射！")
        else:
            # 計算 Xcr
            term = (v2 + v1) / (v2 - v1)
            x_cr = 2 * h * np.sqrt(term)
            st.metric("交錯距離 Xcr", f"{x_cr:.2f} m")
            
    with col2:
        if v1 < v2:
            # 繪製 T-X 圖 (走時曲線)
            x = np.linspace(0, x_cr * 2, 100)
            
            # 直達波 T = X / V1
            t_direct = x / v1
            
            # 折射波 T = X/V2 + Ti (截距時間)
            # Ti = 2h * sqrt(V2^2 - V1^2) / (V1*V2) ... 或是用 cos formula
            # 簡單用 T = X/V2 + 2h*cos(theta)/V1
            sin_theta = v1/v2
            cos_theta = np.sqrt(1 - sin_theta**2)
            t_intercept = (2 * h * cos_theta) / v1
            t_refract = x / v2 + t_intercept
            
            fig_tx, ax_tx = plt.subplots(figsize=(8, 5))
            
            ax_tx.plot(x, t_direct, 'b--', label=f'直達波 (V1={v1})')
            
            # 折射波只在臨界距離後出現，但為了圖表交點清楚，畫全長
            ax_tx.plot(x, t_refract, 'r-', label=f'折射波 (V2={v2})')
            
            # 標示交點
            # 理論上在 Xcr 兩線相交
            t_cr = x_cr / v1 # 或 x_cr/v2 + t_intercept
            ax_tx.plot(x_cr, t_cr, 'ko', markersize=10)
            ax_tx.annotate(f'交錯點\n({x_cr:.1f}m)', xy=(x_cr, t_cr), xytext=(x_cr+10, t_cr-0.05),
                           arrowprops=dict(facecolor='black', shrink=0.05))
            
            ax_tx.set_xlabel("距離 X (m)")
            ax_tx.set_ylabel("時間 T (s)")
            ax_tx.set_title("走時曲線圖 (T-X Diagram)")
            ax_tx.legend()
            ax_tx.grid(True)
            
            st.pyplot(fig_tx)
            
            st.info("""
            **看圖說故事：**
            *   藍線是直達波，一開始它最快。
            *   紅線是折射波，它走了比較遠的路（下到第二層再上來），但因為第二層速度 $V_2$ 快，所以最後**超車**了！
            *   黑點就是超車的瞬間 ($X_{cr}$)。
            """)

st.markdown("---")
st.caption("Designed for tired students by Research Lab. 2025.")
