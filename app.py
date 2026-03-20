import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from ultralytics import YOLO
import av
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3
import hashlib
from datetime import datetime
import io

# --- 1. DATABASE & AUTH LOGIC ---
def init_db():
    conn = sqlite3.connect('waste_history.db')
    c = conn.cursor()
    # Table for Detections
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT,
                  date TEXT, 
                  item_type TEXT, 
                  confidence REAL)''')
    # Table for Users
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect('waste_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, hash_password(password)))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('waste_history.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =? AND password = ?', (username, hash_password(password)))
    data = c.fetchone()
    conn.close()
    return data

def save_to_history(username, item_type, confidence):
    conn = sqlite3.connect('waste_history.db')
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history (username, date, item_type, confidence) VALUES (?, ?, ?, ?)", 
              (username, now, item_type, confidence))
    conn.commit()
    conn.close()

# Initialize DB at startup
init_db()

# --- 2. SESSION STATE MANAGEMENT ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# --- 3. LOGIN / SIGNUP UI ---
def login_page():
    st.title("🔐 EcoVision Access")
    
    tab1, tab2 = st.tabs(["Login", "Create Account"])
    
    with tab1:
        user = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Login"):
            result = login_user(user, pwd)
            if result:
                st.session_state['logged_in'] = True
                st.session_state['username'] = user
                st.success(f"Welcome back, {user}!")
                st.rerun()
            else:
                st.error("Invalid Username or Password")
                
    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pwd = st.text_input("New Password", type="password", key="reg_pwd")
        if st.button("Sign Up"):
            try:
                create_user(new_user, new_pwd)
                st.success("Account created successfully! You can now login.")
            except sqlite3.IntegrityError:
                st.error("Username already exists. Please choose another.")

# --- 4. MAIN APPLICATION ---
if not st.session_state['logged_in']:
    login_page()
else:
    # PAGE CONFIG & THEME
    st.set_page_config(page_title="EcoVision Pro", page_icon="♻️", layout="wide")

    # Custom CSS for the Green "Eco" Theme
    st.markdown("""
        <style>
        .main { background-color: #f9fbf9; }
        .stSidebar { background-color: #e8f5e9 !important; }
        h1, h2, h3 { color: #2e7d32; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
        .stButton>button { background-color: #2e7d32; color: white; border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        return YOLO("best.pt")

    model = load_model()

    DISPOSAL_GUIDE = {
        "Paper -Biodegradable-": "♻️ **Recycle:** Place in the Blue Bin. Ensure it's dry.",
        "Food -Biodegradable-": "🌱 **Compost:** Great for organic waste bins.",
        "Clothes -Biodegradable-": "👕 **Donate/Reuse:** If torn, use for textile recycling.",
        "Hazard": "⚠️ **DANGER:** Dispose of at a designated hazardous waste facility.",
        "Plastic": "🥤 **Recycle:** Rinse and place in the Plastic/Yellow bin."
    }

    # SIDEBAR
    with st.sidebar:
        # Green Bin Icon as seen in your screenshots
        st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
        st.subheader(f"👤 User: {st.session_state['username']}")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = ""
            st.rerun()
        
        st.divider()
        st.header("System Controls")
        
        # Navigation Options (Includes Admin Check)
        menu_options = ["Webcam (Live)", "Image Upload", "View History 📜"]
        if st.session_state['username'].lower() == "admin":
            menu_options.append("Admin Dashboard 📊")
            
        source_radio = st.radio("Select Navigation:", menu_options)
        st.divider()
        conf_level = st.slider("Confidence Threshold", 0.0, 1.0, 0.45)
        st.info("Lower threshold = more detections (but more errors).")

    st.title("♻️ Smart Waste Detection System")

    # WEBCAM INTERFACE
    if source_radio == "Webcam (Live)":
        st.subheader("Live Desktop Streaming")
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=conf_level)
            return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

        webrtc_streamer(key="stream", video_frame_callback=video_frame_callback,
                        media_stream_constraints={"video": True, "audio": False})

    # IMAGE UPLOAD INTERFACE
    elif source_radio == "Image Upload":
        st.subheader("File Analysis")
        uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            results = model(np.array(img), conf=conf_level)
            
            col1, col2 = st.columns(2)
            with col1: st.image(img, caption="Original Image", use_container_width=True)
            with col2: st.image(results[0].plot()[:, :, ::-1], caption="AI Prediction", use_container_width=True)
            
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.subheader("📊 Waste Analysis Dashboard")
                counts = {}
                for box in boxes:
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    counts[label] = counts.get(label, 0) + 1
                    # Log activity to DB
                    save_to_history(st.session_state['username'], label, conf) 
                
                # Metrics
                m_cols = st.columns(len(counts))
                for i, (name, count) in enumerate(counts.items()):
                    m_cols[i].metric(label=name.upper(), value=count)
                
                # Visuals & Guide
                c1, c2 = st.columns([2, 1])
                with c1:
                    df_plot = pd.DataFrame(counts.items(), columns=['Waste Type', 'Quantity'])
                    fig = px.bar(df_plot, x='Waste Type', y='Quantity', color='Waste Type', title="Detection Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.write("📝 **Disposal Instructions**")
                    for name in counts.keys():
                        st.info(DISPOSAL_GUIDE.get(name, "Check local guidelines."))
            else:
                st.warning("No waste detected.")

    # USER PERSONAL HISTORY
    elif source_radio == "View History 📜":
        st.subheader(f"📜 {st.session_state['username']}'s Detection History")
        conn = sqlite3.connect('waste_history.db')
        df = pd.read_sql_query(f"SELECT date, item_type, confidence FROM history WHERE username='{st.session_state['username']}' ORDER BY date DESC", conn)
        conn.close()

        if not df.empty:
            t1, t2, t3 = st.columns(3)
            t1.metric("Total Scans", len(df))
            t2.metric("Most Common", df['item_type'].mode()[0])
            t3.metric("Avg Confidence", f"{df['confidence'].mean():.2f}")

            st.plotly_chart(px.pie(df, names='item_type', hole=0.4, title="Your Lifetime Waste Distribution"), use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No history recorded yet for your account.")

    # ADMIN DASHBOARD (Global Access)
    elif source_radio == "Admin Dashboard 📊":
        st.subheader("📊 Global System Analytics (Admin View)")
        conn = sqlite3.connect('waste_history.db')
        df_all = pd.read_sql_query("SELECT * FROM history ORDER BY date DESC", conn)
        df_users = pd.read_sql_query("SELECT username FROM users", conn)
        conn.close()

        if not df_all.empty:
            a1, a2, a3 = st.columns(3)
            a1.metric("Total Registered Users", len(df_users))
            a2.metric("Total System Scans", len(df_all))
            a3.metric("Most Active User", df_all['username'].mode()[0])

            st.write("### Global Waste Trends")
            fig_admin = px.bar(df_all['item_type'].value_counts().reset_index(), 
                               x='item_type', y='count', color='item_type',
                               title="Waste Detected Across All Users")
            st.plotly_chart(fig_admin, use_container_width=True)
            
            st.write("### Master Audit Log")
            st.dataframe(df_all, use_container_width=True)

            # Export Audit Report (Matches your screenshot button)
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Audit Report",
                data=csv,
                file_name=f"global_audit_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )
        else:
            st.warning("No global data recorded yet.")

    st.divider()
    st.caption(f"EcoVision Pro | Logged in as: {st.session_state['username']} | Built with Streamlit & YOLO")