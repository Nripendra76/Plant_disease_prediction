import streamlit as st
import numpy as np
from PIL import Image
import os
import sys
import json

# Ensure local modules are accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from models.leaf_validator import validate_leaf_image
from config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE
from database import (
    authenticate, create_user, save_prediction,
    get_user_history, get_user_stats,
    get_all_users, get_platform_stats, delete_user, clear_all_predictions
)

# Safe Framework Detection
HAS_TORCH = False
HAS_TF = False
try:
    import torch
    import torchvision.transforms as T
    from models.pytorch_model import OptimizedPlantCNN, build_pytorch_mobilenet_v3
    HAS_TORCH = True
except ImportError:
    pass
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    pass

# Page Config
st.set_page_config(page_title="PlantGuard AI", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════
#  PREMIUM CSS THEME
# ══════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    *, *::before, *::after { font-family: 'Inter', sans-serif !important; }
    .stApp { background: linear-gradient(135deg, #0a0f1a 0%, #0d1b2a 25%, #1b2838 50%, #162029 75%, #0a0f1a 100%); color: #e0e6ed; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #162029 100%) !important; border-right: 1px solid rgba(46, 204, 113, 0.15); }
    section[data-testid="stSidebar"] .stMarkdown h1, section[data-testid="stSidebar"] .stMarkdown h2, section[data-testid="stSidebar"] .stMarkdown h3 { color: #2ecc71 !important; }
    h1, h2, h3 { color: #e0e6ed !important; }
    h1 { font-weight: 800 !important; letter-spacing: -0.5px; }

    .glass-card { background: rgba(255,255,255,0.04); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px; margin: 12px 0; transition: all 0.3s ease; }
    .glass-card:hover { background: rgba(255,255,255,0.07); border-color: rgba(46,204,113,0.3); transform: translateY(-2px); box-shadow: 0 8px 32px rgba(46,204,113,0.1); }

    .hero-container { background: linear-gradient(135deg, rgba(46,204,113,0.15) 0%, rgba(39,174,96,0.08) 50%, rgba(22,160,133,0.12) 100%); border: 1px solid rgba(46,204,113,0.2); border-radius: 20px; padding: 48px 40px; text-align: center; margin-bottom: 32px; position: relative; overflow: hidden; }
    .hero-container::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(46,204,113,0.06) 0%, transparent 70%); animation: pulse 6s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(1.05); opacity: 1; } }
    .hero-title { font-size: 2.6rem; font-weight: 900; background: linear-gradient(135deg, #2ecc71, #27ae60, #16a085); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 12px; position: relative; }
    .hero-subtitle { font-size: 1.15rem; color: #8fa3b8; font-weight: 400; position: relative; }

    .metric-row { display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }
    .metric-card { flex: 1; min-width: 140px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px; text-align: center; transition: all 0.3s ease; }
    .metric-card:hover { border-color: rgba(46,204,113,0.4); box-shadow: 0 0 20px rgba(46,204,113,0.08); }
    .metric-value { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #2ecc71, #16a085); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .metric-label { font-size: 0.8rem; color: #8fa3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

    .severity-low,.severity-none { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); border-radius: 8px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; display: inline-block; }
    .severity-medium { background: rgba(241,196,15,0.15); color: #f1c40f; border: 1px solid rgba(241,196,15,0.3); border-radius: 8px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; display: inline-block; }
    .severity-high { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); border-radius: 8px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; display: inline-block; }
    .severity-critical { background: rgba(192,57,43,0.2); color: #ff5252; border: 1px solid rgba(192,57,43,0.4); border-radius: 8px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; display: inline-block; }

    .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin: 24px 0; }
    .feature-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 24px; transition: all 0.3s ease; }
    .feature-card:hover { background: rgba(255,255,255,0.06); border-color: rgba(46,204,113,0.25); transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,0.3); }
    .feature-icon { font-size: 2rem; margin-bottom: 12px; }
    .feature-title { font-size: 1.05rem; font-weight: 700; color: #e0e6ed; margin-bottom: 6px; }
    .feature-desc { font-size: 0.85rem; color: #8fa3b8; line-height: 1.5; }

    .stButton > button { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 12px 32px !important; font-weight: 700 !important; font-size: 1rem !important; transition: all 0.3s ease !important; box-shadow: 0 4px 16px rgba(46,204,113,0.3) !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(46,204,113,0.45) !important; }

    div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px; }
    div[data-testid="stMetric"] label { color: #8fa3b8 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #2ecc71 !important; font-weight: 800 !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #2ecc71, #16a085) !important; border-radius: 10px; }
    div[data-testid="stExpander"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 8px 20px; color: #8fa3b8; }
    .stTabs [aria-selected="true"] { background: rgba(46,204,113,0.15) !important; color: #2ecc71 !important; }
    [data-testid="stFileUploader"] { background: rgba(255,255,255,0.03); border: 2px dashed rgba(46,204,113,0.3); border-radius: 16px; padding: 20px; }
    hr { border-color: rgba(255,255,255,0.06) !important; }

    .auth-container { max-width: 440px; margin: 60px auto; }
    .auth-card { background: rgba(255,255,255,0.04); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 40px; }
    .user-badge { background: rgba(46,204,113,0.12); border: 1px solid rgba(46,204,113,0.25); border-radius: 10px; padding: 8px 16px; text-align: center; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.auth_page = "login"


# ══════════════════════════════════════════════════
#  AUTH PAGES (LOGIN / SIGNUP)
# ══════════════════════════════════════════════════
def show_login_page():
    st.markdown('<div class="hero-container"><div class="hero-title">🌿 PlantGuard AI</div><div class="hero-subtitle">Sign in to access the Plant Disease Recognition System</div></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        tab_login, tab_signup, tab_admin = st.tabs(["🔑 Login", "📝 Sign Up", "👑 Admin Login"])

        with tab_login:
            st.markdown("### Welcome Back!")
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")

            if st.button("🔓 Sign In", key="btn_login"):
                if username and password:
                    user = authenticate(username, password)
                    if user:
                        if user["role"] == "admin":
                            st.warning("⚠️ Admin accounts must use the **👑 Admin Login** tab.")
                        else:
                            st.session_state.logged_in = True
                            st.session_state.user = user
                            st.success(f"Welcome back, **{user['username']}**! 🎉")
                            st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")
                else:
                    st.warning("Please enter both username and password.")

        with tab_signup:
            st.markdown("### Create Account")
            new_user = st.text_input("Username", key="signup_user", placeholder="Choose a username (3+ chars)")
            new_email = st.text_input("Email", key="signup_email", placeholder="your@email.com")
            new_pass = st.text_input("Password", type="password", key="signup_pass", placeholder="Choose a password (6+ chars)")
            new_pass2 = st.text_input("Confirm Password", type="password", key="signup_pass2", placeholder="Repeat password")

            if st.button("🚀 Create Account", key="btn_signup"):
                if new_pass != new_pass2:
                    st.error("❌ Passwords don't match.")
                elif new_user and new_email and new_pass:
                    success, msg = create_user(new_user, new_email, new_pass)
                    if success:
                        st.success(f"✅ {msg} You can now sign in.")
                    else:
                        st.error(f"❌ {msg}")
                else:
                    st.warning("Please fill in all fields.")

        with tab_admin:
            st.markdown("### 👑 Administrator Access")
            st.markdown('<div class="glass-card" style="padding:16px;"><div style="font-size:0.85rem;color:#8fa3b8;">🔒 This section is reserved for authorized administrators only. Unauthorized access attempts are logged.</div></div>', unsafe_allow_html=True)
            admin_user = st.text_input("Admin Username", key="admin_user", placeholder="Enter admin username")
            admin_pass = st.text_input("Admin Password", type="password", key="admin_pass", placeholder="Enter admin password")

            if st.button("🔐 Admin Sign In", key="btn_admin"):
                if admin_user and admin_pass:
                    user = authenticate(admin_user, admin_pass)
                    if user and user["role"] == "admin":
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.success(f"Welcome, Administrator **{user['username']}**! 👑")
                        st.rerun()
                    elif user:
                        st.error("❌ This account does not have admin privileges.")
                    else:
                        st.error("❌ Invalid admin credentials.")
                else:
                    st.warning("Please enter admin credentials.")


# ══════════════════════════════════════════════════
#  CHECK AUTH — SHOW LOGIN OR MAIN APP
# ══════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login_page()
    st.stop()


# ══════════════════════════════════════════════════
#  DISEASE DATABASE (38 classes)
# ══════════════════════════════════════════════════
DISEASE_DB = {
    'Apple___Apple_scab': {'cause':'Caused by the fungus **Venturia inaequalis**.','symptoms':'Olive-green to dark brown velvety spots on leaves and fruit.','solution':['Apply fungicides (captan or myclobutanil) at bud break.','Remove fallen infected leaves in autumn.','Plant scab-resistant varieties.','Prune for air circulation.','Avoid overhead irrigation.'],'severity':'Medium','type':'🦠 Fungal','reference_image':os.path.join(BASE_DIR,'sample_images','sample_apple_cedar_rust.jpg')},
    'Apple___Black_rot': {'cause':'Caused by the fungus **Botryosphaeria obtusa**.','symptoms':'Purple-bordered brown "frog-eye" lesions. Expanding fruit rot.','solution':['Prune dead wood and mummified fruit.','Apply fungicides from bloom.','Remove all infected debris.','Maintain tree vigor.','Avoid wounding bark.'],'severity':'High','type':'🦠 Fungal','reference_image':None},
    'Apple___Cedar_apple_rust': {'cause':'Caused by **Gymnosporangium juniperi-virginianae**. Requires two hosts.','symptoms':'Bright yellow-orange spots on leaves with tiny black dots.','solution':['Remove nearby juniper/cedar trees.','Apply fungicides at pink bud stage.','Plant rust-resistant varieties.','Remove juniper galls in spring.'],'severity':'Medium','type':'🦠 Fungal','reference_image':os.path.join(BASE_DIR,'sample_images','sample_apple_cedar_rust.jpg')},
    'Apple___healthy': {'cause':'No disease detected.','symptoms':'Healthy vibrant green foliage.','solution':['Continue monitoring.','Maintain watering and pruning.','Practice orchard sanitation.'],'severity':'None','type':'✅ Healthy','reference_image':os.path.join(BASE_DIR,'sample_images','sample_healthy_leaf.jpg')},
    'Blueberry___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Maintain acidic soil (pH 4.5–5.5).','Prune old canes.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Cherry_(including_sour)___Powdery_mildew': {'cause':'Caused by **Podosphaera clandestina**.','symptoms':'White powdery coating on leaves. Leaves curl and yellow.','solution':['Apply sulfur-based fungicides.','Prune for air circulation.','Avoid excess nitrogen.','Water at the base.'],'severity':'Medium','type':'🦠 Fungal','reference_image':None},
    'Cherry_(including_sour)___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Apply dormant sprays in late winter.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'cause':'Caused by **Cercospora zeae-maydis**.','symptoms':'Rectangular gray-to-tan lesions parallel to leaf veins.','solution':['Plant resistant hybrids.','Rotate with non-host crops.','Till crop residue.','Apply foliar fungicides.'],'severity':'High','type':'🦠 Fungal','reference_image':None},
    'Corn_(maize)___Common_rust_': {'cause':'Caused by **Puccinia sorghi**. Wind-blown spores.','symptoms':'Small reddish-brown pustules on both leaf surfaces.','solution':['Plant rust-resistant hybrids.','Apply foliar fungicides if severe.','Scout during cool, humid weather.'],'severity':'Medium','type':'🦠 Fungal','reference_image':os.path.join(BASE_DIR,'sample_images','sample_corn_common_rust.jpg')},
    'Corn_(maize)___Northern_Leaf_Blight': {'cause':'Caused by **Exserohilum turcicum**.','symptoms':'Long elliptical gray-green to tan lesions.','solution':['Plant resistant hybrids.','Crop rotation.','Apply fungicides at early tassel.','Till crop residue.'],'severity':'High','type':'🦠 Fungal','reference_image':None},
    'Corn_(maize)___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Maintain irrigation.','Practice crop rotation.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Grape___Black_rot': {'cause':'Caused by **Guignardia bidwellii**.','symptoms':'Tan spots with dark borders. Fruit turns into black mummies.','solution':['Destroy mummified fruit.','Apply fungicides from bud break.','Prune for air circulation.'],'severity':'High','type':'🦠 Fungal','reference_image':None},
    'Grape___Esca_(Black_Measles)': {'cause':'Complex of fungi through pruning wounds. **No cure.**','symptoms':'Tiger stripe pattern on leaves. Internal wood decay.','solution':['Protect pruning wounds.','Delay pruning to late dormant season.','Remove severely infected vines.','Apply Trichoderma biocontrol.'],'severity':'Critical','type':'🦠 Fungal','reference_image':None},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'cause':'Caused by **Pseudocercospora vitis**.','symptoms':'Dark brown spots with yellowish halos. Premature leaf drop.','solution':['Apply fungicides early.','Remove infected leaves.','Prune for air circulation.','Avoid overhead irrigation.'],'severity':'Medium','type':'🦠 Fungal','reference_image':None},
    'Grape___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Maintain canopy management.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Orange___Haunglongbing_(Citrus_greening)': {'cause':'Caused by **Candidatus Liberibacter asiaticus**. **No cure.**','symptoms':'Asymmetric blotchy yellowing. Small lopsided fruit. Tree decline.','solution':['Control psyllid vectors.','Remove infected trees.','Plant certified disease-free stock.','Monitor psyllid traps.'],'severity':'Critical','type':'🧫 Bacterial','reference_image':None},
    'Peach___Bacterial_spot': {'cause':'Caused by **Xanthomonas arboricola pv. pruni**.','symptoms':'Small dark water-soaked spots. "Shot-hole" appearance on leaves.','solution':['Plant resistant varieties.','Apply copper bactericides.','Avoid overhead irrigation.','Prune for air circulation.'],'severity':'Medium','type':'🧫 Bacterial','reference_image':None},
    'Peach___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Apply dormant sprays.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Pepper,_bell___Bacterial_spot': {'cause':'Caused by **Xanthomonas campestris pv. vesicatoria**.','symptoms':'Small dark spots with yellow halos. Scabby fruit spots.','solution':['Use disease-free seeds.','Apply copper sprays.','Crop rotation (2–3 years).','Remove infected plants.'],'severity':'Medium','type':'🧫 Bacterial','reference_image':None},
    'Pepper,_bell___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Practice crop rotation.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Potato___Early_blight': {'cause':'Caused by **Alternaria solani**.','symptoms':'Dark brown concentric "target spots" on older leaves.','solution':['Apply fungicides at first sign.','Crop rotation (2+ years).','Remove debris.','Maintain nutrition.','Use certified seed potatoes.'],'severity':'Medium','type':'🦠 Fungal','reference_image':os.path.join(BASE_DIR,'sample_images','sample_tomato_early_blight.jpg')},
    'Potato___Late_blight': {'cause':'Caused by **Phytophthora infestans** — Irish Potato Famine pathogen.','symptoms':'Large dark water-soaked lesions. White fuzzy mold. Tuber rot.','solution':['Apply fungicides preventively.','Plant resistant varieties.','Destroy volunteer plants.','Avoid overhead irrigation.'],'severity':'Critical','type':'🦠 Fungal','reference_image':None},
    'Potato___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Use certified seed potatoes.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Raspberry___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Remove old fruiting canes.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Soybean___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Practice crop rotation.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Squash___Powdery_mildew': {'cause':'Caused by **Podosphaera xanthii**.','symptoms':'White powdery growth on leaves. Leaves yellow and die.','solution':['Apply fungicides (sulfur, neem oil).','Plant resistant varieties.','Space plants for circulation.','Water at soil level.'],'severity':'Medium','type':'🦠 Fungal','reference_image':None},
    'Strawberry___Leaf_scorch': {'cause':'Caused by **Diplocarpon earlianum**.','symptoms':'Irregular dark purple to brown spots. Leaves look "scorched."','solution':['Apply fungicides from bloom.','Remove infected leaves.','Plant resistant varieties.','Use drip irrigation.'],'severity':'Medium','type':'🦠 Fungal','reference_image':None},
    'Strawberry___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Mulch to reduce soil splash.'],'severity':'None','type':'✅ Healthy','reference_image':None},
    'Tomato___Bacterial_spot': {'cause':'Caused by **Xanthomonas spp.**','symptoms':'Small dark water-soaked spots with yellow halos. Scabby fruit.','solution':['Use disease-free seeds.','Apply copper bactericides.','Crop rotation (2–3 years).','Remove infected plants.'],'severity':'Medium','type':'🧫 Bacterial','reference_image':None},
    'Tomato___Early_blight': {'cause':'Caused by **Alternaria solani**.','symptoms':'Dark brown "target spots" with concentric rings on lower leaves.','solution':['Apply fungicides at first symptom.','Mulch to prevent splash.','Crop rotation.','Stake plants.','Water at base.'],'severity':'Medium','type':'🦠 Fungal','reference_image':os.path.join(BASE_DIR,'sample_images','sample_tomato_early_blight.jpg')},
    'Tomato___Late_blight': {'cause':'Caused by **Phytophthora infestans**. Spreads extremely rapidly.','symptoms':'Large dark water-soaked lesions. White fuzzy mold underneath.','solution':['Apply fungicides preventively.','Destroy all infected material.','Do not compost.','Plant resistant varieties.'],'severity':'Critical','type':'🦠 Fungal','reference_image':os.path.join(BASE_DIR,'sample_images','sample_tomato_early_blight.jpg')},
    'Tomato___Leaf_Mold': {'cause':'Caused by **Passalora fulva**. High humidity, poor ventilation.','symptoms':'Pale yellow spots above. Olive-green velvety mold below.','solution':['Improve ventilation.','Water at base.','Apply fungicides.','Plant resistant varieties.'],'severity':'Medium','type':'🦠 Fungal','reference_image':None},
    'Tomato___Septoria_leaf_spot': {'cause':'Caused by **Septoria lycopersici**.','symptoms':'Many small circular spots with dark borders and grayish centers.','solution':['Apply fungicides at first sign.','Remove lower infected leaves.','Mulch.','Crop rotation.','Stake plants.'],'severity':'Medium','type':'🦠 Fungal','reference_image':None},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'cause':'**Two-Spotted Spider Mite** (*Tetranychus urticae*). Hot, dry conditions.','symptoms':'Yellow/white stippling. Fine webbing. Leaves bronze and drop.','solution':['Spray with strong water jet.','Apply miticides or neem oil.','Introduce predatory mites.','Increase humidity.'],'severity':'Medium','type':'🦟 Pest','reference_image':None},
    'Tomato___Target_Spot': {'cause':'Caused by **Corynespora cassiicola**.','symptoms':'Brown circular spots with concentric rings on leaves and fruit.','solution':['Apply fungicides preventively.','Remove lower leaves.','Crop rotation.','Avoid overhead irrigation.'],'severity':'Medium','type':'🦠 Fungal','reference_image':None},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'cause':'**TYLCV**, transmitted by whitefly. **No cure.**','symptoms':'Upward leaf curling. Yellow margins. Stunted growth. Flower drop.','solution':['Control whiteflies with insecticides/traps.','Use insect-proof netting.','Plant resistant varieties.','Remove infected plants.'],'severity':'Critical','type':'🧬 Viral','reference_image':None},
    'Tomato___Tomato_mosaic_virus': {'cause':'**ToMV**. Spreads through seeds, tools, hands. Persists for years.','symptoms':'Mosaic light/dark green pattern. Leaf distortion. Fruit browning.','solution':['Use virus-free seeds.','Disinfect tools with bleach.','Destroy infected plants.','Plant resistant varieties (Tm-2 genes).'],'severity':'High','type':'🧬 Viral','reference_image':None},
    'Tomato___healthy': {'cause':'No disease detected.','symptoms':'Healthy with no infection.','solution':['Continue monitoring.','Maintain watering and staking.','Practice crop rotation.'],'severity':'None','type':'✅ Healthy','reference_image':None},
}


# ══════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════
@st.cache_resource(show_spinner="🧠 Loading AI model...")
def load_trained_model():
    if HAS_TORCH:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for p, arch in [(os.path.join(BASE_DIR,"checkpoints","best_custom_cnn_pytorch.pth"),"custom_cnn"),(os.path.join(BASE_DIR,"checkpoints","best_mobilenet_v3_pytorch.pth"),"mobilenet_v3")]:
            if os.path.exists(p):
                try:
                    m = build_pytorch_mobilenet_v3(NUM_CLASSES) if arch=="mobilenet_v3" else OptimizedPlantCNN(NUM_CLASSES)
                    m.load_state_dict(torch.load(p, map_location=torch_device)); m.to(torch_device); m.eval()
                    return ("pytorch", m, p, str(torch_device))
                except Exception: continue
    if HAS_TF:
        for p in [os.path.join(BASE_DIR,"checkpoints","best_optimized_custom_cnn.keras"),os.path.join(BASE_DIR,"checkpoints","best_mobilenet_v3.keras"),os.path.join(BASE_DIR,"trained_model.keras"),os.path.join(BASE_DIR,"trained_model.h5")]:
            if os.path.exists(p):
                try:
                    m = tf.keras.models.load_model(p)
                    return ("tensorflow", m, p, "GPU" if tf.config.list_physical_devices('GPU') else "CPU")
                except Exception: continue
    return (None, None, None, None)

@st.cache_data
def load_training_history():
    hp = os.path.join(BASE_DIR, "training_hist.json")
    if os.path.exists(hp):
        with open(hp,'r') as f: return json.load(f)
    return None

def predict_leaf_disease(pil_image, framework, model, device):
    if framework == "pytorch":
        transform = T.Compose([T.Resize(IMAGE_SIZE), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    elif framework == "tensorflow":
        img = pil_image.convert("RGB").resize(IMAGE_SIZE)
        probs = model.predict(np.expand_dims(np.array(img, dtype=np.float32), axis=0), verbose=0)[0]
    else:
        probs = np.ones(NUM_CLASSES) / NUM_CLASSES
    top_idx = np.argsort(probs)[::-1][:3]
    return top_idx[0], [(CLASS_NAMES[i], float(probs[i])) for i in top_idx], probs


# ══════════════════════════════════════════════════
#  SIDEBAR (LOGGED IN)
# ══════════════════════════════════════════════════
user = st.session_state.user

st.sidebar.markdown(f"""
<div style="text-align:center;padding:16px 0;">
    <div style="font-size:2.5rem;">🌿</div>
    <div style="font-size:1.2rem;font-weight:800;background:linear-gradient(135deg,#2ecc71,#16a085);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">PlantGuard AI</div>
    <div style="font-size:0.7rem;color:#8fa3b8;letter-spacing:2px;text-transform:uppercase;">Disease Recognition System</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div class="user-badge">
    👤 <b>{user['username']}</b> &nbsp;
    <span style="font-size:0.7rem;color:#8fa3b8;">({user['role'].upper()})</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

pages = ["🏠 Home", "ℹ️ About", "📊 Model Performance", "🔬 Disease Recognition", "📜 My History"]
if user["role"] == "admin":
    pages.append("👑 Admin Dashboard")

app_mode = st.sidebar.selectbox("📄 Navigate", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Engine")
if HAS_TORCH:
    cuda_s = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"🔥 PyTorch: {cuda_s}")
if HAS_TF:
    st.sidebar.info(f"🧠 TensorFlow: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

st.sidebar.markdown('<div style="text-align:center;color:#566573;font-size:0.7rem;padding:8px;">v2.1 — Built with ❤️ for Agriculture</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  🏠 HOME
# ══════════════════════════════════════════════════
if app_mode == "🏠 Home":
    st.markdown('<div class="hero-container"><div class="hero-title">🌿 PlantGuard AI</div><div class="hero-subtitle">Advanced Plant Disease Recognition & Leaf Verification System</div></div>', unsafe_allow_html=True)
    img_p = os.path.join(BASE_DIR, "home_page.jpeg")
    if os.path.exists(img_p): st.image(img_p, width="stretch")
    st.markdown('<div class="metric-row"><div class="metric-card"><div class="metric-value">38</div><div class="metric-label">Disease Classes</div></div><div class="metric-card"><div class="metric-value">87K+</div><div class="metric-label">Training Images</div></div><div class="metric-card"><div class="metric-value">96.9%</div><div class="metric-label">Val Accuracy</div></div><div class="metric-card"><div class="metric-value">14</div><div class="metric-label">Crop Species</div></div></div>', unsafe_allow_html=True)
    st.markdown("### How It Works")
    st.markdown('<div class="feature-grid"><div class="feature-card"><div class="feature-icon">🛡️</div><div class="feature-title">Leaf Verification</div><div class="feature-desc">Botanical analysis confirms genuine crop leaves. Non-leaf objects rejected.</div></div><div class="feature-card"><div class="feature-icon">🧠</div><div class="feature-title">AI Diagnosis</div><div class="feature-desc">Deep learning classifies disease across 38 pathologies with confidence scoring.</div></div><div class="feature-card"><div class="feature-icon">💊</div><div class="feature-title">Treatment Advisory</div><div class="feature-desc">Severity ratings and step-by-step treatment plans.</div></div><div class="feature-card"><div class="feature-icon">📜</div><div class="feature-title">History Tracking</div><div class="feature-desc">All your scans saved with full diagnostic details.</div></div></div>', unsafe_allow_html=True)
    st.info("👈 Select **🔬 Disease Recognition** from the sidebar to get started!")

# ══════════════════════════════════════════════════
#  ℹ️ ABOUT
# ══════════════════════════════════════════════════
elif app_mode == "ℹ️ About":
    st.markdown('<div class="hero-container"><div class="hero-title">About PlantGuard AI</div><div class="hero-subtitle">Architecture, Dataset & Technical Details</div></div>', unsafe_allow_html=True)

    # Project Demo Video
    video_path = os.path.join(BASE_DIR, "video.mp4")
    if os.path.exists(video_path):
        st.markdown("### 🎬 Project Demo")
        st.video(video_path)
        st.markdown("")

    t1,t2 = st.tabs(["📚 Dataset","🏗️ Architecture"])
    with t1:
        st.markdown('<div class="glass-card"><h3>📚 Training Dataset</h3><p>Trained on the <b>New Plant Diseases Dataset</b> (PlantVillage benchmark):</p><ul><li><b>87,000+</b> RGB images across 38 categories</li><li><b>14</b> crop species</li><li><b>Train:</b> 70,295 | <b>Val:</b> 17,572 | <b>Test:</b> 33</li></ul></div>', unsafe_allow_html=True)
    with t2:
        st.code("Uploaded Image ──► [Stage 1: Leaf Verification]\n                       │\n                       ├──► ❌ Reject\n                       └──► ✅ [Stage 2: Deep Learning] ──► 🔬 Diagnosis", language="text")
        st.markdown('<div class="glass-card"><h3>🧠 Model</h3><ul><li><b>Baseline:</b> 10-layer CNN, 7.84M params</li><li><b>Optimized:</b> GAP model, 2.15M params</li><li><b>Transfer:</b> MobileNetV3, EfficientNetV2, ResNet50V2</li></ul></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  📊 MODEL PERFORMANCE
# ══════════════════════════════════════════════════
elif app_mode == "📊 Model Performance":
    st.markdown('<div class="hero-container"><div class="hero-title">📊 Model Performance</div><div class="hero-subtitle">Training Metrics & Convergence</div></div>', unsafe_allow_html=True)
    hist = load_training_history()
    if hist:
        import pandas as pd
        fa,fva,fl,fvl,ne = hist['accuracy'][-1]*100, hist['val_accuracy'][-1]*100, hist['loss'][-1], hist['val_loss'][-1], len(hist['accuracy'])
        st.markdown(f'<div class="metric-row"><div class="metric-card"><div class="metric-value">{fa:.1f}%</div><div class="metric-label">Train Acc</div></div><div class="metric-card"><div class="metric-value">{fva:.1f}%</div><div class="metric-label">Val Acc</div></div><div class="metric-card"><div class="metric-value">{fl:.4f}</div><div class="metric-label">Train Loss</div></div><div class="metric-card"><div class="metric-value">{fvl:.4f}</div><div class="metric-label">Val Loss</div></div><div class="metric-card"><div class="metric-value">{ne}</div><div class="metric-label">Epochs</div></div></div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("### 📈 Accuracy"); adf = pd.DataFrame({'Train':hist['accuracy'],'Val':hist['val_accuracy']}); adf.index=range(1,len(adf)+1); st.line_chart(adf)
        with c2:
            st.markdown("### 📉 Loss"); ldf = pd.DataFrame({'Train':hist['loss'],'Val':hist['val_loss']}); ldf.index=range(1,len(ldf)+1); st.line_chart(ldf)
    else:
        st.warning("📂 No training history found.")

# ══════════════════════════════════════════════════
#  🔬 DISEASE RECOGNITION
# ══════════════════════════════════════════════════
elif app_mode == "🔬 Disease Recognition":
    st.markdown('<div class="hero-container"><div class="hero-title">🔬 Disease Recognition</div><div class="hero-subtitle">Upload a crop leaf for AI-powered diagnosis</div></div>', unsafe_allow_html=True)
    framework, model, model_path, device = load_trained_model()
    if model:
        st.markdown(f'<div class="glass-card" style="border-color:rgba(46,204,113,0.3);">⚡ <b>Model:</b> <code>{os.path.basename(model_path)}</code> | <b>{framework.upper()}</b> | <b>{device}</b></div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ Verification Mode — no model checkpoint found.")

    tab_up, tab_samp = st.tabs(["📤 Upload","🧪 Samples"])
    test_image = None
    img_name = "uploaded_image"
    with tab_up:
        uf = st.file_uploader("Upload leaf image:", type=["jpg","jpeg","png"])
        if uf:
            test_image = Image.open(uf); img_name = uf.name
    with tab_samp:
        sopts = {"Select...":None,"🍎 Apple Cedar Rust":os.path.join(BASE_DIR,"sample_images","sample_apple_cedar_rust.jpg"),"🍅 Tomato Early Blight":os.path.join(BASE_DIR,"sample_images","sample_tomato_early_blight.jpg"),"🌽 Corn Common Rust":os.path.join(BASE_DIR,"sample_images","sample_corn_common_rust.jpg"),"🌿 Healthy Leaf":os.path.join(BASE_DIR,"sample_images","sample_healthy_leaf.jpg"),"🚗 Non-Leaf Car":os.path.join(BASE_DIR,"sample_images","sample_non_leaf_car.jpg")}
        sel = st.selectbox("Choose sample:", list(sopts.keys()))
        if sel != "Select..." and sopts[sel] and os.path.exists(sopts[sel]):
            test_image = Image.open(sopts[sel]); img_name = sel

    if test_image:
        st.markdown("---")
        ci, cr = st.columns([1, 1.2])
        with ci:
            st.markdown("### 📸 Input Image"); st.image(test_image, width="stretch")
        with cr:
            st.markdown("### 🔍 Results")
            if st.button("🚀 Run Diagnosis", type="primary"):
                with st.spinner("🛡️ Verifying leaf..."): is_leaf, leaf_msg, meta = validate_leaf_image(test_image)
                if not is_leaf:
                    st.error("### ❌ Not a Plant Leaf!"); st.warning(f"**Result:** {leaf_msg}")
                    st.markdown("---"); st.markdown("### 📸 Valid Examples")
                    rc = st.columns(3)
                    for i,(fn,cap) in enumerate([("sample_apple_cedar_rust.jpg","Apple Rust"),("sample_tomato_early_blight.jpg","Tomato Blight"),("sample_corn_common_rust.jpg","Corn Rust")]):
                        p=os.path.join(BASE_DIR,"sample_images",fn)
                        if os.path.exists(p):
                            with rc[i]: st.image(p, caption=cap, width="stretch")
                else:
                    st.success("✅ **Leaf Verified!**")
                    if model:
                        with st.spinner("🧠 Classifying..."):
                            best_idx, top_preds, raw = predict_leaf_disease(test_image, framework, model, device)
                            best_class, best_conf = top_preds[0]
                        parts = best_class.split("___")
                        plant_name = parts[0].replace("_"," ")
                        condition = parts[1].replace("_"," ") if len(parts)>1 else "Unknown"
                        is_healthy = "healthy" in best_class.lower()
                        info = DISEASE_DB.get(best_class)
                        sev = info['severity'] if info else 'Unknown'
                        dtype = info['type'] if info else '❓'
                        cpct = best_conf*100
                        cemoji = "🟢" if cpct>=80 else ("🟡" if cpct>=50 else "🔴")

                        # Save to database
                        top3_str = " | ".join([f"{n}: {c*100:.1f}%" for n,c in top_preds])
                        save_prediction(user["id"], img_name, best_class, plant_name, condition, best_conf, is_healthy, sev, dtype, top3_str)

                        if is_healthy:
                            st.balloons()
                            st.markdown(f'<div class="glass-card" style="border-color:rgba(46,204,113,0.4);text-align:center;"><div style="font-size:3rem;">🌿</div><div style="font-size:1.4rem;font-weight:800;color:#2ecc71;">Healthy {plant_name}!</div><div style="color:#8fa3b8;">No disease detected</div><div style="margin-top:12px;"><span style="font-size:1.6rem;font-weight:800;color:#2ecc71;">{cemoji} {cpct:.1f}%</span></div></div>', unsafe_allow_html=True)
                        else:
                            ccolor = '#2ecc71' if cpct>=80 else ('#f39c12' if cpct>=50 else '#e74c3c')
                            st.markdown(f'<div class="glass-card" style="border-color:rgba(231,76,60,0.3);"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;"><div><div style="font-size:0.8rem;color:#8fa3b8;text-transform:uppercase;">Disease Detected</div><div style="font-size:1.3rem;font-weight:800;color:#e74c3c;">{condition}</div><div style="font-size:0.9rem;color:#8fa3b8;">Plant: {plant_name}</div></div><div style="text-align:center;"><span style="font-size:1.6rem;font-weight:800;color:{ccolor};">{cemoji} {cpct:.1f}%</span><div style="color:#8fa3b8;font-size:0.75rem;">Confidence</div></div></div><div style="margin-top:12px;display:flex;gap:8px;"><span class="severity-{sev.lower()}">{sev} Severity</span><span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:4px 12px;font-size:0.75rem;">{dtype}</span></div></div>', unsafe_allow_html=True)

                        with st.expander("📊 Top-3 Predictions", expanded=True):
                            for rk,(cn,cf) in enumerate(top_preds,1):
                                st.markdown(f"**{['🥇','🥈','🥉'][rk-1]} {cn.replace('___',' — ').replace('_',' ')}** — `{cf*100:.2f}%`")
                                st.progress(float(cf))
                        if info and info.get('reference_image') and os.path.exists(str(info.get('reference_image',''))):
                            with st.expander("🔬 Reference Comparison"):
                                rc1,rc2 = st.columns(2)
                                with rc1: st.image(test_image, caption="Your Upload", width="stretch")
                                with rc2: st.image(info['reference_image'], caption=f"Reference: {condition}", width="stretch")
                        with st.expander("📋 Treatment Plan", expanded=not is_healthy):
                            if info:
                                st.markdown(f"🦠 **Cause:** {info['cause']}"); st.markdown(f"🔍 **Symptoms:** {info['symptoms']}"); st.markdown("---"); st.markdown("💊 **Treatment:**")
                                for i,s in enumerate(info['solution'],1): st.markdown(f"**{i}.** {s}")
                            elif is_healthy: st.success("🌱 Healthy plant. Continue regular care.")
                            else: st.info("Consult an agricultural specialist.")
                    else:
                        st.info("🔓 Leaf verified. Model checkpoint needed for classification.")
    else:
        st.markdown('<div class="glass-card" style="text-align:center;padding:40px;"><div style="font-size:3rem;">📤</div><div style="font-size:1.1rem;font-weight:600;color:#e0e6ed;margin-top:8px;">Upload a Plant Leaf Image</div><div style="color:#8fa3b8;margin-top:4px;">Select a file or test sample above</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  📜 MY HISTORY
# ══════════════════════════════════════════════════
elif app_mode == "📜 My History":
    st.markdown('<div class="hero-container"><div class="hero-title">📜 My Scan History</div><div class="hero-subtitle">All your past plant disease diagnoses</div></div>', unsafe_allow_html=True)

    stats = get_user_stats(user["id"])
    st.markdown(f'<div class="metric-row"><div class="metric-card"><div class="metric-value">{stats["total_scans"]}</div><div class="metric-label">Total Scans</div></div><div class="metric-card"><div class="metric-value">{stats["healthy_count"]}</div><div class="metric-label">Healthy</div></div><div class="metric-card"><div class="metric-value">{stats["diseased_count"]}</div><div class="metric-label">Diseased</div></div><div class="metric-card"><div class="metric-value">{stats["healthy_ratio"]:.0f}%</div><div class="metric-label">Health Rate</div></div></div>', unsafe_allow_html=True)

    if stats["most_common_disease"] != "N/A":
        clean = stats["most_common_disease"].replace("___"," — ").replace("_"," ")
        st.markdown(f'<div class="glass-card">🦠 <b>Most Common Disease:</b> {clean}</div>', unsafe_allow_html=True)

    history = get_user_history(user["id"])
    if history:
        import pandas as pd
        st.markdown("### 📋 Recent Predictions")
        rows = []
        for h in history:
            rows.append({
                "Date": h["timestamp"][:16] if h["timestamp"] else "",
                "Image": h["image_name"] or "",
                "Plant": h["plant_name"] or "",
                "Disease": h["disease_name"] or "",
                "Confidence": f"{h['confidence']*100:.1f}%",
                "Severity": h["severity"] or "",
                "Status": "✅ Healthy" if h["is_healthy"] else "⚠️ Diseased"
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("📭 No scans yet. Go to **🔬 Disease Recognition** to scan your first leaf!")

# ══════════════════════════════════════════════════
#  👑 ADMIN DASHBOARD
# ══════════════════════════════════════════════════
elif app_mode == "👑 Admin Dashboard":
    st.markdown('<div class="hero-container"><div class="hero-title">👑 Admin Dashboard</div><div class="hero-subtitle">Platform Statistics & User Management</div></div>', unsafe_allow_html=True)

    pstats = get_platform_stats()

    st.markdown(f'<div class="metric-row"><div class="metric-card"><div class="metric-value">{pstats["total_users"]}</div><div class="metric-label">Total Users</div></div><div class="metric-card"><div class="metric-value">{pstats["total_predictions"]}</div><div class="metric-label">Total Scans</div></div><div class="metric-card"><div class="metric-value">{pstats["total_healthy"]}</div><div class="metric-label">Healthy</div></div><div class="metric-card"><div class="metric-value">{pstats["total_diseased"]}</div><div class="metric-label">Diseased</div></div></div>', unsafe_allow_html=True)

    at1, at2, at3 = st.tabs(["👥 Users", "📊 Disease Stats", "🕐 Recent Activity"])

    with at1:
        st.markdown("### 👥 Registered Users")
        users = get_all_users()
        if users:
            import pandas as pd
            urows = [{"Username":u["username"],"Email":u["email"],"Role":u["role"].upper(),"Scans":u["scan_count"],"Joined":str(u["created_at"])[:10]} for u in users]
            st.dataframe(pd.DataFrame(urows), width="stretch", hide_index=True)

    with at2:
        st.markdown("### 🦠 Top Detected Diseases")
        if pstats["disease_distribution"]:
            import pandas as pd
            dd = pstats["disease_distribution"]
            for d in dd:
                clean = d["predicted_class"].replace("___"," — ").replace("_"," ")
                st.markdown(f"**{clean}** — `{d['cnt']} detections`")
                st.progress(min(float(d["cnt"]) / max(dd[0]["cnt"],1), 1.0))
        else:
            st.info("No disease detections yet.")

    with at3:
        st.markdown("### 🕐 Recent Predictions")
        if pstats["recent_predictions"]:
            import pandas as pd
            rrows = [{"User":r["username"],"Disease":r["disease_name"] or "","Confidence":f"{r['confidence']*100:.1f}%","Status":"✅" if r["is_healthy"] else "⚠️","Time":str(r["timestamp"])[:16]} for r in pstats["recent_predictions"]]
            st.dataframe(pd.DataFrame(rrows), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### ⚠️ Danger Zone")
    dc1, dc2 = st.columns(2)
    with dc1:
        if st.button("🗑️ Clear All Prediction History"):
            clear_all_predictions()
            st.success("All prediction history cleared.")
            st.rerun()
