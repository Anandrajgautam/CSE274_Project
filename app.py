"""
================================================================================
NEUROLOGICAL AI DETECTOR - UI
Run this after training models with train_models.py
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tempfile
import warnings
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    import parselmouth
    from parselmouth.praat import call
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"Audio libraries not available: {e}")

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Neurological AI Detector",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# AUDIO FEATURE EXTRACTOR FOR PARKINSON'S (Improved)
# ============================================================================

def extract_parkinsons_features_from_audio(audio_path):
    """
    Extract voice features from audio recording.
    Note: RPDE and DFA are approximated for demonstration.
    For production, use nolds library for proper nonlinear dynamics.
    """
    if not AUDIO_AVAILABLE:
        st.error("Audio libraries not installed. Please install: pip install librosa soundfile praat-parselmouth")
        return None
    
    try:
        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        
        if len(pitch_values) == 0:
            pitch_values = [0]
        
        # Frequency features
        MDVP_Fo = float(np.mean(pitch_values))
        MDVP_Fhi = float(np.max(pitch_values))
        MDVP_Flo = float(np.min(pitch_values))
        
        # Jitter features
        MDVP_Jitter_percent = float(pitch.get_jitter(local=True))
        MDVP_Jitter_abs = float(pitch.get_jitter(local=True, absolute=True))
        MDVP_RAP = float(pitch.get_jitter(rap=True))
        MDVP_PPQ = float(pitch.get_jitter(ppq=True))
        Jitter_DDP = float(pitch.get_jitter(ddp=True))
        
        # Shimmer features
        intensity = sound.to_intensity()
        MDVP_Shimmer = float(intensity.get_shimmer(local=True))
        MDVP_Shimmer_dB = float(intensity.get_shimmer(local_dB=True))
        Shimmer_APQ3 = float(intensity.get_shimmer(apq3=True))
        Shimmer_APQ5 = float(intensity.get_shimmer(apq5=True))
        MDVP_APQ = float(intensity.get_shimmer(apq11=True))
        Shimmer_DDA = float(intensity.get_shimmer(dda=True))
        
        # Voice quality
        harmonicity = sound.to_harmonicity()
        hnr = float(harmonicity.get_value())
        if hnr == -200:
            hnr = 0
        NHR = 1 / hnr if hnr > 0 else 1
        
        # Complex features (APPROXIMATIONS - for demonstration only)
        samples = sound.values[0]
        
        rpde_approx = float(np.std(np.diff(pitch_values)) / (np.mean(np.abs(pitch_values)) + 1e-6))
        rpde_approx = np.clip(rpde_approx, 0, 1)
        
        dfa_approx = float(np.std(samples) / (np.mean(np.abs(samples)) + 1e-6))
        dfa_approx = np.clip(dfa_approx, 0.3, 1.0)
        
        spread1 = float(np.std(samples) / (np.mean(np.abs(samples)) + 1e-6))
        spread2 = float(np.percentile(np.abs(samples), 75))
        D2 = float(np.mean(np.abs(np.diff(samples))))
        
        if len(pitch_values) > 1:
            ppe = float(-np.sum(np.diff(pitch_values) ** 2) / len(pitch_values))
        else:
            ppe = 0.3
        PPE = np.clip(ppe, 0, 1)
        
        features = [
            MDVP_Fo, MDVP_Fhi, MDVP_Flo,
            MDVP_Jitter_percent, MDVP_Jitter_abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
            MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA,
            NHR, hnr,
            rpde_approx, dfa_approx, spread1, spread2, D2, PPE
        ]
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


# ============================================================================
# LOAD MODELS (With proper error handling)
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models with proper error handling"""
    models = {}
    
    try:
        data = joblib.load('models/parkinsons_model.pkl')
        models['parkinsons'] = data
        st.success("✅ Parkinson's model loaded")
    except FileNotFoundError:
        st.error("❌ Parkinson's model not found. Run train_models.py first!")
    except Exception as e:
        st.error(f"❌ Error loading Parkinson's model: {e}")
    
    try:
        data = joblib.load('models/alzheimers_model.pkl')
        models['alzheimers'] = data
        st.success("✅ Alzheimer's model loaded")
    except FileNotFoundError:
        st.error("❌ Alzheimer's model not found. Run train_models.py first!")
    except Exception as e:
        st.error(f"❌ Error loading Alzheimer's model: {e}")
    
    try:
        data = joblib.load('models/stroke_model.pkl')
        models['stroke'] = data
        st.success("✅ Stroke model loaded")
    except FileNotFoundError:
        st.error("❌ Stroke model not found. Run train_models.py first!")
    except Exception as e:
        st.error(f"❌ Error loading Stroke model: {e}")
    
    return models


# ============================================================================
# ALZHEIMER'S UI - MATCHING EXACT DATASET FEATURES (34 features)
# ============================================================================

def alzheimers_ui(model_data):
    """
    Alzheimer's Disease UI - Matches the exact 34 features from the dataset:
    PatientID, Age, Gender, Ethnicity, EducationLevel, BMI, Smoking, AlcoholConsumption,
    PhysicalActivity, DietQuality, SleepQuality, FamilyHistoryAlzheimers,
    CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension, SystolicBP,
    DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides,
    MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL, Confusion,
    Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness, Diagnosis
    """
    
    st.markdown("## 🧠 Alzheimer's Disease Prediction")
    st.markdown("### Complete Clinical Assessment (Based on Research Dataset)")
    
    # Create tabs for organized input
    tab_demo, tab_medical, tab_cognitive, tab_symptoms = st.tabs([
        "📋 Demographics & Lifestyle", "🏥 Medical History", "🧠 Cognitive Assessment", "⚠️ Symptoms"
    ])
    
    # Store features in a dictionary
    features = {}
    
    with tab_demo:
        st.markdown("#### Demographic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            features['Age'] = st.number_input("Age (years)", 40, 90, 75, 
                                              help="Age range in dataset: 60-90 years")
        
        with col2:
            features['Gender'] = st.selectbox("Gender", ["Female", "Male"], 
                                              help="0=Female, 1=Male")
        
        with col3:
            features['Ethnicity'] = st.selectbox("Ethnicity", 
                                                 ["Caucasian", "African American", "Hispanic", "Asian", "Other"],
                                                 help="0=Caucasian, 1=African American, 2=Hispanic, 3=Asian, 4=Other")
        
        st.markdown("#### Lifestyle Factors")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            features['EducationLevel'] = st.slider("Education Level", 0, 5, 2,
                                                   help="0=No formal, 1=Primary, 2=Secondary, 3=Tertiary, 4=Post-graduate, 5=Doctorate")
        
        with col5:
            height = st.number_input("Height (cm)", 140, 200, 165)
            weight = st.number_input("Weight (kg)", 40, 150, 70)
            features['BMI'] = weight / ((height/100) ** 2)
            st.metric("BMI", f"{features['BMI']:.1f}")
        
        with col6:
            features['Smoking'] = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"], 
                                               help="0=Non-smoker, 1=Smoker")
            features['Smoking'] = 1 if features['Smoking'] == "Smoker" else 0
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            features['AlcoholConsumption'] = st.slider("Alcohol Consumption (drinks/week)", 0, 20, 5,
                                                        help="Weekly alcohol intake in drinks")
        
        with col8:
            features['PhysicalActivity'] = st.slider("Physical Activity (hours/week)", 0, 20, 5,
                                                      help="Weekly physical activity in hours")
        
        with col9:
            features['DietQuality'] = st.slider("Diet Quality Score", 0, 10, 5,
                                                help="0=Poor, 10=Excellent Mediterranean diet")
        
        features['SleepQuality'] = st.slider("Sleep Quality Score", 0, 10, 6,
                                              help="0=Poor, 10=Excellent")
    
    with tab_medical:
        st.markdown("#### Medical History (Select Yes if present)")
        col1, col2 = st.columns(2)
        
        with col1:
            features['FamilyHistoryAlzheimers'] = 1 if st.selectbox("Family History of Alzheimer's", ["No", "Yes"]) == "Yes" else 0
            features['CardiovascularDisease'] = 1 if st.selectbox("Cardiovascular Disease", ["No", "Yes"]) == "Yes" else 0
            features['Diabetes'] = 1 if st.selectbox("Diabetes", ["No", "Yes"]) == "Yes" else 0
            features['Depression'] = 1 if st.selectbox("Depression", ["No", "Yes"]) == "Yes" else 0
        
        with col2:
            features['HeadInjury'] = 1 if st.selectbox("History of Head Injury", ["No", "Yes"]) == "Yes" else 0
            features['Hypertension'] = 1 if st.selectbox("Hypertension", ["No", "Yes"]) == "Yes" else 0
        
        st.markdown("#### Vital Signs & Cholesterol")
        col3, col4 = st.columns(2)
        
        with col3:
            features['SystolicBP'] = st.number_input("Systolic Blood Pressure (mmHg)", 90, 180, 120)
            features['DiastolicBP'] = st.number_input("Diastolic Blood Pressure (mmHg)", 60, 120, 80)
        
        with col4:
            features['CholesterolTotal'] = st.number_input("Total Cholesterol (mg/dL)", 150, 300, 200)
            features['CholesterolLDL'] = st.number_input("LDL Cholesterol (mg/dL)", 50, 200, 120)
            features['CholesterolHDL'] = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 50)
            features['CholesterolTriglycerides'] = st.number_input("Triglycerides (mg/dL)", 50, 400, 150)
    
    with tab_cognitive:
        st.markdown("#### Cognitive Assessment Scores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            features['MMSE'] = st.slider("MMSE Score (Mini-Mental State Exam)", 0, 30, 25,
                                         help="0-30, higher is better. <24 suggests cognitive impairment")
            
            features['FunctionalAssessment'] = st.slider("Functional Assessment Score", 0, 10, 8,
                                                         help="0=Severe impairment, 10=Normal function")
        
        with col2:
            features['ADL'] = st.slider("ADL Score (Activities of Daily Living)", 0, 10, 8,
                                        help="0=Complete dependence, 10=Complete independence")
        
        st.markdown("#### Behavioral Indicators")
        col3, col4 = st.columns(2)
        
        with col3:
            features['MemoryComplaints'] = 1 if st.selectbox("Memory Complaints", ["No", "Yes"]) == "Yes" else 0
            features['BehavioralProblems'] = 1 if st.selectbox("Behavioral Problems", ["No", "Yes"]) == "Yes" else 0
        
        with col4:
            features['DifficultyCompletingTasks'] = 1 if st.selectbox("Difficulty Completing Tasks", ["No", "Yes"]) == "Yes" else 0
            features['Forgetfulness'] = 1 if st.selectbox("Forgetfulness", ["No", "Yes"]) == "Yes" else 0
    
    with tab_symptoms:
        st.markdown("#### Presenting Symptoms (Select Yes if present)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            features['Confusion'] = 1 if st.selectbox("Confusion/Disorientation", ["No", "Yes"], key="confusion") == "Yes" else 0
        
        with col2:
            features['Disorientation'] = 1 if st.selectbox("Time/Place Disorientation", ["No", "Yes"], key="disorientation") == "Yes" else 0
        
        with col3:
            features['PersonalityChanges'] = 1 if st.selectbox("Personality Changes", ["No", "Yes"], key="personality") == "Yes" else 0
        
        st.markdown("#### Additional Clinical Information")
        
        # Add Doctor in Charge (not used in prediction but for completeness)
        doctor = st.text_input("Doctor in Charge (Optional)", placeholder="e.g., Dr. Smith")
    
    # Build feature array in the exact order expected by the model
    # Order: Age, Gender, Ethnicity, EducationLevel, BMI, Smoking, AlcoholConsumption,
    # PhysicalActivity, DietQuality, SleepQuality, FamilyHistoryAlzheimers,
    # CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension, SystolicBP,
    # DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides,
    # MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL, Confusion,
    # Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness
    
    # Map ethnicity to numeric
    ethnicity_map = {"Caucasian": 0, "African American": 1, "Hispanic": 2, "Asian": 3, "Other": 4}
    ethnicity_value = ethnicity_map[features['Ethnicity']]
    
    clinical_features = [
        features['Age'],                                    # Age
        1 if features['Gender'] == "Male" else 0,           # Gender (0=Female, 1=Male)
        ethnicity_value,                                    # Ethnicity
        features['EducationLevel'],                         # EducationLevel
        features['BMI'],                                    # BMI
        features['Smoking'],                                # Smoking
        features['AlcoholConsumption'],                     # AlcoholConsumption
        features['PhysicalActivity'],                       # PhysicalActivity
        features['DietQuality'],                            # DietQuality
        features['SleepQuality'],                           # SleepQuality
        features['FamilyHistoryAlzheimers'],                # FamilyHistoryAlzheimers
        features['CardiovascularDisease'],                  # CardiovascularDisease
        features['Diabetes'],                               # Diabetes
        features['Depression'],                             # Depression
        features['HeadInjury'],                             # HeadInjury
        features['Hypertension'],                           # Hypertension
        features['SystolicBP'],                             # SystolicBP
        features['DiastolicBP'],                            # DiastolicBP
        features['CholesterolTotal'],                       # CholesterolTotal
        features['CholesterolLDL'],                         # CholesterolLDL
        features['CholesterolHDL'],                         # CholesterolHDL
        features['CholesterolTriglycerides'],               # CholesterolTriglycerides
        features['MMSE'],                                   # MMSE
        features['FunctionalAssessment'],                   # FunctionalAssessment
        features['MemoryComplaints'],                       # MemoryComplaints
        features['BehavioralProblems'],                     # BehavioralProblems
        features['ADL'],                                    # ADL
        features['Confusion'],                              # Confusion
        features['Disorientation'],                         # Disorientation
        features['PersonalityChanges'],                     # PersonalityChanges
        features['DifficultyCompletingTasks'],              # DifficultyCompletingTasks
        features['Forgetfulness'],                          # Forgetfulness
    ]
    
    # Display feature summary
    with st.expander("📋 View All Collected Features"):
        feature_names = [
            "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking", 
            "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
            "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
            "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal",
            "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
            "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
            "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
            "Forgetfulness"
        ]
        
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": clinical_features
        })
        st.dataframe(feature_df, use_container_width=True)
    
    st.markdown("---")
    
    if st.button("🔍 Analyze Alzheimer's Risk", type="primary", use_container_width=True):
        features_array = np.array(clinical_features).reshape(1, -1)
        
        # Scale if scaler is available
        if 'scaler' in model_data and model_data['scaler'] is not None:
            features_array = model_data['scaler'].transform(features_array)
        
        model = model_data['model']
        
        # Get prediction
        probabilities = model.predict_proba(features_array)[0]
        prediction = model.predict(features_array)[0]
        
        # Probability for Alzheimer's (class 1)
        alz_probability = probabilities[1]
        
        st.markdown("---")
        
        # Display result
        col_result, col_gauge = st.columns(2)
        
        with col_result:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2>⚠️ HIGH RISK</h2>
                    <h3>Alzheimer's Disease Indicators Present</h3>
                    <p>Risk Score: {alz_probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>✅ LOW RISK</h2>
                    <h3>No Alzheimer's Disease Indicators</h3>
                    <p>Confidence: {(1-alz_probability):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_gauge:
            gauge_color = "#ff6b6b" if alz_probability > 0.6 else "#f39c12" if alz_probability > 0.3 else "#4ecdc4"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=alz_probability * 100,
                title={"text": "Alzheimer's Risk Score"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 30], "color": "#a8e6cf"},
                        {"range": [30, 60], "color": "#ffd3b6"},
                        {"range": [60, 100], "color": "#ffaaa5"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": alz_probability * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factor analysis
        st.markdown("### 📊 Risk Factor Analysis")
        
        risk_factors = []
        protective_factors = []
        
        if features['Age'] > 80:
            risk_factors.append(f"🔴 **Age >80** ({features['Age']} years) - Strong risk factor")
        elif features['Age'] > 70:
            risk_factors.append(f"🟠 **Age >70** ({features['Age']} years) - Moderate risk factor")
        
        if features['FamilyHistoryAlzheimers'] == 1:
            risk_factors.append("🔴 **Family history** of Alzheimer's disease")
        
        if features['MMSE'] < 24:
            risk_factors.append(f"🔴 **Low MMSE score** ({features['MMSE']}/30) - Cognitive impairment")
        elif features['MMSE'] < 27:
            risk_factors.append(f"🟡 **Borderline MMSE** ({features['MMSE']}/30) - Monitor closely")
        else:
            protective_factors.append(f"✅ **Normal MMSE** ({features['MMSE']}/30)")
        
        if features['MemoryComplaints'] == 1:
            risk_factors.append("🟡 **Memory complaints** reported")
        
        if features['Confusion'] == 1 or features['Disorientation'] == 1:
            risk_factors.append("🟡 **Disorientation/Confusion** present")
        
        if features['PersonalityChanges'] == 1:
            risk_factors.append("🟡 **Personality changes** observed")
        
        if features['Hypertension'] == 1:
            risk_factors.append("🟡 **Hypertension** - Vascular risk factor")
        
        if features['Diabetes'] == 1:
            risk_factors.append("🟡 **Diabetes** - Metabolic risk factor")
        
        if features['CardiovascularDisease'] == 1:
            risk_factors.append("🟡 **Cardiovascular disease**")
        
        if features['PhysicalActivity'] > 10:
            protective_factors.append("✅ **High physical activity** - Protective")
        
        if features['DietQuality'] > 7:
            protective_factors.append("✅ **Good diet quality** - Mediterranean diet benefits")
        
        if features['EducationLevel'] >= 3:
            protective_factors.append("✅ **Higher education** - Cognitive reserve")
        
        if risk_factors:
            st.markdown("#### ⚠️ Risk Factors Identified:")
            for factor in risk_factors:
                st.write(factor)
        else:
            st.markdown("#### ✅ No Major Risk Factors Identified")
        
        if protective_factors:
            st.markdown("#### 🛡️ Protective Factors:")
            for factor in protective_factors:
                st.write(factor)
        
        # Clinical recommendations
        st.markdown("### 📋 Clinical Recommendations")
        
        if alz_probability > 0.7:
            st.error("""
            **🚨 HIGH RISK - Immediate Action Required**
            
            **Neurological Referral (Urgent):**
            - Comprehensive neuropsychological assessment within 2 weeks
            - Neurology consultation for definitive diagnosis
            - Brain imaging (MRI with volumetric analysis)
            
            **Clinical Management:**
            - Consider cholinesterase inhibitors if appropriate
            - Cognitive rehabilitation referral
            - Caregiver education and support
            """)
        elif alz_probability > 0.4:
            st.warning("""
            **⚠️ MODERATE RISK - Schedule Evaluation**
            
            **Recommended Actions:**
            - Neuropsychological screening within 1 month
            - Cognitive specialist referral
            - Vascular risk factor optimization
            
            **Lifestyle Interventions:**
            - Mediterranean diet implementation
            - Regular aerobic exercise (150 min/week)
            - Cognitive stimulation activities
            """)
        else:
            st.success("""
            **✅ LOW RISK - Routine Health Maintenance**
            
            **Recommendations:**
            - Annual cognitive screening
            - Maintain healthy lifestyle
            - Regular physical and cognitive activity
            - Routine follow-up in 12 months
            """)
        
        st.markdown("---")
        st.caption("""
        **Clinical Decision Support Tool** - This prediction is based on clinical features and should be used 
        as a screening tool only. Final diagnosis requires comprehensive neurological evaluation.
        """)


# ============================================================================
# PARKINSON'S UI (With Audio Upload)
# ============================================================================

def parkinsons_ui(model_data):
    """Parkinson's Disease UI with Audio Upload"""
    
    st.markdown("## 🎤 Parkinson's Disease Prediction")
    
    tab1, tab2 = st.tabs(["🎙️ Upload Voice Recording", "📝 Enter Manual Features"])
    features = None
    
    with tab1:
        st.markdown("### 🎙️ Voice Recording Analysis")
        st.info("📌 **Instructions:** Record the patient saying a sustained vowel sound (like 'ahhh...') for 5-10 seconds.")
        
        if not AUDIO_AVAILABLE:
            st.warning("⚠️ Audio libraries not installed. Install with: pip install librosa soundfile praat-parselmouth")
        
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            help="Upload a clear voice recording. WAV format is best."
        )
        
        if audio_file is not None and AUDIO_AVAILABLE:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                audio_path = tmp_file.name
            
            st.audio(audio_file, format='audio/wav')
            
            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
                
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(np.linspace(0, duration, len(y)), y)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Voice Waveform')
                ax.set_xlim(0, duration)
                st.pyplot(fig)
                plt.close()
                
                st.info(f"📊 Duration: {duration:.1f}s | Sample Rate: {sr} Hz")
                
                if duration < 2:
                    st.warning("⚠️ Recording is very short. For best results, record at least 5 seconds.")
                
            except Exception as e:
                st.warning(f"Could not display waveform: {e}")
            
            if st.button("🎯 Extract Features from Audio", type="primary"):
                with st.spinner("Analyzing voice recording..."):
                    features = extract_parkinsons_features_from_audio(audio_path)
                
                if features is not None:
                    st.success("✅ Features extracted successfully!")
                    
                    with st.expander("📊 View Extracted Voice Features"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MDVP:Fo(Hz)", f"{features[0]:.2f}")
                            st.metric("MDVP:Fhi(Hz)", f"{features[1]:.2f}")
                            st.metric("MDVP:Flo(Hz)", f"{features[2]:.2f}")
                        with col2:
                            st.metric("Jitter(%)", f"{features[3]:.5f}")
                            st.metric("Shimmer", f"{features[8]:.5f}")
                            st.metric("HNR", f"{features[15]:.2f}")
                        with col3:
                            st.metric("RPDE (approx)", f"{features[16]:.4f}")
                            st.metric("DFA (approx)", f"{features[17]:.4f}")
                    
                    st.session_state['parkinson_features'] = features
                    st.session_state['features_source'] = 'audio'
                    
        elif audio_file is not None and not AUDIO_AVAILABLE:
            st.error("Audio processing libraries not installed. Please run: pip install librosa soundfile praat-parselmouth")
    
    with tab2:
        st.markdown("### 📝 Manual Voice Feature Entry")
        st.markdown("Enter the **22 voice measurements** manually:")
        
        col1, col2, col3 = st.columns(3)
        manual_features = []
        
        with col1:
            st.markdown("#### 📊 Frequency")
            fo = st.number_input("MDVP:Fo(Hz)", value=120.0, format="%.3f", key="fo")
            fhi = st.number_input("MDVP:Fhi(Hz)", value=157.0, format="%.3f", key="fhi")
            flo = st.number_input("MDVP:Flo(Hz)", value=75.0, format="%.3f", key="flo")
            manual_features.extend([fo, fhi, flo])
            
            st.markdown("#### 📈 Jitter")
            jitter_pct = st.number_input("MDVP:Jitter(%)", value=0.00784, format="%.6f", key="jitter_pct")
            jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00007, format="%.6f", key="jitter_abs")
            rap = st.number_input("MDVP:RAP", value=0.00370, format="%.6f", key="rap")
            ppq = st.number_input("MDVP:PPQ", value=0.00554, format="%.6f", key="ppq")
            jitter_ddp = st.number_input("Jitter:DDP", value=0.01109, format="%.6f", key="ddp")
            manual_features.extend([jitter_pct, jitter_abs, rap, ppq, jitter_ddp])
        
        with col2:
            st.markdown("#### 📉 Shimmer")
            shimmer = st.number_input("MDVP:Shimmer", value=0.04374, format="%.6f", key="shimmer")
            shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.426, format="%.3f", key="shimmer_db")
            apq3 = st.number_input("Shimmer:APQ3", value=0.02182, format="%.6f", key="apq3")
            apq5 = st.number_input("Shimmer:APQ5", value=0.03130, format="%.6f", key="apq5")
            apq = st.number_input("MDVP:APQ", value=0.02971, format="%.6f", key="apq")
            shimmer_dda = st.number_input("Shimmer:DDA", value=0.06545, format="%.6f", key="shimmer_dda")
            manual_features.extend([shimmer, shimmer_db, apq3, apq5, apq, shimmer_dda])
            
            st.markdown("#### 🎵 Voice Quality")
            nhr = st.number_input("NHR", value=0.02211, format="%.6f", key="nhr")
            hnr = st.number_input("HNR", value=21.033, format="%.3f", key="hnr")
            manual_features.extend([nhr, hnr])
        
        with col3:
            st.markdown("#### 🔬 Non-linear (approx)")
            st.caption("Note: These are approximations. For clinical use, use specialized software.")
            rpde = st.number_input("RPDE", value=0.414783, format="%.6f", key="rpde")
            dfa = st.number_input("DFA", value=0.815285, format="%.6f", key="dfa")
            spread1 = st.number_input("spread1", value=-4.813031, format="%.6f", key="spread1")
            spread2 = st.number_input("spread2", value=0.266482, format="%.6f", key="spread2")
            d2 = st.number_input("D2", value=2.301442, format="%.6f", key="d2")
            ppe = st.number_input("PPE", value=0.284654, format="%.6f", key="ppe")
            manual_features.extend([rpde, dfa, spread1, spread2, d2, ppe])
        
        if st.button("✓ Use Manual Features", key="manual_btn"):
            features = np.array(manual_features)
            st.session_state['parkinson_features'] = features
            st.session_state['features_source'] = 'manual'
            st.success("✅ Manual features saved!")
    
    st.markdown("---")
    
    if st.button("🔍 Analyze Parkinson's Risk", type="primary", use_container_width=True):
        if 'parkinson_features' in st.session_state:
            features = st.session_state['parkinson_features']
            source = st.session_state.get('features_source', 'unknown')
            st.info(f"📊 Using features from: {source.upper()} input")
        else:
            st.warning("⚠️ Please upload audio or enter features first.")
            return
        
        features_array = features.reshape(1, -1)
        
        if 'scaler' in model_data:
            features_array = model_data['scaler'].transform(features_array)
        
        model = model_data['model']
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][1]
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2>⚠️ HIGH RISK</h2>
                    <h3>Parkinson's Disease Detected</h3>
                    <p>Confidence: {probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>✅ LOW RISK</h2>
                    <h3>No Parkinson's Disease Detected</h3>
                    <p>Confidence: {(1-probability):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100 if prediction == 1 else (1-probability) * 100,
                title={"text": "Confidence Score"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff6b6b" if prediction == 1 else "#4ecdc4"}
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📋 Clinical Recommendation")
        if probability > 0.7:
            st.error("🚨 **URGENT**: Consult neurologist immediately")
        elif probability > 0.5:
            st.warning("⚠️ **Moderate Risk**: Schedule neurological evaluation")
        else:
            st.success("✅ **Low Risk**: Routine monitoring recommended")


# ============================================================================
# STROKE UI
# ============================================================================

def stroke_ui(model_data):
    """Stroke Risk UI"""
    
    st.markdown("## ❤️ Stroke Risk Prediction")
    
    col1, col2, col3 = st.columns(3)
    features = []
    
    with col1:
        st.markdown("### Demographics")
        age = st.number_input("Age", 0, 120, 65)
        gender = st.selectbox("Gender", ["Female", "Male"])
        features.extend([1 if gender == "Male" else 0, age])
        
        st.markdown("### Medical History")
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        married = st.selectbox("Ever Married", ["No", "Yes"])
        features.extend([
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            1 if married == "Yes" else 0
        ])
    
    with col2:
        st.markdown("### Lifestyle")
        work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        work_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
        features.append(work_map[work])
        
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        features.append(1 if residence == "Urban" else 0)
        
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        smoke_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}
        features.append(smoke_map[smoking])
    
    with col3:
        st.markdown("### Clinical Measurements")
        glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        features.extend([glucose, bmi])
    
    if st.button("🔍 Assess Stroke Risk", type="primary", use_container_width=True):
        features_array = np.array(features).reshape(1, -1)
        
        if 'scaler' in model_data:
            features_array = model_data['scaler'].transform(features_array)
        
        model = model_data['model']
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][1]
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2>⚠️ HIGH STROKE RISK</h2>
                    <p>Risk Score: {probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>✅ LOW STROKE RISK</h2>
                    <p>Risk Score: {(1-probability):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100 if prediction == 1 else (1-probability) * 100,
                title={"text": "Risk Score"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff6b6b" if prediction == 1 else "#4ecdc4"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "red"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📋 Clinical Recommendation")
        if probability > 0.6:
            st.error("🚨 **HIGH RISK**: Immediate medical consultation and lifestyle intervention recommended")
        elif probability > 0.4:
            st.warning("⚠️ **Moderate Risk**: Schedule cardiovascular assessment")
        else:
            st.success("✅ **Low Risk**: Continue healthy lifestyle")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Neurological AI Disease Detector</h1>
        <p>Semi-Supervised Learning | Early Detection of Neurological Conditions</p>
        <p><small>✨ Only 20% labeled data needed | 80% reduction in labeling costs</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    
    if not models:
        st.error("⚠️ No models loaded! Please run train_models.py first.")
        return
    
    disease = st.radio(
        "Select Condition",
        ["🧠 Alzheimer's Disease", "🎤 Parkinson's Disease", "❤️ Stroke Risk"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if disease == "🎤 Parkinson's Disease":
        parkinsons_ui(models['parkinsons'])
    elif disease == "🧠 Alzheimer's Disease":
        alzheimers_ui(models['alzheimers'])
    else:
        stroke_ui(models['stroke'])
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 1rem;">
        <p>⚠️ This is a decision support tool. Always consult with medical professionals.</p>
        <p>Powered by Semi-Supervised Learning | XGBoost | Gradient Boosting</p>
        <p><small>Note: RPDE and DFA values are approximations. For clinical use, use specialized software.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()