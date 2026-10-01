import os
import random
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

st.set_page_config(layout="wide", page_title="802.11bf Black-Box Defense Simulator")

# Read from environment or fallback to Docker Compose internal network service name
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.title("🛡️ Hardware-Quantized Universal Defense Simulator")
st.markdown("Simulating a 2-bit Reconfigurable Intelligent Surface (RIS) blinding a Black-Box Random Forest eavesdropper on real RF data.")

activity_map = {
    0: "Getting Down",
    1: "Getting Up",
    2: "Lying Down",
    3: "Empty Room",
    4: "Sitting",
    5: "Standing",
    6: "Walking",
}

if st.button("📡 Intercept Real RF Wave & Deploy Hardware UAP", type="primary"):
    with st.spinner("Intercepting physical CSI matrix and applying 2-bit phase shift..."):
        wave_id = random.randint(0, 99)
        
        try:
            # Query the backend container via internal DNS
            response = requests.get(f"{BACKEND_URL}/intercept?wave_id={wave_id}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                true_act = activity_map[data['true_label']]
                
                st.info(f"**Ground Truth:** The human in the room is actually **{true_act}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.error("### 🚨 Eavesdropper (Clean RF Wave)")
                    st.write(f"**AI Prediction:** {activity_map[data['clean_guess']]}")
                    st.write(f"**Confidence:** {data['clean_confidence']:.2f}%")
                    
                with col2:
                    st.success("### 🛡️ Eavesdropper (Quantized UAP Applied)")
                    st.write(f"**AI Hallucination:** {activity_map[data['obf_guess']]}")
                    st.write(f"**Confidence:** {data['obf_confidence']:.2f}%")
                
                # Subcarrier 50 was a null subcarrier (flat zero line)
                # Subcarrier 15 or 25 contains active dynamic bodily reflections
                orig_matrix = np.array(data["original_wave"])
                scram_matrix = np.array(data["scrambled_wave"])
                active_subcarrier = int(np.argmax(np.var(orig_matrix, axis=1)))
                orig_wave = orig_matrix[active_subcarrier, :]
                scram_wave = scram_matrix[active_subcarrier, :]

                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(orig_wave, label="Clean Physical Wave", color='#1f77b4', alpha=0.9)
                ax.plot(scram_wave, label="2-Bit UAP Wave", color='#d62728', alpha=0.9, linestyle='dashed')
                ax.set_title(f"Hardware Physical Layer Comparison (Dynamic Subcarrier {active_subcarrier})")
                ax.set_ylabel("Amplitude")
                ax.set_xlabel("Time (Packets)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
            else:
                st.error(f"Backend returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Backend connection failed: {e}")