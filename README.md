# Water Pollution AI 🌊

An AI-based web tool that detects visible indicators of water pollution from stream images and recommends relevant reporting resources.

## Features
- Image classification using ResNet18
- Detects:
  - debris/trash
  - foam or sheen
  - murky/discolored water
- Provides real-world reporting links
- Simple web interface using Streamlit

## Run locally

```bash
cd water-pollution-ai
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
