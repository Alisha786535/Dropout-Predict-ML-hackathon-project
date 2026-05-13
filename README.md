# 🎓 Student Dropout Early Warning System – Deployment Guide

This guide explains how to **deploy the Student Dropout Early Warning System** built with **Streamlit**.

---
## User-Interface

## Prerequisites

Before deploying, ensure you have:

- Python 3.10+ installed
- Git installed
- Virtual environment tool (`venv`) or Conda (optional but recommended)
- Streamlit installed (`pip install streamlit`)

---

## Local Deployment

### 1. Clone the repository

```bash
  git clone https://github.com/Alisha786535/Dropout-Predict-ML-hackathon-project.git
  cd Dropout-Predict-ML-hackathon-project
```
### 2. Create a virtual environment
```bash
  python -m venv venv
  # Activate environment
  # Windows:
  venv\Scripts\activate
  # macOS/Linux:
  source venv/bin/activate
```
### 3. Install dependencies
```bash
  pip install -r requirements.txt
```
### 4. Run the Streamlit app
```bash
  streamlit run app.py
```
