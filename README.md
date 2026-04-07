<div align="center">
<h1>🌍 POLKA (environmental)</h1>
  <h3>The Impact of Natural Factors on the Type of Political System</h3>
</div>

> **⚠️ Project Status: Conceptual & Active Development** > *Please note that the methodology, theoretical frameworks, and machine learning architectures outlined in this repository represent an initial **concept**. This project is in active development, and the classification logic, datasets, and codebase will be frequently modified, refined, and iterated upon as the research progresses.*

**Project POLKA** is an advanced Machine Learning and Data Science initiative dedicated to analyzing the impact of natural determinism on the formation of political systems. By evaluating environmental, geographical, and natural factors, this project aims to uncover correlations and build predictive models that forecast political, sub-political, and economic system types.

---

## 🎯 Project Overview

Historically, geography and environment have heavily influenced human organization. Project POLKA takes structured environmental data (e.g., climate types, land boundaries, soil composition, mountain ratios) and applies modern machine learning techniques to answer a core question: *Can we predict a region's political and economic system based purely on its natural factors?*

We are employing a **Hybrid Agentic Workflow**, utilizing robust gradient-boosting models for tabular predictions and LLM agents for data orchestration and explainability.

---

## 🚀 Roadmap and Phases

### Phase 1: Data Analysis and Modeling (Current)
- **Data Ingestion & Cleaning:** Processing complex datasets, handling international formatting discrepancies, and preparing structural data.
- **Exploratory Data Analysis (EDA):** Visualizing relationships between natural factors and political systems using `matplotlib`, `seaborn`, and `plotly`.
- **Predictive Modeling:** Training, evaluating, and tuning tree-based models (XGBoost, LightGBM, Random Forest) to predict categorical labels (`system_type`, `sub_system_type`, `economic_type`).

### Phase 2: Deployment and Agentic Interface (Future)
- **LLM Orchestration:** Utilizing LLM agents for automated data cleaning and interpreting complex feature importance (explainable AI).
- **Online Interface:** Building an interactive UI (via Streamlit, Gradio, or FastAPI) for users to input natural determinism factors and receive real-time predictions.
- **RAG Integration:** Implementing a vector database (e.g., PostgreSQL with pgvector) to store historical case studies and classification logic, enabling context-aware LLM explanations.

---

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost / LightGBM
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Future Deployment:** Streamlit / FastAPI, LangChain, PostgreSQL (pgvector)

---

## 📂 Repository Structure

```text
main
│
├── data/
│   └── environmental_data - Arkusz1.csv   # Twój zbiór danych
│
├── agents/
│   ├── __init__.py
│   └── data_analyst.py                    
│
├── .env                                   
├── main.py                                
└── requirements.txt                       
