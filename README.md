<div align="center">
  <h1>🌍 POLKA (environmental)</h1>
  <h3>The Impact of Natural Factors on the Type of Political Systems and Civilizations</h3>
</div>

> **⚠️ Project Status: Active Development (Phase 1)**  
> *The methodology, theoretical frameworks, and machine learning pipelines outlined in this repository represent an evolving scientific framework exploring environmental determinism. The codebase, classification logic, and datasets are actively tested and refined as research progresses.*

**Project POLKA** is an advanced Machine Learning and Data Science initiative dedicated to analyzing how natural determinism shapes the formation of political systems, state structures, and economic frameworks[cite: 1]. By structuring complex historical and geographical factors—such as soil composition, geomorphological landscapes, climate tiers, and boundary configurations—this project builds predictive models capable of forecasting political and economic classification types.

---

## 🎯 Project Overview

Historically, geography and physical environment have heavily constrained human political organization. Project POLKA ingests structured environmental data (e.g., Chernozem/Black Earth ratios, mountain barriers, river configurations, and Köppen climate zones) and applies modern data science to answer a fundamental historical question: *Can we accurately predict a state's political structure and economic model purely from its geographical baseline?*

We employ a **Hybrid Architecture**:
* **Quantitative Machine Learning:** Utilizing robust gradient-boosting models (CatBoost) for multi-class tabular prediction of political and economic topologies (`system_type`, `sub_system_type`, `economic_type`)[cite: 1].
* **Local Agentic Workflow:** Integrating containerized Python pipelines with local Large Language Models (LLM via Ollama & LangChain) for intelligent exploratory data analysis (Smart EDA) and cross-tabulation mapping.

---

## 🚀 Roadmap and Phases

### Phase 1: Data Analysis, Modeling & Smart EDA (Current)
- **Data Ingestion & Cleaning:** Processing complex historical datasets (`environmental_data.csv`), handling international formatting discrepancies, and enforcing rigorous classification rules[cite: 1].
- **Exploratory Data Analysis & Heatmaps:** Visualizing relationships and conditional probabilities between natural topologies (e.g., `dominant_landscape`, `dominant_soil`) and political targets (`system_type`)[cite: 1].
- **Predictive Modeling & Balancing:** Training and tuning gradient-boosting algorithms (CatBoost) with class-weight balancing to accurately evaluate minority categories (e.g., Federal and Central systems).
- **Local LLM Integration (Smart EDA):** Deploying a containerized LangChain pandas dataframe agent running locally via Ollama (Llama 3) to execute natural language queries over the dataset safely and privately.

### Phase 2: Deployment and Agentic Interface (Future)
- **Explainable AI (XAI):** Translating complex feature importances and model decisions into natural language insights.
- **Interactive Web Interface:** Building a user-facing dashboard (via Streamlit or FastAPI) where users can input geographical and environmental parameters to receive real-time structural predictions.
- **Advanced Contextual RAG:** Implementing vector databases to store historical case studies and classification guidelines for context-aware historical reasoning.

---

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, CatBoost
* **Visualization:** Matplotlib, Seaborn
* **Agentic Framework & Local LLM:** LangChain, LangChain-Experimental, LangChain-Ollama, Ollama (Llama 3.1)
* **Containerization:** Docker (Isolated local execution ensuring 100% data privacy)

---

## 📁 Repository Structure

```text
POLKA_environmental/
│
├── data/
│   └── environmental_data.csv       # Primary geographical and historical dataset[cite: 1]
├── main.py                          # Core ML pipeline (CatBoost training, metrics, heatmaps)
├── app.py                           # Local Smart EDA Agent (LangChain + Ollama integration)
├── Dockerfile                       # Container definition for isolated Python execution
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
