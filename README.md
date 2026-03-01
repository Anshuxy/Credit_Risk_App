<!-- 💳 Project 10: Intelligent Credit Risk Scoring & Agentic Lending Decision Support -->

From Financial Risk Modeling to Automated Lending Advice

<!-- 📌 Project Overview -->
This project involves the design and implementation of an AI-driven credit analytics system that evaluates borrower credit risk and evolves into an agentic AI lending decision support assistant.

Milestone 1 (Current): Classical machine learning techniques applied to historical borrower data to predict creditworthiness, default probability, and loan risk.

Milestone 2 (Future): Extension into an agent-based AI application that autonomously reasons about borrower profiles and retrieves regulatory guidelines using RAG.


<!-- 🔗 Live Demo -->
Hosted Link: [https://creditriskapp-midsem.streamlit.app/]


<!-- 📺 Video Link -->
Watch the Project Demonstration Video: [. ]


<!-- 🛠️ Technology Stack (Milestone 1) -->

Component                 Technology
ML Models                 Logistic Regression, Random Forest, Scikit-Learn 
UI Framework              Streamlit 
Hosting                   Streamlit Community Cloud 
Environment               Python 3.13


<!-- 🛠️ Technical Integrity & "No GenAI" Affirmation -->
As per the evaluation criteria, I formally affirm that the core logic, model training, and decision-making pipelines of this project are my own work and are not direct outputs of Generative AI.

3 Distinct Technical Sub-Features =>

1. Ensemble Scoring Engine: A dual-model approach using Logistic Regression and Random Forest to provide both a statistical baseline and high-accuracy ensemble predictions.

2. Robust Data Pipeline: Custom-built preprocessing including feature scaling (StandardScaler), handling of class imbalance, and categorical encoding.

3. Real-time Risk Dashboard: A sophisticated Streamlit interface featuring "Glass Cards," real-time inference, and visual risk-driver analysis.


<!-- 📂 Repository Structure --> 

CREDIT_RISK_APP/
├── .gitignore               # Prevents tracking of junk files and environments
├── app.py                   # Main Streamlit application and UI logic
├── credit_model.pkl         # Serialized primary model for risk prediction
├── credit_risk_dataset (3).csv # Raw labeled dataset for training/testing
├── logistic_model.pkl       # Saved Logistic Regression baseline model
├── README.md                # Project documentation and setup guide
├── requirements.txt         # List of Python dependencies for deployment
├── rf_model.pkl             # Saved Random Forest challenger model
├── risk_drivers.png         # Visual asset for EDA/Dashboard reporting
└── train_model.py           # Core script for preprocessing and ML training


<!-- 📊 Model Performance Summary -->

Based on the Technical Implementation criteria, we include a baseline comparison between our two models:

Model                       Accuracy          ROC-AUC Score            Primary Strength
Logistic Regression         0.85 approx       0.87 approx              High Explainability & Fast Inference 
Random Forest               ~0.92+            ~0.94+                   Superior Accuracy & Non-linear capture 


<!-- 👥 Team Contributions (Milestone 1) -->

According to our specialized work distribution:

Himani Pinjani (Member 1): The Data Engineer
Responsibility: Find the dataset and write the Python scripts to clean it.

Ankita Thakur (Member 2): The ML Engineer
Responsibility: Build and train the actual prediction models.

Anshu Yadav (Member 3): UI Developer and Deployment Lead
Responsibility: Build the interface and Ensure the project is submitted correctly and runs online.

Farhana Pervin (Member4): Documentation Lead
Responsibility: Created the Model Performance Evaluation Report.
