# 💳 Credit Score AI

A machine learning-powered application to assess creditworthiness and predict default probability. Built with **Streamlit** and **XGBoost**.

## 🚀 Features

*   **Real-Time Prediction**: Instantly calculate credit scores based on 20+ financial and personal attributes.
*   **Smart Document Auto-Fill**: Upload a loan application (PDF/DOCX/TXT) and the AI automatically extracts and fills the form.
*   **Advanced AI Model**: Powered by **XGBoost** with class balancing to accurately detect high-risk applicants (approx. 75%+ accuracy).
*   **Dynamic Insights**: Determining feature importance in real-time based on the specific trained model, not static assumptions.
*   **Explainable UI**: Interactive charts visualizations to understand *why* a score was given.
*   **User-Friendly Interface**: Clean, dark-mode design with intuitive form inputs.

## 🛠️ Tech Stack

*   **Python 3.9+**
*   **Frontend**: Streamlit
*   **Model**: XGBoost, Scikit-Learn
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/credit-score-app.git
    cd credit-score-app
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 How to Run

1.  Make sure your data file `german_credit_data.csv` is in the `data/` folder or the project root.
2.  Run the Streamlit app:
    ```bash
    streamlit run src/frontend.py
    ```
3.  Open your browser at `http://localhost:8501`.

## 🧠 Model Logic

The application uses an **XGBoost Classifier** that has been fine-tuned for the German Credit Dataset. 
*   **Class Balancing**: The model automatically adjusts weights to pay more attention to "bad" credit risks, which are often underrepresented in data.
*   **Preprocessing**: Categorical variables (Sex, Housing, Purpose) are one-hot encoded, and numerical values are scaled using `StandardScaler`.

## 📊 Data Insights

A comprehensive analysis of the dataset and feature importance is available in:
*   [**`DATA_REPORT.md`**](DATA_REPORT.md): Breaks down every attribute (Checking Status, Credit History, etc.) and its specific impact on the risk score based on the trained model.

## 🧪 Testing Auto-Fill

The application supports **Document Auto-Fill**. To test this feature:

1.  Use the provided sample file: **`dummy_application.pdf`**.
2.  Upload it to the "Auto-Fill" sidebar in the app.
3.  The form fields (Age, Job, Amount, etc.) will automatically populate.

**Generating a New Test PDF:**
If you want to create a fresh test file, run:
```bash
python src/gen_pdf.py
```
This will generate/overwrite `dummy_application.pdf` with sample data.
