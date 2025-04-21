# Stellar type classification machine learning Streamlit app

This project is a machine learning application for predicting stellar types (Galaxy, Star, Quasar) using Streamlit app. The model is trained on a real astronomical dataset and allows users to input stellar parameters to receive instant predictions.

---

## Project Structure

- `app.py` - Streamlit web application for interactive predictions.
- `train_model.py` - Script to preprocess data, train the model, and save the model and label encoder.
- `main.py` - (Currently unused placeholder for future extensions.)
- `star_classification.csv` - Main dataset (large CSV file with stellar features and classes).
- `star_classification.csv.zip` - Zipped version of the dataset.
- `stellar_model.pkl` - Trained RandomForest model (binary file).
- `label_encoder.pkl` - LabelEncoder instance for decoding predictions.
- `requirements.txt` - Python dependencies.

---

## Setup Instructions

1. **Clone the repository and navigate to the project directory:**
    ```bash
    git clone <repo_url>
    cd ml_stellar_classification
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Train the model:**
    If you want to retrain the model, run:
    ```bash
    python train_model.py
    ```
    This will generate `stellar_model.pkl` and `label_encoder.pkl`.

4. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## Usage

- Open the app in your browser (Streamlit will provide a local URL).
- Enter the required stellar parameters in the input fields.
- Click the **PREDICT** button to get the predicted stellar class (Galaxy, Star, Quasar).

---

## Notes

- The dataset is large; ensure you have sufficient memory.
- Model and encoder files are required to run the app. Retrain if missing.
- Python 3.7+ is recommended.

---

## Credits

- Built with [scikit-learn](https://scikit-learn.org/), [Streamlit](https://streamlit.io/), [pandas](https://pandas.pydata.org/), and [numpy](https://numpy.org/).
- Dataset: [SDSS Star Classification Dataset](https://www.kaggle.com/datasets/erikbruin/star-dataset)

