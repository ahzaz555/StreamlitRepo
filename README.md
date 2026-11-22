# Hotel Booking Cancellation Analysis

This project explores hotel booking cancellations using the Hotel Booking Demand dataset. The main goal is to understand cancellation patterns and evaluate how well machine learning models can predict whether a reservation will be canceled.

I used Logistic Regression and Random Forest for prediction. I also developed a Streamlit dashboard to explore booking data and added a LangChain-based AI assistant using Google's Gemini model.

## Project Files

- `DataScienceProject.ipynb` - Data exploration, preprocessing, model training, and evaluation.
- `app.py` - Streamlit dashboard with interactive filters and AI integration.
- `requirements.txt` - Python packages required to run the project.

## Dataset

The project uses the Hotel Booking Demand dataset, which contains 119,390 records and 32 columns.

Dataset source:
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

Download `hotel_bookings.csv` and place it in the same directory as `app.py`.

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/ahzaz555/StreamlitRepo.git
```

2. Open the project directory:

```bash
cd StreamlitRepo
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Make sure `hotel_bookings.csv` is in the project directory.

5. Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard should open in your browser.

## AI Assistant

The dashboard includes a basic AI assistant implemented using LangChain and Gemini.

To use this feature, configure the `GOOGLE_API_KEY` environment variable with your own Gemini API key before starting Streamlit.

The dashboard's filters and charts can be used without an API key.

## Current Analysis

The notebook covers:

- Exploring the dataset and identifying missing values
- Examining duplicate records
- Preparing features for machine learning
- Splitting data into training, validation, and test sets
- Logistic Regression and Random Forest
- Comparing model performance using accuracy, precision, recall, F1-score, and ROC-AUC

## Limitations

The dataset contains bookings from only two hotels, so the results may not generalize to other properties.

The analysis also considers how duplicate records and class imbalance affect the results.

This project is being developed as part of my Data Science coursework.
