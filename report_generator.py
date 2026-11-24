
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


def create_pdf_report(df):

    # Dataset statistics
    cleaned_df = df.drop_duplicates()

    original_rows = len(df)
    cleaned_rows = len(cleaned_df)
    duplicates_removed = original_rows - cleaned_rows

    original_rate = df["is_canceled"].mean() * 100
    cleaned_rate = cleaned_df["is_canceled"].mean() * 100

    # Missing values
    missing_children = df["children"].isna().sum()
    missing_country = df["country"].isna().sum()
    missing_agent = df["agent"].isna().sum()
    missing_company = df["company"].isna().sum()

    # Create the PDF
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    content = []

    # Small helper functions to avoid repeating the same code
    def add_heading(title):
        content.append(Paragraph(title, styles["Heading2"]))

    def add_text(text):
        content.append(Paragraph(text, styles["BodyText"]))
        content.append(Spacer(1, 10))

    def add_table(rows, widths):
        table = Table(rows, colWidths=widths, repeatRows=1)

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))

        content.append(table)
        content.append(Spacer(1, 15))

    # Report title
    content.append(
        Paragraph(
            "Hotel Booking Cancellation Analysis",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 12))

    add_text(
        "Data Science Project Report"
    )

    # Introduction
    add_text(
        "This project looks at hotel booking cancellations and the "
        "factors that may be related to them. I used the Hotel Booking "
        "Demand dataset to compare cancellation patterns, run "
        "statistical tests, and build two machine learning models. "
        "I also created a Streamlit dashboard to display the results "
        "and added a question-answer feature using LangChain and Gemini."
    )

    content.append(Spacer(1, 10))

    # ---------------------------------------
    # 1. Dataset Overview
    # ---------------------------------------

    add_heading("1. Dataset Overview")

    add_text(
        "The dataset contains booking information from a city hotel "
        "and a resort hotel. It includes booking details such as lead "
        "time, room price, hotel type, and cancellation status [1]."
    )

    overview_data = [
        ["Metric", "Original Data", "After Deduplication"],
        ["Total Bookings", f"{original_rows:,}", f"{cleaned_rows:,}"],
        [
            "Cancellation Rate",
            f"{original_rate:.2f}%",
            f"{cleaned_rate:.2f}%"
        ]
    ]

    add_table(overview_data, [150, 150, 170])

    # ---------------------------------------
    # 2. Statistical Analysis
    # ---------------------------------------

    add_heading("2. Statistical Analysis")

    add_text(
        "A chi-square test was used to check whether hotel type and "
        "cancellation status were related. The null hypothesis was "
        "that hotel type and cancellation status were independent. "
        "The chi-square statistic was 447.66, with a p-value below "
        "0.001. Based on this result, the null hypothesis was rejected."
    )

    add_text(
        "After removing duplicate rows, the cancellation rate was "
        "30.04% for City Hotel and 23.48% for Resort Hotel. "
        "This suggests an association between hotel type and "
        "cancellation status, although the test does not establish "
        "that hotel type causes cancellations."
    )

    add_text(
        "Independent two-sample t-tests were also used to compare "
        "lead time and average daily rate (ADR) between canceled and "
        "non-canceled bookings. The null hypothesis for each test "
        "was that the two groups had equal means."
    )

    add_text(
        "Canceled bookings had a higher average lead time "
        "(105.72 days) than non-canceled bookings (70.10 days). "
        "They also had a higher average ADR (117.77 compared with "
        "102.00). Both tests gave p-values below 0.001, indicating "
        "statistically significant differences."
    )

    # ---------------------------------------
    # 3. Data Cleaning
    # ---------------------------------------

    add_heading("3. Data Cleaning and Preprocessing")

    add_text(
        f"The original dataset had {original_rows:,} rows. "
        f"I found {duplicates_removed:,} exact duplicate rows and "
        f"removed them, leaving {cleaned_rows:,} rows. "
        f"The cancellation rate changed from {original_rate:.2f}% "
        f"to {cleaned_rate:.2f}% after this step."
    )

    add_text(
        f"Missing values were found in children ({missing_children:,}), "
        f"country ({missing_country:,}), agent ({missing_agent:,}), "
        f"and company ({missing_company:,}). Numerical missing values "
        "were handled using median imputation, while missing "
        "categorical values were represented as Unknown. "
        "I also created has_agent and has_company indicators."
    )

    add_text(
        "Categorical predictors were one-hot encoded, and numerical "
        "predictors were scaled where needed. The preprocessing "
        "steps were fitted using training data to prevent "
        "information leakage into the test set."
    )

    add_text(
        "After cleaning, about 72.51% of bookings were not canceled "
        "and 27.49% were canceled. Because the classes were not "
        "equally represented, I compared standard and class-balanced "
        "versions of the machine learning models."
    )

    content.append(PageBreak())

    # ---------------------------------------
    # 4. Machine Learning Models
    # ---------------------------------------

    add_heading("4. Machine Learning Models and Evaluation")

    add_text(
        "I used Logistic Regression and Random Forest to predict "
        "whether a booking would be canceled. The target variable "
        "was is_canceled. Twelve predictors were selected, including "
        "hotel type, lead time, ADR, previous cancellations, "
        "deposit type, and special requests."
    )

    add_text(
        "Reservation status was excluded because it reveals "
        "information about the outcome. Including it could cause "
        "data leakage and make the models appear better than "
        "they would be for new bookings."
    )

    add_text(
        "The data was split into approximately 80% training and "
        "20% testing. A group-aware approach was used so identical "
        "selected feature patterns would not appear in both sets. "
        "An additional validation split from the training data "
        "was used to compare the models."
    )

    content.append(
        Paragraph(
            "Model Performance (Internal Validation)",
            styles["Heading3"]
        )
    )

    model_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        ["LR Standard", "0.7568", "0.6540", "0.2668", "0.3790", "0.7432"],
        ["LR Balanced", "0.6912", "0.4622", "0.6736", "0.5482", "0.7440"],
        ["RF Standard", "0.7675", "0.7930", "0.2223", "0.3472", "0.8034"],
        ["RF Balanced", "0.7484", "0.5368", "0.6956", "0.6060", "0.8047"]
    ]

    add_table(model_data, [110, 75, 80, 70, 55, 75])

    add_text(
        "The balanced Random Forest had the highest validation "
        "F1-score (0.6060) and recall (0.6956). The standard "
        "Random Forest had higher accuracy but missed more "
        "cancellations. I selected the balanced Random Forest "
        "because identifying cancellations was the main goal."
    )

    # ---------------------------------------
    # 5. Final Test Evaluation
    # ---------------------------------------

    add_heading("5. Final Held-Out Test Evaluation")

    add_text(
        "The selected balanced Random Forest was retrained on all "
        "70,020 training records. It was then evaluated on the "
        "17,376 test records that had not been used for model "
        "selection or training."
    )

    test_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        ["RF Balanced", "0.7445", "0.5284", "0.7042", "0.6038", "0.8065"]
    ]

    add_table(test_data, [110, 75, 80, 70, 55, 75])

    content.append(
        Paragraph(
            "Final Test Confusion Matrix",
            styles["Heading3"]
        )
    )

    confusion_data = [
        ["Actual / Predicted", "Not Canceled (0)", "Canceled (1)"],
        ["Not Canceled (0)", "9,553", "3,019"],
        ["Canceled (1)", "1,421", "3,383"]
    ]

    add_table(confusion_data, [150, 155, 155])

    add_text(
        "The model correctly identified 3,383 canceled bookings "
        "and missed 1,421 cancellations. It also incorrectly flagged "
        "3,019 bookings that were not canceled. Its test recall "
        "was 0.7042, so it detected about 70% of actual cancellations."
    )

    add_text(
        "The final test F1-score was 0.6038, close to the validation "
        "F1-score of 0.6060. This was encouraging because the "
        "performance remained similar on the held-out data. "
        "The test accuracy was 0.7445, compared with a "
        "majority-class baseline of 0.7235."
    )

    content.append(PageBreak())

    # ---------------------------------------
    # 6. Streamlit and LangChain
    # ---------------------------------------

    add_heading("6. Streamlit and LangChain Implementation")

    add_text(
        "I built a Streamlit dashboard to make the analysis easier "
        "to explore. The application reads the hotel booking CSV "
        "file and displays the number of records, a small sample "
        "of the data, and the cancellation rate."
    )

    add_text(
        "The sidebar has two filters: hotel type and maximum lead "
        "time. Changing these filters updates the records being "
        "displayed and the cancellation-rate bar chart. "
        "The PDF download uses the complete dataset rather than "
        "the filtered records."
    )

    add_text(
        "I also added a question-answer feature using LangChain. "
        "The application takes a question from the user, places it "
        "into a prompt, sends the prompt to Gemini, and displays "
        "the returned text. PromptTemplate is used to prepare the "
        "question, ChatGoogleGenerativeAI calls the model, and "
        "StrOutputParser converts the response into text."
    )

    add_text(
        "The Google API key is read from an environment variable "
        "instead of being written directly in the source code. "
        "The current prompt provides Gemini with a short, fixed "
        "summary of the cleaned dataset. It does not pass the "
        "current filters or individual booking records to Gemini, "
        "so the AI response should be treated as a general "
        "explanation rather than a calculation from the chart."
    )

    # ---------------------------------------
    # 7. Discussion and Conclusion
    # ---------------------------------------

    add_heading("7. Discussion and Conclusion")

    add_text(
        "The analysis found that cancellation patterns differed "
        "between the two hotel types. Canceled bookings also "
        "had longer average lead times and higher average daily "
        "rates. These findings help describe the data, but they "
        "do not prove what caused a customer to cancel."
    )

    add_text(
        "The balanced Random Forest performed better at identifying "
        "cancellations than the standard models, based on recall "
        "and F1-score. However, it also produced false alarms. "
        "A hotel using this model would need to consider the "
        "cost of contacting customers who were not actually "
        "going to cancel."
    )

    add_text(
        "There are some limitations. The data comes from only two "
        "hotels, so the results may not be the same for other hotels. "
        "There is no unique booking ID, meaning some identical rows "
        "might represent separate bookings. ADR also contains "
        "unusual values that could affect some statistics."
    )

    add_text(
        "Overall, this project gave me practice with cleaning data, "
        "statistical testing, training classification models, and "
        "evaluating results. The Streamlit application also provides "
        "a way to explore the data and download the project report."
    )

   

    # ---------------------------------------
    # 9. GitHub Repository - Keep this last
    # ---------------------------------------

    add_heading("9. Source Code")

    content.append(
        Paragraph(
            '<link href="https://github.com/ahzaz555/StreamlitRepo">'
            'https://github.com/ahzaz555/StreamlitRepo'
            '</link>',
            styles["BodyText"]
        )
    )

    # Build the PDF
    doc.build(content)

    return buffer.getvalue()
