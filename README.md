# Loan-Approval-Prediction
End-to-end ML pipeline for Loan Approval Prediction using Python, Pandas, and Scikit-Learn. Achieved 92.7% accuracy and 96.6% ROC-AUC using Logistic Regression.
# 🌸 FlowerAura Customer Sentiment Analysis: An End-to-End Data Science Project

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

> **Decoding the "pulse" of FlowerAura buyers through comprehensive customer review scraping, NLP, and predictive sentiment analysis.**

Hi everyone! 👋 Welcome to my milestone Data Analytics project. 

In the competitive world of e-commerce, customer feedback is the most honest source of truth. I conducted a comprehensive study of **FlowerAura** customer reviews to decode the "pulse" of their buyers. This project is a complete Data Science pipeline—from the challenges of scraping dynamic web content to cleaning messy, unstructured text and finally training a Machine Learning model to predict customer satisfaction.

---

## 🧐 The "Why": Project Objective
As a Data Analyst, my goal was to move beyond simple ratings. A 5-star rating tells you *that* a customer is happy, but the review text tells you *why*. By processing hundreds of customer reviews, I aimed to:
1. **Identify the drivers of customer delight** (what keeps them coming back).
2. **Pinpoint friction points** (the specific issues leading to negative feedback).
3. **Build a predictive engine** that can categorize sentiment for future reviews automatically.

---

## ⚙️ The Technical Toolkit
I utilized a robust, industry-standard stack of Python libraries to execute this pipeline:
* 🌐 **Web Scraping:** `Requests` for network protocols and `BeautifulSoup` for parsing complex DOM structures.
* 🧹 **Data Wrangling:** `Pandas` and `NumPy` for high-efficiency data cleaning and structural organization.
* 🧠 **Natural Language Processing (NLP):** 
    * `Regex` (Regular Expressions) for granular text cleaning.
    * `TextBlob` for calculating Polarity (emotional valence) and Subjectivity (opinionated vs. factual).
    * `NLTK` for tokenization and text normalization.
* 📊 **Visualization:** `Matplotlib` and `Seaborn` for statistical plotting, and `WordCloud` for qualitative theme extraction.
* 🤖 **Machine Learning:** `Scikit-Learn` for TF-IDF feature engineering and Logistic Regression classification.

---

## 🚀 The Pipeline: My Workflow

### 1. Web Scraping & Data Extraction
I automated the extraction of 50 pages of product reviews. The scraper didn't just grab raw text; it systematically captured metadata, including user names, city locations, purchase occasions, timestamps, and star ratings, resulting in a structured dataset ready for analysis.

### 2. Text Normalization & Preprocessing
Data from the web is inherently noisy. I implemented a cleaning pipeline that:
* Handled missing and duplicate values to ensure model integrity.
* Standardized column nomenclature and formatted datetime strings.
* Normalized review text by lowercasing, stripping punctuation, and removing high-frequency, low-meaning words (stop words) to focus the analysis on emotional keywords.

### 3. Sentiment Engineering
Using `TextBlob`, I generated sentiment scores for every review. By applying a mathematical threshold (Polarity >= 0.1 = Positive), I successfully quantified thousands of lines of subjective text into a binary classification model (Positive vs. Negative).

### 4. Qualitative & Quantitative EDA
I visualized the data to reveal hidden stories:
* **Sentiment Dynamics:** Comparing how specific star ratings correlate with emotional polarity.
* **Length Analysis:** Analyzing why negative reviews tend to be more concise while positive reviews often go into descriptive detail.
* **Thematic Extraction:** Using Word Clouds, I created a visual representation of the most frequently used terms. This highlighted the vocabulary of satisfaction versus the vocabulary of frustration.

### 5. Predictive Modeling
To prove the analytical potential of this data, I built a **Logistic Regression** model. I converted the text data into a numerical format using the **TF-IDF (Term Frequency-Inverse Document Frequency)** algorithm. The model was trained to recognize the "shape" of positive versus negative language. 

🏆 **The Result:** The model achieved an **impressive 90.2% accuracy** in sentiment classification on unseen test data!

---

## 💡 Business Insights & Recommendations
Data is only valuable if it leads to action. Based on my analysis:
* 🚚 **Delivery Reliability:** Negative sentiment is primarily driven by logistics. Implementing a real-time delivery tracking dashboard is the #1 recommendation to improve customer experience.
* 🎉 **Seasonal Marketing:** Peak order volume coincides with birthdays and anniversaries. Marketing efforts should be heavily front-loaded (7–10 days in advance) for these specific dates to capture the highest conversion segments.
* 🛡️ **Proactive Recovery:** The model's ability to predict negative sentiment in real-time allows the support team to intervene *before* a bad review is ever posted, turning a potential detractor into a loyal advocate.

---

## 📁 Repository Contents
* 📄 `FlowerAura_Final_Analysis.csv`: The cleaned, processed, and analyzed dataset.
* 📓 `floweraura_webscrapping.ipynb`: The complete Jupyter Notebook documenting every cell of my code and analysis.

*Thanks for checking out my work! Feel free to leave a star ⭐ or fork the project if you find it useful. 🚀*

---

## 👨‍💻 Author

**Mohit Kumar**
*   📧 **Email:** [kumarmohitamua@gmail.com](mailto:kumarmohitamua@gmail.com)
*   🐙 **GitHub:** [@MohitMintorian](https://github.com/MohitMintorian)
*   💼 **LinkedIn:** [Mohit Kumar](https://www.linkedin.com/in/) 

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" />
</p>
