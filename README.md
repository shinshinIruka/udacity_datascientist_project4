# Customer Segmentation and Response Prediction

This project applies **unsupervised learning** and **classification techniques** to analyze and predict customer responses for a mail-order company. By leveraging demographic data and machine learning models, the goal is to:
1. Segment customers based on shared attributes.
2. Predict if a potential customer responding to a marketing campaign.

---

## Motivation
Understanding customer segments and predicting response behavior is critical for optimizing marketing campaigns. By identifying key customer traits and predicting responders, the company can reduce marketing costs and improve campaign effectiveness.

---

## Libraries Used
The project uses the following Python libraries:
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For machine learning algorithms and evaluation metrics.
- **Matplotlib**: For visualization of results.
- **Seaborn**: For advanced data visualization.


Install the required libraries using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## Repository Structure
The repository contains the following files:

| File Name                       | Description                                                                                  |
|---------------------------------|----------------------------------------------------------------------------------------------|
| `data/Udacity_AZDIAS_052018.csv`     | Demographics data for the general population of Germany (input dataset).                    |
| `data/Udacity_CUSTOMERS_052018.csv`  | Demographics data for customers of the mail-order company (input dataset).                  |
| `data/Udacity_MAILOUT_052018_TRAIN.csv` | Training data for the marketing campaign response prediction (includes the `RESPONSE` column). |
| `data/Udacity_MAILOUT_052018_TEST.csv`  | Testing data for the marketing campaign response prediction (target labels withheld).       |
| `Arvato Project Workbook.ipynb`                       | Main jupyter notebook containing the workflow for EDA, data visualization, data procecssing, dimensionality reduction, clustering, and modeling. |
| `README.md`                     | Documentation about the project (this file).                                                |

---

## Project Workflow
### 0. **Data Exploring and Preprocessing**
- Perform an initial analysis and cleaning of the datasets to understand their structure and features.

### 1. **Unsupervised Learning for Customer Segmentation**
- **Dimensionality Reduction with PCA**: Reduced the dataset to key principal components that explain the variance in the data.
- **Clustering with KMeans**: Grouped customers and general population into segments to identify the characteristics of the core customer base.
- **Result**: Identified the demographic segments more likely to be customers of the mail-order company.

### 2. **Supervised Learning for Response Prediction**
- **Data Preprocessing**:
  - Addressed missing values and scaled features.
  - Handled dataset imbalance using techniques likeclass weighting...
- **Model Training and Evaluation**:
  - Used Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier for classification.
  - Evaluated models using **ROC-AUC** considering the imbalanced nature of the data.
- **Result**: Achieved a ROC-AUC of 0.752 with initial models. Improvements through tuning are ongoing.

---

## Summary of Results
1. **Customer Segmentation**:
   - PCA revealed that a subset of components captures the majority of variance in demographic features.
   - KMeans clustering identified key population segments aligned with the company's core customer base.

2. **Marketing Campaign Response Prediction**:
   - The most predictive models achieved:
     - **ROC-AUC**: 0.752 (indicating reasonable but improvable separation of responders and non-responders).
   - Further improvements (feature engineering, hyperparameter tuning) are needed for better predictions.

---

## Future Work
- Investigate advanced clustering techniques (e.g., DBSCAN or hierarchical clustering) for better segmentation.
- Fine-tune the Gradient Boosting model’s hyperparameters to further improve performance.
- Explore additional ensemble models like LightGBM or CatBoost for comparison.
- Investigate feature engineering techniques or external data sources to enhance predictive power.

---

## Github
- Link: https://github.com/shinshinIruka/udacity_datascientist_project4.git

## Blog
- Link: [Medium Blog](https://medium.com/@thinhelca/customer-segmentation-report-for-arvato-financial-services-d4e26a252191)


## Acknowledgment: 
- The data for this project is from the AZ Direct GmbH data, solely for use in this Bertelsmann Capstone project.
- I warrant that I will delete any and all data I downloaded within 2 weeks after my completion of the Bertelsmann Capstone project and the program.
