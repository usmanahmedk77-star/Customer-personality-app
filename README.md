## 👨‍💻 Author

**Usman Ahmed Khan**

Machine Learning / Data Science Project

---

# Customer Personality Analysis & Segmentation App

A machine learning project that analyzes customer behavior, performs customer segmentation using **K-Means clustering**, and provides business-oriented recommendations through an interactive **Streamlit dashboard**.

The project takes a feature-engineered customer dataset, applies a reusable clustering pipeline, assigns customers to meaningful business segments, and visualizes customer characteristics such as income, spending, purchasing behavior, recency, and campaign engagement.

---

## 🚀 Project Overview

Customer segmentation helps businesses understand different groups of customers and design targeted marketing strategies.

This project uses **K-Means Clustering** to divide customers into four meaningful segments:

1. **Budget Shoppers**
2. **Emerging / Mid-Tier Customers**
3. **High-Value Customers**
4. **Premium & Campaign-Responsive Customers**

The final application allows users to:

* Explore overall customer statistics
* Analyze customer segments
* Visualize customer spending and income
* Compare purchasing channels
* View segment profiles
* Predict the segment of an existing customer
* Predict the segment of a new customer
* Generate business-oriented marketing recommendations

---

## 🧠 Machine Learning Approach

### Algorithm

The project uses:

**K-Means Clustering**

The final model uses **4 clusters (K=4)**.

Before clustering, selected numerical features are standardized using:

**StandardScaler**

The clustering model is saved as a reusable `.joblib` artifact so that predictions can be made without retraining the model.

---

## 📊 Features Used for Clustering

The model uses the following customer features:

* Customer Age
* Income
* Total Spending
* Recency
* Customer Tenure
* Family Size
* Total Campaign Acceptance
* Number of Web Purchases
* Number of Store Purchases
* Number of Catalog Purchases

Highly correlated/redundant features were removed during feature selection to improve the clustering pipeline.

---

## 👥 Customer Segments

### 1. Budget Shoppers

Lower-value and price-sensitive customers.

**Marketing focus:**

* Budget-friendly offers
* Starter bundles
* Cashback/points
* Deal-based promotions
* Increasing purchase frequency

### 2. Emerging / Mid-Tier Customers

Established customers with moderate value and potential for growth.

**Marketing focus:**

* Loyalty programs
* Milestone rewards
* Cross-selling
* Increasing average order value

### 3. High-Value Customers

Customers with strong spending and/or purchasing frequency.

**Marketing focus:**

* Customer retention
* Premium products
* Basket-size expansion
* Frequency rewards
* Personalized recommendations

### 4. Premium & Campaign-Responsive Customers

High-priority customers with strong value and campaign responsiveness.

**Marketing focus:**

* VIP loyalty
* Early access
* Premium products
* Exclusive campaigns
* Referral programs

---

## 📈 Dashboard

The Streamlit dashboard provides several analytical views.

### Customer Personality Dashboard

Displays key performance indicators such as:

* Total Customers
* Number of Segments
* Average Income
* Average Spending
* Average Recency

It also provides visualizations for:

* Customer distribution by segment
* Average spending by segment
* Average income by segment
* Purchasing behavior by channel

### Customer Segmentation

Users can select individual customer segments and analyze:

* Income vs. Total Spending
* Spending distribution
* Average customer characteristics
* Segment profiles

### Customer Prediction

The application supports two prediction modes:

**Existing Customer**

* Select a customer ID
* Run the saved ML model
* Receive the predicted customer segment

**New Customer**

* Enter engineered customer features
* Apply the saved scaler and K-Means model
* Predict the appropriate customer segment

---

## 🗂️ Project Structure

```text
customer-personality-app/
│
├── app.py
│
├── config/
│   └── recommendations.py
│
├── data/
│   ├── customer_personality_feature_engineered.csv
│   └── customer_segments.csv
│
├── models/
│   └── clustering_model.joblib
│
├── src/
│   └── clustering_pipeline.py
│
└── .vscode/
    └── settings.json
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Plotly**
* **Streamlit**
* **K-Means Clustering**
* **StandardScaler**

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-personality-app.git
```

### 2. Navigate to the project directory

```bash
cd customer-personality-app
```

### 3. Install the required libraries

```bash
pip install pandas numpy scikit-learn joblib plotly streamlit
```

---

## ▶️ Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL where the dashboard can be opened in a browser.

---

## 🔄 Machine Learning Pipeline

The overall workflow is:

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Feature Selection
       ↓
Standard Scaling
       ↓
K-Means Clustering
       ↓
Cluster Profiling
       ↓
Business Segment Naming
       ↓
Saved ML Model
       ↓
Streamlit Dashboard
       ↓
Customer Prediction & Recommendations
```

---

## 💾 Saved Model

The project includes a saved clustering model:

```text
models/clustering_model.joblib
```

The model contains the trained:

* StandardScaler
* K-Means model
* Cluster profiles
* Segment mappings

This allows the application to make predictions using the existing trained model instead of retraining every time the dashboard starts.

---

## 📁 Dataset

The project uses a customer personality dataset containing information related to:

* Demographics
* Income
* Family information
* Product spending
* Purchase channels
* Campaign responses
* Customer activity
* Recency
* Customer tenure

Feature engineering was performed to create additional customer behavior indicators such as:

* Customer Age
* Customer Tenure
* Total Children
* Family Size
* Total Spending
* Total Purchases
* Average Spending per Purchase
* Digital Engagement
* Deal Dependency
* Preferred Shopping Channel
* Product Preference
* Customer Activity Level

---

## 💡 Business Value

The project transforms raw customer data into actionable customer groups.

Businesses can use the resulting segments to:

* Personalize marketing campaigns
* Identify high-value customers
* Improve customer retention
* Create targeted discounts
* Recommend suitable products
* Increase purchase frequency
* Improve average order value
* Design loyalty programs
* Allocate marketing budgets more effectively

---

## 🔮 Future Improvements

Possible improvements include:

* Adding real-time database integration
* Deploying the Streamlit application online
* Adding automated model retraining
* Monitoring model performance
* Adding customer lifetime value prediction
* Implementing more clustering algorithms
* Adding advanced recommendation systems
* Integrating automated marketing campaign generation

---

