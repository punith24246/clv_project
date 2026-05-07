# 📊 CLV & Churn Intelligence Dashboard

A comprehensive **Streamlit-based analytics dashboard** for Customer Lifetime Value (CLV) prediction and churn probability analysis. Built on the **UCI Online Retail II dataset** using probabilistic modeling (BG/NBD, Gamma-Gamma) and survival analysis.

---

## 🎯 Project Overview

This project analyzes customer value and retention patterns from the **UCI Online Retail II dataset**, combining advanced statistical modeling with interactive visualizations. It delivers actionable insights for customer segmentation, retention strategy, and revenue optimization across **3 customer segments**: Champions, Loyal, and Lost customers.

### Key Capabilities
- 🔮 **Predict customer lifetime value** for the next 12 months
- 📈 **Estimate churn probability** using Bayesian inference
- 👤 **Deep customer profiling** with RFM segmentation
- 📊 **Survival analysis** by customer segment
- 🧠 **Interactive Bayesian updater** to simulate churn probability changes
- 💰 **CLV Leaderboard** to identify high-value customers

### Dataset
**Source**: UCI Online Retail II Dataset  
**Customers**: 2,074 unique customers  
**Total CLV**: £5.9M+  
**Currency**: Pounds Sterling (£)

---

## 🚀 Features

### 1. **Overview Dashboard** 🏠
- Total customer count and estimated churn rate
- Average 12-month CLV across all customers
- Champion customer count
- Customer segment distribution (pie chart)
- Average spend by segment (bar chart)
- RFM scatter plot analysis
- Bayesian churn probability distribution
- Pareto analysis highlighting top revenue contributors

**Key Metrics**:
- Total Customers: 2,074
- Total Predicted CLV (12M): £5,924,656.75
- Avg CLV per Customer: £2,856.63
- Est. Churn Rate: ~1.7%
- Champion Customers: 275

### 2. **Customer Lookup** 🔍
- Search any customer by ID
- View complete RFM profile (Recency, Frequency, Monetary)
- CLV prediction (12-month CLV, probability of being active, predicted purchases)
- Churn risk assessment (HIGH/MEDIUM/LOW) with recommended actions
- Bayesian churn probability tracking over 12 months
- Posterior beta distribution visualization

### 3. **Survival Curves** 📈
- Kaplan-Meier survival curves by customer segment
- Churn rate comparison across segments
- Median survival times by segment
- Detailed churn statistics (count, percentage)

### 4. **CLV Leaderboard** 💰
- Top 20 customers by 12-month CLV
- Total predicted CLV and top 10% contribution metrics (54% of total value)
- CLV by segment analysis
- Total, average, and count metrics per segment
- CLV vs probability of being active scatter plot
- Visual segmentation with color-coded bars

### 5. **Bayesian Updater** 🧠
- Interactive monthly purchase history simulation
- Real-time Bayesian probability updates
- 95% credible interval visualization
- Prior strength and churn belief customization
- Posterior beta distribution display
- Confidence tracking through 12 months

---

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB RAM (minimum)
- Stable internet connection (for Streamlit Cloud deployment)

### Python Dependencies
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.3
lifelines==0.27.8          # Survival analysis
lifetimes==0.11.3          # BG/NBD & Gamma-Gamma models
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0             # Interactive visualizations
streamlit==1.28.0
openpyxl==3.1.2
xlrd==2.0.1
pickle5==0.0.12
```

---

## 🛠️ Installation & Setup

### Step 1: Clone or Download the Project
```bash
cd clv_project
```

### Step 2: Create a Virtual Environment
**On Windows:**
```bash
python -m venv clv_env
clv_env\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv clv_env
source clv_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r clv_requirements.txt
```

### Step 4: Prepare Data Files
Ensure these files are in the project directory:
- `final_clv_predictions.csv` - CLV predictions with customer data (main file)
- `rfm_with_churn.csv` - RFM scores and Bayesian churn probabilities
- `survival_data.csv` - Customer lifetime and churn status

### Step 5: Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your default browser.

---

## 📁 Project Structure

```bash
CLV-Churn-Intelligence/
│
│
├── assets/
│   ├── bayesian_churn_update.png
│   ├── cox_hazard_ratios.png
│   ├── eda_retail.png
│   ├── km_curves.png
│   ├── kmeans_selection.png
│   ├── rfm_scatter.html
│   └── ttest_confidence_interval.png
│
├── data/
│   ├── clv_predictions.csv
│   ├── final_clv_predictions.csv
│   ├── retail_clean.csv
│   ├── rfm_segments.csv
│   ├── rfm_with_churn.csv
│   └── survival_data.csv
│
├── models/
│   ├── bgf_model.pkl
│   ├── bgf_params.json
│   ├── cox_model.pkl
│   ├── ggf_model.pkl
│   ├── ggf_params.json
│   ├── rfm_kmeans.pkl
│   └── rfm_scaler.pkl
│
├── notebooks/
│   └── Clv_notebook.ipynb
│
├── scripts/
│   └──debug_data.py
│
├── .gitattributes
├── .gitignore
├── README.md
├── app.py
└── online_retail_II.csv
```
### 📌 Folder Description

- **assets/** → Generated visualizations, plots, and dashboard assets  
- **data/** → Processed datasets used by the Streamlit dashboard  
- **models/** → Trained ML/DL models and serialized objects  
- **notebooks/** → Jupyter notebooks for experimentation and analysis  
- **scripts/** → Helper scripts and preprocessing utilities  
- **app.py** → Main Streamlit dashboard application  
```

---

## 📊 Data Files Explained

### `final_clv_predictions.csv` ⭐ **MAIN FILE**
**Complete merged dataset with all customer information**
- `CustomerID`: Unique customer identifier
- `recency`: Days since last purchase
- `frequency`: Number of purchases
- `monetary`: Total spending (£)
- `rfm_score`: Combined RFM score (0-15)
- `segment`: Customer segment (Champions, Loyal, Lost)
- `bayesian_churn_prob`: Estimated churn probability
- `clv_12m`: Predicted 12-month customer lifetime value (£)
- `prob_alive`: Probability customer is still active
- `predicted_purchases_90d`: Expected purchases in next 90 days

### `rfm_with_churn.csv`
- RFM analysis results with Bayesian churn probabilities
- Used for Overview and Customer Lookup pages
- 2,074 customers with segment assignments

### `survival_data.csv`
- Customer lifetime duration and churn status
- Used for Survival Curves and median survival time calculations
- Contains lifetime_days and churned flag

### `online_retail_II.csv`
- **Source**: UCI Online Retail II Dataset
- Raw transaction data used for analysis
- Contains transaction details, dates, quantities, unit prices

---

## 🔬 Technical Background

### Models Used

#### 1. **BG/NBD (Beta-Geometric/Negative Binomial Distribution)**
- Predicts probability customer is still "alive"
- Estimates future purchase frequency
- Implemented via `lifetimes` library
- Handles non-contractual relationships

#### 2. **Gamma-Gamma Model**
- Estimates customer monetary value based on spending patterns
- Combines with BG/NBD predictions for CLV calculation
- Formula: `CLV = P(Alive) × Expected Frequency × Monetary Value`

#### 3. **Survival Analysis (Kaplan-Meier)**
- Estimates customer lifetime by segment
- Calculates median survival times
- Uses `lifelines.KaplanMeierFitter`
- Handles censoring for ongoing relationships

#### 4. **Bayesian Churn Inference**
- Beta-Binomial model for churn probability updates
- Updates belief as new purchase data arrives
- Uses Bayesian conjugate priors
- Visualizes posterior distribution

---

## 👥 Customer Segments

### 3 Segments Based on RFM Clustering

| Segment | Count | Avg CLV | Characteristics |
|---------|-------|---------|-----------------|
| **Champions** 🟣 | 275 | High | Best customers - high frequency & monetary value, low recency |
| **Loyal** 🔵 | 1,323 | Medium | Regular customers - good frequency and spend, recent activity |
| **Lost** 🔴 | 476 | Low | At-risk/inactive - high recency (long time since purchase) |

### Color Scheme
- **Champions** 🟣 Purple (#9B59B6)
- **Loyal** 🔵 Blue (#3498DB)
- **Lost** 🔴 Red (#E74C3C)

## 💡 How to Use Each Page

### 🏠 Overview
1. **Best for**: Executive summary and quick KPIs
2. **What you get**: High-level metrics, segment distribution, Pareto analysis
3. **Action**: Monitor overall churn rate and identify top revenue generators
4. **Key insight**: Top 10% of customers contribute 54% of total CLV

### 🔍 Customer Lookup
1. **Best for**: Individual customer investigation
2. **How to use**: 
   - Select a customer from dropdown
   - Review RFM profile and churn risk
   - Monitor CLV prediction and probability of staying active
3. **Action**: Identify at-risk customers for retention campaigns
4. **Example**: Find high-value customers at churn risk for targeted offers

### 📈 Survival Curves
1. **Best for**: Understanding segment behavior over time
2. **What you get**: Median survival days, churn rates by segment
3. **Action**: Tailor retention strategies by segment lifetime patterns
4. **Insight**: Compare survival trajectories across Champions, Loyal, Lost

### 💰 CLV Leaderboard
1. **Best for**: Customer prioritization and VIP management
2. **What you get**: Top 20 customers, CLV by segment, contribution analysis
3. **Action**: Focus resources on high-CLV customers, optimize marketing budget
4. **Data**: Total CLV £5.9M from 2,074 customers

### 🧠 Bayesian Updater
1. **Best for**: Understanding churn probability dynamics
2. **How to use**:
   - Toggle months where customer purchased
   - Adjust prior strength and belief
   - Observe probability convergence
3. **Action**: Learn how data updates change churn estimates
4. **Insight**: See confidence intervals tighten as more data arrives

---

## ⚙️ Configuration

### Streamlit Settings
The app uses `st.set_page_config()` to set:
- Page title: "CLV & Churn Intelligence"
- Layout: Wide (optimized for large screens)
- Sidebar: Expanded by default

### Caching
- `@st.cache_data`: Caches CSV loading for performance
- `@st.cache_resource`: Caches model loading (pickle files)

---

## 🐛 Troubleshooting

### Issue: "Data not found" Error
**Solution**: Ensure all CSV files are in the project directory
```bash
# Check file existence
ls final_clv_predictions.csv rfm_with_churn.csv survival_data.csv
```

### Issue: "CLV data not available for this customer"
**Solution**: Customer exists in RFM but not in CLV predictions. Verify data alignment:
```python
python debug_data.py
```

### Issue: Streamlit not starting
**Solution**: Verify installation and port availability
```bash
# Check Streamlit installation
pip list | grep streamlit

# Use different port if 8501 is busy
streamlit run app.py --server.port 8502
```

### Issue: Slow performance
**Solution**: Clear Streamlit cache
```bash
# Windows
rmdir %USERPROFILE%\.streamlit

# macOS/Linux
rm -rf ~/.streamlit
```

---

## 📈 Key Statistics

### Overall Metrics
- **Total Customers**: 2,074
- **Total Predicted CLV (12M)**: £5,924,656.75
- **Average CLV per Customer**: £2,856.63
- **Maximum Single Customer CLV**: £80,014.81
- **Estimated Churn Rate**: 1.7%
- **Top 10% Customer Contribution**: 54% of total CLV

### Segment Metrics
| Metric | Champions | Loyal | Lost |
|--------|-----------|-------|------|
| Count | 275 | 1,323 | 476 |
| Percentage | 13.3% | 63.8% | 22.9% |
| Avg CLV | High | Medium | Low |
| Churn Risk | Low | Low | High |

---

## 🚀 Deployment

### Deploy on Streamlit Cloud
1. Push code to GitHub repository
2. Visit https://share.streamlit.io
3. Connect GitHub repo and select `app.py`
4. Deploy with one click

### Deploy on Heroku
```bash
git push heroku main
heroku open
```

### Deploy on AWS
Use EC2 with Streamlit Server or containerize with Docker

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Lifelines Library](https://lifelines.readthedocs.io) - Survival Analysis
- [Lifetimes Library](https://github.com/CamDavidsonPilon/lifetimes) - BG/NBD Models
- [Plotly Documentation](https://plotly.com/python/) - Interactive Visualizations
- [UCI Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

---

## 🤝 Contributing

To improve this project:
1. Identify bugs or feature requests
2. Create detailed issue reports
3. Submit pull requests with improvements
4. Update documentation accordingly

---

## 📝 License

This project is provided as-is for educational and business intelligence purposes.

---

## 👨‍💼 Author Notes

**Built for**: Business analysts, data scientists, and marketing teams  
**Dataset**: UCI Online Retail II (e-commerce transactions)  
**Use case**: Customer retention strategy, marketing budget allocation, VIP management  
**Segments**: 3 main segments (Champions, Loyal, Lost)  
**Last updated**: May 2026

---

## 🆘 Support

For issues or questions:
1. Run `python debug_data.py` to diagnose data problems
2. Check Streamlit app terminal for error messages
3. Review this README for troubleshooting
4. Verify all dependencies are correctly installed

---

**Happy analyzing! 📊✨**
