# CreditSetu — Enterprise AI Lead Intelligence Engine for Retail Lending

[![IDBI Bank Theme](https://img.shields.io/badge/Branding-IDBI_Bank_Light-00543B.svg?style=for-the-badge&logoColor=white)](https://www.idbi.bank)
[![Hackathon Track](https://img.shields.io/badge/IDBI_Innovate_2026-Track_02-F37021.svg?style=for-the-badge)](https://www.idbi.bank)
[![API Status](https://img.shields.io/badge/API-Connected-emerald.svg?style=for-the-badge)](#api-documentation)
[![Database](https://img.shields.io/badge/Database-SQLite_Seeded-blue.svg?style=for-the-badge)](#database-schema)
[![Platform Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](#tech-stack)
[![Frontend Stack](https://img.shields.io/badge/Frontend-React_18%20%2B%20Vite-138B7B.svg?style=for-the-badge&logo=react&logoColor=white)](#tech-stack)

### 🚀 Live Production Deployment Links
*   **Live Dashboard Web App:** [https://credit-setu-iota.vercel.app](https://credit-setu-iota.vercel.app)
*   **Live API Documentation (Swagger Docs):** [https://creditsetu.onrender.com/docs](https://creditsetu.onrender.com/docs)
*   **Live API Base URL:** [https://creditsetu.onrender.com](https://creditsetu.onrender.com)

---

CreditSetu is an enterprise-grade AI lead intelligence engine designed to identify, rank, and explain high-quality retail lending prospects from customer transaction data. 

Specifically targeting **Thin-File** and **New-To-Credit (NTC)** segments that traditional bureau-based scoring models fail to evaluate, CreditSetu leverages Account Aggregator-style behavioral streams to assess creditworthiness, detect real-time life-event triggers, estimate repayment capacity, and apply risk-mitigating guardrails.

> [!NOTE]
> **Production Scale:** The live production environment is pre-seeded with **200 customers** (generating 152,545 transactions) to operate within Render's Free Tier memory ceiling (512MB RAM). However, the engine's data pipeline is fully scalable and was benchmarked locally on **5,000 customers** (3.87 Million transactions).

---

## 📖 Table of Contents
1. [Core Features & Innovation](#-core-features--innovation)
2. [Detailed System Architecture](#%EF%B8%8F-detailed-system-architecture)
3. [Scoring Engines Deep Dive](#-scoring-engines-deep-dive)
4. [Algorithmic & Mathematical Formulations](#-algorithmic--mathematical-formulations)
5. [Performance & Code Optimizations](#-performance--code-optimizations)
6. [Database Schema Reference](#-database-schema-reference)
7. [API Specification & Payloads](#-api-specification--payloads)
8. [Model Benchmarking & Evaluation](#-model-benchmarking--evaluation)
9. [Quick Start & Installation](#-quick-start--installation)
10. [Project Directory Tree](#-project-directory-tree)
11. [Developer Guidelines](#-developer-guidelines)
12. [Disclaimer & Credits](#-disclaimer--credits)

---

## 🌟 Core Features & Innovation

*   **Behavioral Credit Assessment:** Evaluates customers using transactional activity (UPI spending entropy, rent consistency, NACH bounces, and income stability) instead of relying solely on bureau histories.
*   **Intent Signal Detection (ruptures PELT):** Uses change-point detection algorithms to recognize life events like **EMI closures** (capacity expansion) or **income step-ups** directly from raw transaction flows.
*   **Capacity Regression Modeling:** Estimates safe monthly repayment capacity using LightGBM regressors that natively handle missing values (perfect for thin files).
*   **Risk Guardrail Engine:** Combines hard compliance checks (NACH bounces, excessive active lenders) with a logit-based classifier to assign customers to **Safe**, **Watch**, or **Suppress** tiers.
*   **Explainable AI (SHAP attributions):** Translates complex LightGBM decision trees into natural language explanations using normalized relative percentage feature attributions.
*   **IDBI Bank Light Theme UI:** Fully styled to follow IDBI Bank branding (Deep Green, Teal, and Orange), featuring responsive Recharts visualizations, direct-page pagination inputs, and stable loading overlays.

---

## 🗺️ Detailed System Architecture

```mermaid
graph TB
    subgraph "1. Data Layer"
        A[Synthetic Data Generator] --> B[Customer Profiles]
        A --> C[Daily Transaction Stream<br/>AA/UPI Structure]
        B --> D[(SQLite DB)]
        C --> D
    end

    subgraph "2. Feature Extraction & Engineering"
        D --> E[Batch Feature Pipeline]
        E --> F[Income Regularity CV]
        E --> G[Gig Payout Score]
        E --> H[EMI Burden Trends]
        E --> I[Rent & Spending Entropy]
        E --> J[30-day Rolling Cash Flow]
    end

    subgraph "3. Change-Point & Scoring Engines"
        J --> K[Intent Signal Engine<br/>ruptures PELT Detection]
        F & G & H & I --> L[Capacity Regression Engine<br/>LightGBM Regressor]
        H & I --> M[Risk Guardrail Engine<br/>Rules + Logit LGBM]
    end

    subgraph "4. Explainability & Attribution"
        L --> N[SHAP TreeExplainer]
        N --> O[Relative Impact Normalizer]
        O --> P[NLG Template Engine]
    end

    subgraph "5. Decisioning & API Delivery"
        K & L & M --> Q[Composite Scorer<br/>Lead Score Ingestion]
        P --> Q
        Q --> R{Guardrail<br/>Tier?}
        R -->|Safe / Watch| S[Qualified Lead Table]
        R -->|Suppress| T[Excluded Leads]
        S & T --> U[FastAPI Endpoints]
        U --> V[React Lead Dashboard]
    end
```

---

## ⚙️ Scoring Engines Deep Dive

CreditSetu evaluates each customer profile across three independent behavioral scoring dimensions, which are then combined to form a final composite lead score.

### 1. Intent Signal Detection Engine
Traditional lead generation models are static, relying on pre-determined campaign lists. The **Intent Engine** monitors transaction streams in real-time to detect major financial changes (life-event triggers). Using the `ruptures` PELT change-point algorithm, it detects shifts in rolling daily cash flow. 

If a customer's expenses drop significantly (e.g. they finish paying off an active loan), the engine flags an `emi_closure` event. If regular deposits step up, it flags an `income_step_up`. These triggers degrade exponentially over time, prioritizing recent signals to recommend the right loan products at the optimal time.

### 2. Behavioral Capacity Scoring Engine
Traditional credit scoring models reject thin-file or New-To-Credit (NTC) applicants because their CIBIL score is null. The **Capacity Engine** estimates repayment capacity by training a LightGBM Regressor on behavioral transaction features:
-   **Income Stability ($income\_cv$):** Coefficient of variation of monthly income credits. Higher volatility lowers the capacity score.
-   **Income Regularity ($income\_timing\_cv$):** Measures the consistency of salary deposit dates.
-   **Rent Consistency:** Evaluates the regularity of rent outflows.
-   **Spending Diversity ($entropy$):** Computes Shannon entropy on merchant categories. Highly concentrated spending (e.g. heavy cash withdrawals or gaming) lowers scoring.

### 3. Risk Guardrail Engine
The **Risk Guardrail Engine** protects the bank's asset quality. It combines hard rules with a machine learning classification model to filter out high-risk applicants:
-   **Hard Rules:** Instantly suppresses any customer who has $\ge 2$ NACH debit bounces in the last 3 months, or has $> 5$ concurrent active micro-lenders.
-   **ML Risk Classifier:** A LightGBM binary classifier predicts probability of repayment stress.
-   **Operational Tiers:** Assigns leads to **Safe** (passed to dashboard), **Watch** (highlighted with orange alerts for manual underwriting), or **Suppress** (fully excluded from sales workflows).

---

## 🔬 Algorithmic & Mathematical Formulations

### 1. Change-Point Detection (Ruptures PELT)
Daily net flow $y_t$ is defined as credits minus debits on day $t$. To smooth monthly salary and rent spikes, the signal is smoothed with a 30-day rolling average:
$$s_t = \frac{1}{30} \sum_{i=0}^{29} y_{t-i}$$

The Pruned Exact Linear Time (PELT) algorithm minimizes the penalized cost function over partitions of the smoothed signal:
$$\min_{k, \tau} \sum_{i=0}^{k} \mathcal{C}(s_{\tau_i \dots \tau_{i+1}}) + \beta k$$
Where $\mathcal{C}$ is the Radial Basis Function (RBF) cost model, $k$ is the number of change-points, and $\beta = 1.5$ is the penalty weight. Detected breakpoints are then classified as:
*   **EMI Closure:** If a regular monthly EMI debit pattern disappears after the breakpoint.
*   **Income Step-Up:** If the mean of salary/gig payouts increases by $\ge 12\%$ after the breakpoint.
*   **New Commitment:** If a new regular monthly EMI debit is established.

### 2. Repayment Capacity Regressor
 Repayment capacity is predicted using a LightGBM Regressor. Standard credit models fail when CIBIL score is null. LightGBM handles missing values natively by routing null values down the optimal decision path determined during training.

### 3. Risk Guardrail Tiers
To prevent the model from generating binary $0/1$ probabilities, we inject training noise into the target variable $y_{\text{stressed}} \in \{0, 1\}$. The classifier learns a continuous logit probability:
$$P(\text{Stress} \mid X) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + \epsilon)}}$$
Where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is Gaussian noise. This continuous risk score is mapped into three distinct operational tiers:
*   **Safe ($P < 0.35$):** Passed to the sales dashboard with standard processing.
*   **Watch ($0.35 \le P < 0.65$):** Passed but highlighted to credit analysts for closer review.
*   **Suppress ($P \ge 0.65$):** Excluded from the dashboard to protect the bank's asset quality.

### 4. Normalized SHAP Attributions
Standard SHAP values express feature influence in raw model output units (Rupees or Log-Odds). This squashes non-income features (like bounces) under massive salary numbers. 

We normalize SHAP values $\phi_i$ to express **relative percentage impact**:
$$\text{Relative Impact}_i = \frac{|\phi_i|}{\sum_{j} |\phi_j|} \times 100$$
This guarantees that behavioral signals (such as a single payment bounce or high active lender count) are visible to credit managers on the UI charts.

---

## ⚡ Performance & Code Optimizations

The pipeline is optimized to process **5,000 customers** with **3.87 Million transactions** in under **6 minutes**:

*   **Groupby Pre-indexing ($O(N)$ lookup):** Instead of scanning the entire 3.87M row transactions table inside the feature loop (which is $O(N \cdot M)$ and takes 15+ minutes), the pipeline groups the dataframe once by customer ID:
    ```python
    grouped_txns = {cust_id: group for cust_id, group in transactions_df.groupby("customer_id")}
    ```
    This reduced feature extraction time for 5,000 customers to **69.5 seconds**.
*   **Vectorized Net Flow:** Replaced row-wise `.apply()` lambda operations with vectorized `np.where` arrays inside the change-point detector:
    ```python
    txns_dated["signed_amount"] = np.where(txns_dated["type"] == "credit", txns_dated["amount"], -txns_dated["amount"])
    ```
*   **SQLAlchemy Bulk Insertion:** Changed individual row commits to database bulk inserts in batches of 50,000 records:
    ```python
    db.bulk_insert_mappings(Transaction, batch)
    ```
    This reduced database insert latency for 3.87M rows from ~1.5 hours to **79.6 seconds**. Used in both CLI seeding and API dataset regeneration endpoints.

---

## 🗄️ Database Schema Reference

CreditSetu uses a single-file SQLite database. The schema is defined via SQLAlchemy ORM models:

### 1. `customers` Table
Stores customer demographic data, persona types, and ground-truth validation targets.

| Column Name | Data Type | Key/Index | Ingest Equivalent | Description |
|---|---|---|---|---|
| `customer_id` | `VARCHAR(20)` | Primary Key | CRM Profile ID | Unique identifier for the customer (e.g. `CUST-03085`) |
| `name` | `VARCHAR(100)` | - | KYC Records | Full legal name of customer |
| `age` | `INTEGER` | - | KYC Records | Customer age |
| `gender` | `VARCHAR(1)` | - | KYC Records | Customer gender (M/F) |
| `occupation` | `VARCHAR(100)` | - | HR / Employer registry | Profession category |
| `persona_type` | `VARCHAR(50)` | - | Segment tagger | Behavioral persona (e.g. `salaried_stable`, `gig_worker`) |
| `bureau_score` | `FLOAT` | Nullable | CIBIL Direct API | Bureau score. Null indicates New-to-Credit (NTC) |
| `city` | `VARCHAR(50)` | - | KYC Records | Resident city location |
| `account_open_date`| `VARCHAR(20)` | - | CBS account registry| Date the bank account was opened |
| `monthly_income` | `FLOAT` | - | Income estimator | Inferred monthly baseline income |
| `emi_count` | `INTEGER` | - | Active Loans Registry | Active loans count running in ledger |
| `total_emi` | `FLOAT` | - | Active Loans Registry | Combined monthly EMI burden |
| `true_repayment_capacity` | `FLOAT` | - | Ground truth simulator | True repayment capacity (validation only) |
| `life_events` | `TEXT` | - | Ground truth simulator | JSON serialized list of validation events |
| `observation_months` | `INTEGER` | - | Ledger observation | Months of history available in the ledger |

### 2. `transactions` Table
Stores daily transactional ledger entries matching the Account Aggregator deposit schema.

| Column Name | Data Type | Key/Index | Ingest Equivalent | Description |
|---|---|---|---|---|
| `id` | `INTEGER` | Primary Key | System generated | Auto-increment ID |
| `txn_id` | `VARCHAR(30)` | Unique, Index | AA Provider ID | Unique transaction hash |
| `customer_id` | `VARCHAR(20)` | ForeignKey, Index | CBS ID | Link to `customers` table |
| `date` | `VARCHAR(20)` | - | AA Provider Timestamp | Transaction date |
| `amount` | `FLOAT` | - | AA Provider Value | Transaction value in INR |
| `type` | `VARCHAR(10)` | - | AA Inflow/Outflow indicator | `credit` or `debit` |
| `category` | `VARCHAR(50)` | - | Internal Classifier | Spent type (salary, emi, rent, grocery) |
| `counterparty` | `VARCHAR(100)` | - | UPI VPA / Payroll Name | Payee / Payor ID |
| `channel` | `VARCHAR(10)` | - | AA Provider metadata | UPI, NEFT, RTGS, NACH |
| `narration` | `VARCHAR(200)` | - | Ledger string | Raw bank ledger string |
| `is_bounce` | `BOOLEAN` | - | Return Code Handler | True if NACH returned due to insufficient funds |

### 3. `scores` Table
Stores evaluation scoring output, risk categories, suggested products, and SHAP features.

| Column Name | Data Type | Key/Index | Ingest Equivalent | Description |
|---|---|---|---|---|
| `customer_id` | `VARCHAR(20)` | Primary Key, Index | Scoring linkage | Link to `customers` table |
| `scored_at` | `VARCHAR(30)` | - | Pipeline clock | Generation timestamp |
| `intent_score` | `FLOAT` | - | Intent Engine | Raw intent trigger score [0, 1] |
| `intent_event_type` | `VARCHAR(50)` | - | Intent Engine | Type of life event detected |
| `intent_event_recency_days` | `INTEGER` | - | Intent Engine | Days since change-point occurred |
| `capacity_score` | `FLOAT` | - | Capacity Engine | Normalized repayment capacity [0, 1] |
| `capacity_amount` | `FLOAT` | - | Capacity Engine | Inferred safe monthly capacity amount in INR |
| `capacity_confidence` | `FLOAT` | - | Capacity Engine | Model confidence spread index |
| `guardrail_score` | `FLOAT` | - | Guardrail Engine | Continuous risk probability [0, 1] |
| `guardrail_tier` | `VARCHAR(20)` | - | Guardrail Engine | Risk category (Safe, Watch, Suppress) |
| `guardrail_reasons` | `TEXT` | - | Guardrail Engine | JSON list of risk factors |
| `composite_score` | `FLOAT` | - | Scorer Mixer | Weighted composite score [0, 1] |
| `is_qualified_lead` | `BOOLEAN` | - | Decision Matrix | True if passed for sales distribution |
| `suggested_product` | `VARCHAR(50)` | - | Decision Matrix | Target lending product recommended |
| `explanation` | `TEXT` | - | SHAP NLG Engine | Text explanation of score drivers |
| `shap_contributions` | `TEXT` | - | SHAP Explainer | JSON list of relative SHAP percentages |
| `top_features` | `TEXT` | - | SHAP Explainer | JSON list of top 3 impact features |

---

## 🔌 API Specification & Payloads

FastAPI exposes interactive Swagger docs at **`http://localhost:8000/docs`**.

### Key Endpoints

#### 1. `GET /api/leads`
Retrieves the ranked list of leads for the dashboard table. Supports filtering and pagination.
*   **Query Parameters:**
    *   `page`: Page index (default: `1`)
    *   `page_size`: Page limit (default: `20`)
    *   `min_score`: Filter by minimum composite score (default: `0.0`)
    *   `product_type`: Filter by product (e.g. `Micro-Credit Line`)
    *   `exclude_suppressed`: Hide over-leveraged leads (default: `true`)
*   **Sample Response:**
    ```json
    {
      "leads": [
        {
          "customer_id": "CUST-03085",
          "name": "Dhruv Reddy",
          "age": 34,
          "occupation": "Software Engineer",
          "persona_type": "salaried_stable",
          "city": "Mumbai",
          "bureau_score": 712.0,
          "composite_score": 0.88,
          "intent_score": 0.90,
          "capacity_score": 0.85,
          "guardrail_tier": "Safe",
          "is_qualified_lead": true,
          "suggested_product": "Home Loan",
          "explanation": "Score driven by high income regularity and stable monthly surplus."
        }
      ],
      "total": 2745,
      "page": 1,
      "page_size": 20
    }
    ```

#### 2. `GET /api/score/{customer_id}`
Retrieves the score breakdown and normalized SHAP attributions for a single customer.
*   **Sample Response:**
    ```json
    {
      "customer_id": "CUST-02144",
      "composite_score": 0.88,
      "intent_score": 0.73,
      "intent_event_type": "emi_closure",
      "intent_event_recency_days": 14,
      "capacity_score": 0.99,
      "capacity_amount": 49717.0,
      "guardrail_score": 0.12,
      "guardrail_tier": "Safe",
      "suggested_product": "Home Loan",
      "explanation": "Score driven by Monthly Income (+67%), Active Lenders (+13%), and Income Stability (+7%). Recent EMI closure detected 14 days ago.",
      "shap_contributions": [
        { "feature": "income_mean", "display_name": "Monthly Income", "value": 154769.0, "contribution": 67.71 },
        { "feature": "concurrent_lender_count", "display_name": "Active Lenders", "value": 1.0, "contribution": 13.46 },
        { "feature": "income_cv", "display_name": "Income Stability", "value": 0.035, "contribution": 7.49 },
        { "feature": "bureau_score", "display_name": "Bureau Score", "value": null, "contribution": -0.71 }
      ]
    }
    ```

#### 3. `GET /api/customers/db-stats/summary`
Retrieves aggregated statistics of the seeded database for the frontend Data Engine view.
*   **Sample Response:**
    ```json
    {
      "total_customers": 5000,
      "total_transactions": 3877572,
      "total_scores": 5000,
      "persona_distribution": {
        "salaried_stable": 1750,
        "gig_worker": 1000,
        "new_to_credit": 750,
        "over_leveraged": 750,
        "self_employed": 750
      },
      "category_distribution": {
        "groceries": 1010421,
        "shopping": 810117,
        "food_delivery": 606807,
        "salary": 29260,
        "emi": 84256,
        "rent": 14194,
        "nach_bounce": 2602
      },
      "total_bounces": 2602,
      "ntc_count": 2075,
      "has_bureau_count": 2925,
      "avg_transactions_per_customer": 775.5
    }
    ```

---

## 📊 Model Benchmarking & Evaluation

To ensure unbiased evaluation, the benchmark runner evaluates models on a **fresh test dataset** (1,000 customers, ~770,000 transactions) that is separate from the operational database.

To execute the evaluations:
```bash
cd backend
python scripts/run_benchmark.py --n_customers 1000
```

### Evaluation Metrics Summary

*   **Capacity Scoring Model:**
    *   **AUC-ROC:** `0.9918` (evaluating capacity threshold at median)
    *   **RMSE:** `₹2,967` (Root Mean Squared Error of repayment capacity prediction)
    *   **$R^2$ Variance:** `0.9279` (Coeff. of determination indicating variance explained)
*   **Risk Guardrail Model:**
    *   **AUC-ROC:** `0.7962` (discrimination between stressed and safe customers)
    *   **False Positive Rate (FPR):** `11.51%` (safe customers incorrectly suppressed)
    *   **False Negative Rate (FNR):** `34.43%` (stressed customers incorrectly passed)
*   **Intent Signal Model:**
    *   **Precision:** `0.3394` (accuracy of change-point classifications)
    *   **Recall:** `0.4526` (share of actual events captured by PELT)
*   **Composite Scoring Pipeline:**
    *   **Precision @ Top 20%:** `0.995` (percentage of top-ranked leads that are valid)
    *   **Avg scoring latency:** `21.19 ms` per customer record.

---

## 🚀 Quick Start & Installation

### Option 1: Docker Compose (Recommended)
This runs the entire stack in isolated Docker containers:

```bash
# Clone the repository
git clone https://github.com/adarshcod30/CreditSetu.git
cd CreditSetu

# Spin up services
docker-compose up --build
```
This single command:
1. Builds the backend and frontend containers.
2. Seeds SQLite with 5,000 customers.
3. Trains the LightGBM models.
4. Serves the API at **`http://localhost:8000`** and the Dashboard at **`http://localhost:5173`**.

### Option 2: Local Manual Setup

#### 1. Backend Setup
```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run seeding script (generates data, trains models, scores database)
python scripts/seed_database.py --n_customers 5000

# Start backend server
uvicorn app.main:app --port 8000 --reload
```

#### 2. Frontend Setup
```bash
cd frontend

# Install node dependencies
npm install

# Start Vite dev server
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## 📂 Project Directory Tree

```
CreditSetu/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── seed_database.py         # DB seeding script
│   │   └── run_benchmark.py         # Benchmarks evaluator
│   ├── app/
│   │   ├── main.py                  # FastAPI server entry point
│   │   ├── database.py              # SQLite connection config
│   │   ├── config.py                # Environment configuration settings
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   ├── schemas/                 # Pydantic v2 schemas
│   │   ├── data_generation/         # Synthetic customer/transaction generators
│   │   ├── features/                # Feature engineering and change-point detectors
│   │   ├── engines/                 # Scorer logic (Intent, Capacity, Guardrail)
│   │   ├── explainability/          # SHAP attribution calculations
│   │   └── api/                     # FastAPI route files
│   └── tests/                       # Pytest unit tests
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── tailwind.config.js
    ├── index.html
    └── src/
        ├── App.jsx                  # Main app navigation & structure
        ├── index.css                # Global design system variables
        ├── pages/
        │   ├── LeadDashboard.jsx    # Leads page with Pie/Bar aggregate charts
        │   ├── CustomerDetail.jsx   # Profile page with SHAP and Timeline charts
        │   ├── BenchmarkView.jsx    # Evaluation page with ROC and metrics
        │   └── DataArchitectureView.jsx# Database explorer & metadata page
        ├── components/
        │   ├── LeadTable.jsx        # Lead list table with stable overlay loading
        │   ├── FilterPanel.jsx      # Multi-metric select dropdown panels
        │   ├── ScoreBreakdownCard.jsx# Gauge indicators for scoring components
        │   └── TransactionTimeline.jsx# Net cash flow timeline chart
        └── api/
            └── client.js            # Axios client
```

---

## 🛠️ Developer Guidelines

### Adding a New Persona Archetype
To introduce a new customer segment (e.g. agricultural traders):
1. Open `backend/app/data_generation/personas.py`.
2. Append a new archetype dictionary to the `PERSONAS` dictionary defining the income range, base CIBIL score, transaction bounce probabilities, and target loan product:
   ```python
   "agri_trader": {
       "income_range": (30000, 90000),
       "bureau_range": (600, 750),
       "bounce_probability": 0.08,
       "selected_lenders": ["Agri Credit Corp", "NABARD Partner"],
       ...
   }
   ```
3. Re-run `python scripts/seed_database.py --n_customers 5000` to retrain the models and reseed.

### Customizing Risk Guardrail Rules
To add a new risk rule (e.g. suppressing leads who had more than 2 NACH bounces in 3 months):
1. Open `backend/app/engines/guardrail_engine.py`.
2. Add the evaluation check to the rule list inside `score()`:
   ```python
   if features.get("nach_bounce_count_3m", 0) > 2:
       reasons.append("Multiple recent NACH bounces (3m)")
       tier = "Suppress"
   ```

---

## ⚖️ Disclaimer & Credits

*   **Disclaimer:** This is a prototype built for demonstration purposes. Customer names, transaction ledgers, UPI VPAs, and narration details are synthetic.
*   **Hackathon Track:** Built for the **IDBI Innovate 2026 Hackathon (Track 02 — Lead Generation, Behavioural Analytics, Retail Lending)**.

*Developed pair-programmed with Google DeepMind Antigravity IDE.*
