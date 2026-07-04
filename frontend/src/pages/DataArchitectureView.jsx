import React, { useState, useEffect } from 'react';
import { Database, Table, Layers, Users, TrendingUp, Info, ShieldAlert, Cpu } from 'lucide-react';
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';
import api from '../api/client';

const COLORS = ['#00543B', '#138B7B', '#F37021', '#E8452D', '#64748B'];

const TABLE_SCHEMAS = {
  customers: {
    title: 'Customers Table (Demographics & Baseline Profiles)',
    description: 'Stores base customer records, demographic data, calibrated occupational personas, and credit bureau scores (if available).',
    source: 'IDBI Core Banking System (CBS) & CRM Datasets',
    columns: [
      { name: 'customer_id', type: 'VARCHAR(20)', key: 'PRIMARY KEY', source: 'CBS Ingest', desc: 'Unique alphanumeric identifier for the customer.' },
      { name: 'name', type: 'VARCHAR(100)', key: '-', source: 'KYC Document', desc: 'Full legal name retrieved during onboarding.' },
      { name: 'age', type: 'INTEGER', key: '-', source: 'KYC Document', desc: 'Age in years.' },
      { name: 'gender', type: 'VARCHAR(1)', key: '-', source: 'KYC Document', desc: 'Gender (M/F).' },
      { name: 'occupation', type: 'VARCHAR(50)', key: '-', source: 'Employer Registry', desc: 'Profession category (e.g. Software Engineer, Delivery Partner).' },
      { name: 'persona_type', type: 'VARCHAR(30)', key: '-', source: 'Calibrated Archetype', desc: 'Behavioral segment (e.g. salaried_stable, gig_worker, ntc_no_bureau).' },
      { name: 'bureau_score', type: 'FLOAT', key: 'NULLABLE', source: 'CIBIL Bureau Pull', desc: 'Traditional bureau score. Null represents New-to-Credit (NTC).' },
      { name: 'monthly_income', type: 'FLOAT', key: '-', source: 'Income Estimator', desc: 'Self-reported monthly income baseline.' },
      { name: 'emi_count', type: 'INTEGER', key: '-', source: 'Active Loans Registry', desc: 'Count of active loan accounts currently running.' },
      { name: 'total_emi', type: 'FLOAT', key: '-', source: 'Active Loans Registry', desc: 'Aggregated monthly debt servicing obligations.' },
    ]
  },
  transactions: {
    title: 'Transactions Table (Account Aggregator Ledger)',
    description: 'Stores detailed day-to-day debit/credit ledger histories mimicking Sahamati-compliant Account Aggregator financial statements.',
    source: 'Account Aggregator (AA) Consent Flow Stream',
    columns: [
      { name: 'id', type: 'INTEGER', key: 'PRIMARY KEY', source: 'System Generated', desc: 'Auto-incrementing record ID.' },
      { name: 'txn_id', type: 'VARCHAR(30)', key: 'UNIQUE', source: 'AA Provider Payload', desc: 'Unique transaction reference hash.' },
      { name: 'customer_id', type: 'VARCHAR(20)', key: 'FOREIGN KEY', source: 'CBS Link', desc: 'Foreign key referencing the Customers table.' },
      { name: 'date', type: 'VARCHAR(20)', key: '-', source: 'AA Provider Payload', desc: 'ISO format date string of transaction.' },
      { name: 'amount', type: 'FLOAT', key: '-', source: 'AA Provider Payload', desc: 'Transaction value in INR.' },
      { name: 'type', type: 'VARCHAR(10)', key: '-', source: 'AA Provider Payload', desc: 'Credit (inflow) or Debit (outflow).' },
      { name: 'category', type: 'VARCHAR(50)', key: '-', source: 'Parser Classifier', desc: 'Spending/earning classification (e.g. Salary, Rent, EMI, Grocery, Utility).' },
      { name: 'counterparty', type: 'VARCHAR(100)', key: '-', source: 'AA Provider Payload', desc: 'UPI VPA address, bank account description, or corporate payroll name.' },
      { name: 'channel', type: 'VARCHAR(10)', key: '-', source: 'AA Provider Payload', desc: 'Payment channel (e.g. UPI, NEFT, NACH, RTGS).' },
      { name: 'narration', type: 'VARCHAR(200)', key: '-', source: 'AA Provider Payload', desc: 'Raw, unparsed bank ledger description string.' },
      { name: 'is_bounce', type: 'BOOLEAN', key: '-', source: 'Return Code Parser', desc: 'True if a NACH debit is returned/bounced due to insufficient balance.' },
    ]
  },
  scores: {
    title: 'Scores Table (AI Engine Model Output)',
    description: 'Stores final outputs of the scoring engines, including SHAP relative feature contributions, risk classifications, and suggested products.',
    source: 'CreditSetu Scoring Pipeline Output',
    columns: [
      { name: 'customer_id', type: 'VARCHAR(20)', key: 'PRIMARY KEY', source: 'System Generated', desc: 'Link to the evaluated customer.' },
      { name: 'intent_score', type: 'FLOAT', key: '-', source: 'Intent Engine', desc: 'Signal strength [0, 1] of positive credit events detected.' },
      { name: 'intent_event_type', type: 'VARCHAR(50)', key: 'NULLABLE', source: 'Intent Engine (PELT)', desc: 'Category of change-point detected (e.g. emi_closure, income_step_up).' },
      { name: 'intent_event_recency_days', type: 'INTEGER', key: 'NULLABLE', source: 'Intent Engine (PELT)', desc: 'Recency of change-point occurrence.' },
      { name: 'capacity_score', type: 'FLOAT', key: '-', source: 'Capacity Engine', desc: 'Normalized repayment index [0, 1] derived from LightGBM regressor.' },
      { name: 'capacity_amount', type: 'FLOAT', key: '-', source: 'Capacity Engine', desc: 'Predicted safe monthly surplus amount in INR.' },
      { name: 'guardrail_score', type: 'FLOAT', key: '-', source: 'Guardrail Engine', desc: 'Probability score [0, 1] from the risk classifier.' },
      { name: 'guardrail_tier', type: 'VARCHAR(20)', key: '-', source: 'Guardrail Engine', desc: 'Risk categorization (Safe, Watch, Suppress).' },
      { name: 'composite_score', type: 'FLOAT', key: '-', source: 'Composite Scorer', desc: 'Combined ranking index used for sales allocation.' },
      { name: 'suggested_product', type: 'VARCHAR(50)', key: '-', source: 'Decision Engine', desc: 'Target retail credit product recommended.' },
      { name: 'explanation', type: 'TEXT', key: '-', source: 'SHAP Explainer', desc: 'Natural language text explanation of score drivers.' },
      { name: 'shap_contributions', type: 'TEXT', key: '-', source: 'SHAP Explainer', desc: 'JSON list of normalized SHAP relative weights.' },
    ]
  }
};

export default function DataArchitectureView() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeSchema, setActiveSchema] = useState('customers');

  useEffect(() => {
    async function loadStats() {
      try {
        const res = await api.getDbStats();
        setStats(res.data);
      } catch (err) {
        console.error('Failed to load database stats', err);
      } finally {
        setLoading(false);
      }
    }
    loadStats();
  }, []);

  const formatPersonaName = (name) => {
    return name
      .replace(/_/g, ' ')
      .split(' ')
      .map(w => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ');
  };

  const personaChartData = stats
    ? Object.entries(stats.persona_distribution).map(([key, val]) => ({
        name: formatPersonaName(key),
        value: val
      }))
    : [];

  const categoryChartData = stats
    ? Object.entries(stats.category_distribution)
        .map(([key, val]) => ({
          name: key.charAt(0).toUpperCase() + key.slice(1),
          count: val
        }))
        .sort((a, b) => b.count - a.count)
    : [];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-[#00543B]/20 border-t-[#00543B] rounded-full animate-spin" />
          <span className="text-gray-500 text-sm font-semibold">Reading database structures...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Top Banner explaining the "Why" */}
      <div className="bg-gradient-to-r from-[#00543B] to-[#138B7B] text-white p-6 rounded-2xl shadow-sm relative overflow-hidden">
        <div className="absolute right-0 top-0 translate-x-10 -translate-y-10 opacity-10">
          <Database className="w-64 h-64" />
        </div>
        <div className="max-w-3xl relative z-10">
          <div className="flex items-center gap-2 px-3 py-1 bg-white/10 rounded-full border border-white/20 text-xs font-bold w-fit mb-4">
            <Cpu className="w-3.5 h-3.5" />
            <span>DPDP & KYC Compliant Development Environment</span>
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight">CreditSetu Synthetic Data Engine</h1>
          <p className="text-sm mt-2 text-white/80 leading-relaxed font-medium">
            In compliance with retail lending regulations and customer data privacy protections (DPDP Act), CreditSetu runs its evaluations on statistically calibrated synthetic data. The pipeline simulates complete transaction streams structurally identical to real Account Aggregator consent flows, allowing IDBI relationship managers to validate risk models and behavioral features safely.
          </p>
        </div>
      </div>

      {/* Database KPI Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
          <KpiCard
            label="Total Customers Profiled"
            value={stats.total_customers.toLocaleString()}
            description="Active database accounts evaluated by composite scorer."
            icon={<Users className="w-5 h-5" />}
            colorClass="text-[#00543B]"
            bgClass="bg-[#00543B]/10"
          />
          <KpiCard
            label="Ingested UPI & Bank Ledgers"
            value={stats.total_transactions.toLocaleString()}
            description="Individual deposit/debit transactions generated."
            icon={<Database className="w-5 h-5" />}
            colorClass="text-[#138B7B]"
            bgClass="bg-[#138B7B]/10"
          />
          <KpiCard
            label="New-To-Credit / Thin-File"
            value={`${((stats.ntc_count / stats.total_customers) * 100).toFixed(0)}%`}
            description={`${stats.ntc_count.toLocaleString()} customers with zero traditional credit history.`}
            icon={<TrendingUp className="w-5 h-5" />}
            colorClass="text-[#F37021]"
            bgClass="bg-[#F37021]/10"
          />
          <KpiCard
            label="Avg Ledger Rows / Customer"
            value={stats.avg_transactions_per_customer}
            description="Mean transactions processed in feature engine per client."
            icon={<Layers className="w-5 h-5" />}
            colorClass="text-slate-600"
            bgClass="bg-slate-100"
          />
        </div>
      )}

      {/* Interactive Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Persona Distribution Pie Chart */}
        <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
          <h3 className="text-sm font-bold text-gray-800 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Users className="w-4 h-4 text-[#00543B]" />
            Seeded Persona Distributions
          </h3>
          <p className="text-xs text-gray-500 font-medium mb-4">Breakdown of customer segments engineered in the database</p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={personaChartData}
                  cx="50%"
                  cy="45%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {personaChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [value, 'Accounts']} />
                <Legend layout="horizontal" verticalAlign="bottom" wrapperStyle={{ fontSize: '10px', fontWeight: 'bold' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Transaction Categories Distribution Bar Chart */}
        <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
          <h3 className="text-sm font-bold text-gray-800 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Database className="w-4 h-4 text-[#138B7B]" />
            Ledger Transaction Allocations
          </h3>
          <p className="text-xs text-gray-500 font-medium mb-4">Volume distribution across transaction category types</p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryChartData} layout="vertical" margin={{ top: 5, right: 10, left: 30, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis type="number" tick={{ fontSize: 10, fill: '#64748B', fontWeight: 600 }} stroke="#CBD5E1" />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: '#475569', fontWeight: 600 }} stroke="#CBD5E1" width={65} />
                <Tooltip formatter={(value) => [value.toLocaleString(), 'Transactions']} />
                <Bar dataKey="count" fill="#138B7B" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Interactive Database Schema Explorer */}
      <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
        <div className="p-5 border-b border-gray-200 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-gray-50/50">
          <div>
            <h3 className="text-sm font-extrabold text-gray-800 uppercase tracking-wider flex items-center gap-1.5">
              <Table className="w-4 h-4 text-[#00543B]" />
              Database Table Schemas
            </h3>
            <p className="text-xs text-gray-500 mt-1 font-medium">Select a table to view columns, types, and real-world system data sources.</p>
          </div>
          <div className="flex gap-2 bg-gray-100 p-1.5 rounded-lg border border-gray-200 w-fit">
            {Object.keys(TABLE_SCHEMAS).map((key) => (
              <button
                key={key}
                onClick={() => setActiveSchema(key)}
                className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all uppercase tracking-wider ${
                  activeSchema === key
                    ? 'bg-white text-[#00543B] shadow-sm'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                {key}
              </button>
            ))}
          </div>
        </div>

        <div className="p-6">
          <div className="mb-4">
            <h4 className="text-md font-bold text-gray-900">{TABLE_SCHEMAS[activeSchema].title}</h4>
            <p className="text-xs text-gray-500 mt-1 font-medium">{TABLE_SCHEMAS[activeSchema].description}</p>
            <div className="mt-2.5 inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold rounded-full">
              <Layers className="w-3.5 h-3.5" />
              <span>Real Equivalent Source: {TABLE_SCHEMAS[activeSchema].source}</span>
            </div>
          </div>

          <div className="overflow-x-auto border border-gray-200 rounded-lg">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="px-4 py-3 text-xs font-extrabold text-gray-400 uppercase tracking-wider">Column</th>
                  <th className="px-4 py-3 text-xs font-extrabold text-gray-400 uppercase tracking-wider">SQL Data Type</th>
                  <th className="px-4 py-3 text-xs font-extrabold text-gray-400 uppercase tracking-wider">Constraint</th>
                  <th className="px-4 py-3 text-xs font-extrabold text-gray-400 uppercase tracking-wider">Ingest Origin</th>
                  <th className="px-4 py-3 text-xs font-extrabold text-gray-400 uppercase tracking-wider">Operational Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {TABLE_SCHEMAS[activeSchema].columns.map((col, i) => (
                  <tr key={i} className="hover:bg-gray-50/50">
                    <td className="px-4 py-3 text-sm font-extrabold text-gray-800">{col.name}</td>
                    <td className="px-4 py-3 text-sm font-mono text-gray-600 font-bold bg-gray-50/50">{col.type}</td>
                    <td className="px-4 py-3 text-xs">
                      {col.key !== '-' ? (
                        <span className="px-2 py-0.5 bg-amber-50 text-amber-700 border border-amber-200 rounded font-extrabold">
                          {col.key}
                        </span>
                      ) : (
                        <span className="text-gray-400 font-bold">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs font-bold text-gray-600">{col.source}</td>
                    <td className="px-4 py-3 text-xs font-medium text-gray-500 leading-relaxed max-w-sm">{col.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Behavioral Persona Profiles Accordion */}
      <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
        <h3 className="text-sm font-bold text-gray-800 uppercase tracking-wider mb-4 flex items-center gap-1.5">
          <Users className="w-4 h-4 text-[#F37021]" />
          Behavioral Persona Profiles Inspector
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <PersonaCard
            title="Salaried Stable"
            description="Represents traditional retail banking clients with stable employment. Typical profiles: IT Professionals, Government Employees."
            markers={['High CIBIL score (700-800)', 'Fixed-date monthly payroll credits', 'Consistent monthly rent/utilities debits', 'Zero transaction bounces']}
            fit="High composite score, mapped for standard high-value Home Loans or premium Credit Cards."
            badgeClass="bg-emerald-50 text-emerald-800 border-emerald-200"
          />
          <PersonaCard
            title="Salaried Unstable"
            description="Represents mid-to-lower income corporate employees with occasional income delays or slightly higher transaction bounce counts."
            markers={['Medium CIBIL score (600-680)', 'Varying monthly payroll credit dates', 'Occasional NACH return debits', 'High EMI burden ratio']}
            fit="Moderate composite score, mapped for Auto Loans or Personal Loans with closer credit analyst watch."
            badgeClass="bg-amber-50 text-amber-800 border-amber-200"
          />
          <PersonaCard
            title="Gig Economy Worker"
            description="Represents the modern freelance/gig workforce segment (e.g. Swiggy delivery, Uber partners). Zero fixed monthly salary credits."
            markers={['Null or Thin Bureau file', 'Fragmented, frequent UPI credits from multiple aggregators', 'Highly volatile daily net flow', 'Diverse spending category patterns']}
            fit="Scored behaviorally using UPI credit frequency, mapped to Micro-Credit Lines or Retail Credit Cards."
            badgeClass="bg-[#138B7B]/10 text-[#138B7B] border-[#138B7B]/20"
          />
          <PersonaCard
            title="New-To-Credit (NTC)"
            description="Represents young professionals or students entering the credit market for the first time. Completely invisible to bureaus."
            markers={['Exactly Null bureau score', 'Regular entry-level salary credits', 'Stable monthly cash surplus', 'Good rent consistency score']}
            fit="Evaluated behaviorally. Mapped to entry-level Retail Credit Cards to establish credit history."
            badgeClass="bg-[#00543B]/10 text-[#00543B] border-[#00543B]/20"
          />
          <div className="md:col-span-2">
            <PersonaCard
              title="Overleveraged Stressed"
              description="Represents high-risk borrowers displaying critical financial distress signals. Explicitly caught by risk guardrails."
              markers={['Low CIBIL score (<580)', 'Numerous micro-credit EMIs running concurrently', 'Frequent NACH bounces (2+ in 3 months)', 'Negative monthly cash surplus']}
              fit="Assigned to the Suppress Tier, protecting the bank's asset quality from high delinquency risk."
              badgeClass="bg-red-50 text-red-800 border-red-200"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function KpiCard({ label, value, description, icon, colorClass, bgClass }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm flex flex-col justify-between hover:shadow-md transition-shadow">
      <div>
        <div className="flex items-start justify-between mb-3">
          <div className={`p-2 rounded-lg ${bgClass} ${colorClass}`}>
            {icon}
          </div>
        </div>
        <div className={`text-2xl font-extrabold ${colorClass}`}>{value}</div>
        <div className="text-xs font-bold text-gray-700 mt-1">{label}</div>
      </div>
      {description && (
        <p className="text-[10px] text-gray-500 font-semibold mt-2 leading-relaxed">{description}</p>
      )}
    </div>
  );
}

function PersonaCard({ title, description, markers, fit, badgeClass }) {
  return (
    <div className="bg-gray-50/50 border border-gray-200 rounded-xl p-5 flex flex-col justify-between hover:border-gray-300 transition-colors">
      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-extrabold text-gray-900">{title}</h4>
          <span className={`px-2 py-0.5 text-[10px] font-extrabold border rounded-full uppercase tracking-wider ${badgeClass}`}>
            Archetype
          </span>
        </div>
        <p className="text-xs text-gray-500 font-semibold leading-relaxed mb-4">{description}</p>
        
        <div className="space-y-1.5 mb-4">
          <h5 className="text-[10px] font-extrabold text-gray-400 uppercase tracking-wider">Transactional Markers:</h5>
          <ul className="space-y-1">
            {markers.map((m, i) => (
              <li key={i} className="text-xs font-semibold text-gray-600 flex items-center gap-1.5">
                <span className="w-1 h-1 rounded-full bg-gray-400" />
                {m}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="pt-3 border-t border-gray-200 text-xs leading-relaxed font-semibold text-[#00543B] flex gap-1.5">
        <Info className="w-4 h-4 shrink-0 text-[#138B7B]" />
        <span><strong className="text-gray-900">Lead Match:</strong> {fit}</span>
      </div>
    </div>
  );
}
