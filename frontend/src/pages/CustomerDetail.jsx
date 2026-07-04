import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, User, MapPin, Briefcase, CreditCard } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import api from '../api/client';
import ScoreBreakdownCard from '../components/ScoreBreakdownCard';
import TransactionTimeline from '../components/TransactionTimeline';

export default function CustomerDetail() {
  const { customerId } = useParams();
  const navigate = useNavigate();
  const [customer, setCustomer] = useState(null);
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const [custRes, scoreRes] = await Promise.all([
          api.getCustomer(customerId),
          api.getScore(customerId),
        ]);
        setCustomer(custRes.data);
        setScore(scoreRes.data);
      } catch (err) {
        console.error('Failed to fetch customer:', err);
        setError(err.response?.data?.detail || 'Failed to load customer data');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [customerId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-[#00543B]/20 border-t-[#00543B] rounded-full animate-spin" />
          <span className="text-gray-500 text-sm font-semibold">Loading profile & behavioral signals...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-16">
        <p className="text-red-500 text-lg font-bold">{error}</p>
        <button
          onClick={() => navigate('/')}
          className="mt-4 text-[#00543B] font-bold hover:underline"
        >
          ← Back to Lead Dashboard
        </button>
      </div>
    );
  }

  // Parse SHAP contributions for bar chart
  const shapData = (score?.shap_contributions || [])
    .filter(f => Math.abs(f.contribution) > 0.001)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8)
    .map(f => ({
      name: f.display_name,
      value: parseFloat(f.contribution.toFixed(3)),
    }));

  // Parse change points from score details
  let changePoints = [];
  try {
    if (score?.intent_details) {
      changePoints = score.intent_details.filter(d => d.date);
    }
  } catch (e) {}

  return (
    <div className="space-y-6">
      {/* Back button + Header */}
      <div className="flex items-center gap-4 border-b border-gray-200 pb-4">
        <button
          onClick={() => navigate('/')}
          className="p-2.5 rounded-lg bg-white border border-gray-200 hover:bg-gray-50 transition-colors shadow-sm"
        >
          <ArrowLeft className="w-5 h-5 text-gray-700" />
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-extrabold text-gray-900">{customer?.name}</h1>
          <p className="text-xs text-gray-500 font-semibold mt-0.5">{customerId} · Observation Account</p>
        </div>
        {score && (
          <div className="text-right">
            <div className="text-3xl font-extrabold text-[#00543B]">
              {(score.composite_score * 100).toFixed(0)}%
            </div>
            <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mt-0.5">Composite Score</div>
          </div>
        )}
      </div>

      {/* Customer Info Bar */}
      {customer && (
        <div className="bg-white border border-gray-200 p-5 rounded-xl shadow-sm">
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-y-4 gap-x-6">
            <InfoItem icon={<User className="w-4 h-4 text-gray-500" />} label="Age" value={`${customer.age}, ${customer.gender === 'M' ? 'Male' : 'Female'}`} />
            <InfoItem icon={<Briefcase className="w-4 h-4 text-gray-500" />} label="Occupation" value={customer.occupation} />
            <InfoItem icon={<MapPin className="w-4 h-4 text-gray-500" />} label="City" value={customer.city} />
            <InfoItem icon={<CreditCard className="w-4 h-4 text-gray-500" />} label="Bureau Score" value={customer.bureau_score ? Math.round(customer.bureau_score) : 'Thin File (N/A)'} />
            <InfoItem label="Persona" value={customer.persona_type?.replace(/_/g, ' ')} className="capitalize" />
            <InfoItem label="Monthly Income" value={`₹${(customer.monthly_income || 0).toLocaleString('en-IN')}`} />
            <InfoItem label="Active EMIs" value={customer.emi_count || 0} />
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Score Breakdown - Left column */}
        <div className="lg:col-span-1">
          <ScoreBreakdownCard score={score} />
        </div>

        {/* Charts - Right columns */}
        <div className="lg:col-span-2 space-y-6">
          {/* Explanation Callout */}
          {score?.explanation && (
            <div className="bg-gradient-to-r from-[#00543B]/5 to-[#138B7B]/5 border border-[#00543B]/10 rounded-xl p-5 shadow-sm">
              <h3 className="text-xs font-bold uppercase tracking-wider text-[#00543B] mb-2">AI Signal Attribution</h3>
              <p className="text-sm text-gray-700 leading-relaxed font-medium">{score.explanation}</p>
            </div>
          )}

          {/* Transaction Timeline */}
          <TransactionTimeline
            transactions={customer?.transactions || []}
            changePoints={changePoints}
          />

          {/* SHAP Feature Contributions */}
          {shapData.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
              <h3 className="text-md font-extrabold text-gray-800 mb-4 uppercase tracking-wider">
                Feature Contribution Analysis (SHAP)
              </h3>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={shapData}
                    layout="vertical"
                    margin={{ top: 5, right: 20, left: 120, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                    <XAxis
                      type="number"
                      tick={{ fontSize: 10, fill: '#64748B', fontWeight: 600 }}
                      stroke="#CBD5E1"
                      tickFormatter={(val) => `${val}%`}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fontSize: 11, fill: '#475569', fontWeight: 600 }}
                      width={115}
                      stroke="#CBD5E1"
                    />
                    <Tooltip
                      formatter={(val) => [`${val > 0 ? '+' : ''}${val}%`, 'Relative Influence']}
                      contentStyle={{
                        backgroundColor: '#FFFFFF',
                        border: '1px solid #E2E8F0',
                        borderRadius: '8px',
                        fontSize: '12px',
                        fontWeight: 'bold',
                        color: '#1E293B',
                      }}
                    />
                    <Bar dataKey="value" name="SHAP Impact" radius={[0, 4, 4, 0]}>
                      {shapData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={entry.value > 0 ? '#138B7B' : '#E8452D'}
                          fillOpacity={0.85}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-gray-500 mt-2 font-semibold">
                🟢 Teal represents positive contribution to lead suitability | 🔴 Red/Orange represents risk/negative contribution
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function InfoItem({ icon, label, value, className = "" }) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-1.5 text-xs font-bold text-gray-400 uppercase tracking-wider">
        {icon}
        <span>{label}</span>
      </div>
      <span className={`text-sm font-extrabold text-gray-800 mt-1 ${className}`}>{value}</span>
    </div>
  );
}
