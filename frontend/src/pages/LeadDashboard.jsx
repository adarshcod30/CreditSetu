import React, { useState, useEffect, useCallback } from 'react';
import { Users, TrendingUp, ShieldAlert, BarChart3, RefreshCw } from 'lucide-react';
import {
  ResponsiveContainer, PieChart, Pie, Cell,
  BarChart as RechartsBarChart, Bar as RechartsBar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';
import api from '../api/client';
import FilterPanel from '../components/FilterPanel';
import LeadTable from '../components/LeadTable';

function StatCard({ label, value, icon, colorClass, bgClass, subtext }) {
  return (
    <div className="bg-white border border-gray-200/80 rounded-xl p-5 shadow-sm hover:shadow-md transition-all duration-200">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{label}</p>
          <p className="text-3xl font-extrabold mt-1 text-gray-900">{value}</p>
          {subtext && <p className="text-xs text-gray-500 mt-1 font-medium">{subtext}</p>}
        </div>
        <div className={`p-3 rounded-xl ${bgClass} ${colorClass}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

export default function LeadDashboard() {
  const [leads, setLeads] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [seedingStatus, setSeedingStatus] = useState('idle'); // 'idle', 'loading', 'success', 'error'
  const [elapsedTime, setElapsedTime] = useState(0);
  const [page, setPage] = useState(1);
  const [pageInput, setPageInput] = useState('1');
  const [total, setTotal] = useState(0);
  const pageSize = 20;

  // Filters
  const [minScore, setMinScore] = useState(0);
  const [productType, setProductType] = useState('');
  const [showSuppressed, setShowSuppressed] = useState(false);

  const fetchLeads = useCallback(async () => {
    setLoading(true);
    try {
      const params = {
        page,
        page_size: pageSize,
        min_score: minScore,
        exclude_suppressed: !showSuppressed,
      };
      if (productType) params.product_type = productType;

      const [leadsRes, statsRes] = await Promise.all([
        api.getLeads(params),
        api.getDashboardStats(),
      ]);

      setLeads(leadsRes.data.leads);
      setTotal(leadsRes.data.total);
      setStats(statsRes.data);
    } catch (err) {
      console.error('Failed to fetch leads:', err);
    } finally {
      setLoading(false);
    }
  }, [page, minScore, productType, showSuppressed]);

  useEffect(() => {
    fetchLeads();
  }, [fetchLeads]);

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [minScore, productType, showSuppressed]);

  const totalPages = Math.ceil(total / pageSize);

  useEffect(() => {
    setPageInput(page.toString());
  }, [page]);

  const submitPage = () => {
    const val = parseInt(pageInput, 10);
    if (!isNaN(val) && val >= 1 && val <= totalPages) {
      setPage(val);
    } else {
      setPageInput(page.toString());
    }
  };

  // Prepare chart data from stats
  const riskChartData = stats ? [
    { name: 'Safe Leads', value: stats.safe_count, color: '#138B7B' },
    { name: 'Watch List', value: stats.watch_count, color: '#F37021' },
    { name: 'Risk Suppressed', value: stats.suppressed_count, color: '#E8452D' },
  ].filter(d => d.value > 0) : [];

  const productChartData = stats ? Object.entries(stats.product_distribution).map(([name, value]) => ({
    name,
    value,
  })).sort((a, b) => b.value - a.value) : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 pb-4">
        <div>
          <h1 className="text-2xl font-extrabold text-gray-800">Lead Dashboard</h1>
          <p className="text-sm text-gray-500 mt-1 font-medium">
            Ranked retail lending prospects using transaction behaviour models.
          </p>
        </div>
        <div className="flex flex-col items-end gap-1.5">
          <button
            onClick={() => {
              if (stats && stats.total_customers === 0) {
                setSeedingStatus('loading');
                setElapsedTime(0);

                const timerInterval = setInterval(() => {
                  setElapsedTime(prev => prev + 1);
                }, 1000);

                let isFinished = false;

                const pollHealth = async () => {
                  try {
                    const res = await api.healthCheck();
                    if (res.data.status === 'healthy' && res.data.customers_loaded > 0) {
                      isFinished = true;
                      clearInterval(timerInterval);
                      clearInterval(pollInterval);
                      setSeedingStatus('success');
                      setTimeout(() => {
                        window.location.reload();
                      }, 1500);
                    }
                  } catch (e) {
                    console.log('Waiting for backend response...', e);
                  }
                };

                const pollInterval = setInterval(pollHealth, 5000);

                api.generateData({ n_customers: 500, seed: 42 })
                  .then(() => {
                    if (!isFinished) {
                      isFinished = true;
                      clearInterval(timerInterval);
                      clearInterval(pollInterval);
                      setSeedingStatus('success');
                      setTimeout(() => {
                        window.location.reload();
                      }, 1500);
                    }
                  })
                  .catch((err) => {
                    console.log('Seeding request status pending on backend. Continuing background polling...', err);
                  });

                // Safety timeout after 10 minutes
                setTimeout(() => {
                  if (!isFinished) {
                    isFinished = true;
                    clearInterval(timerInterval);
                    clearInterval(pollInterval);
                    setSeedingStatus('error');
                  }
                }, 600000);
              } else {
                fetchLeads();
              }
            }}
            disabled={loading || seedingStatus === 'loading'}
            className="flex items-center gap-2 px-4 py-2 bg-[#F37021] hover:bg-[#d65f1a] disabled:opacity-50 text-white text-sm font-semibold rounded-lg shadow-sm transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${(loading || seedingStatus === 'loading') ? 'animate-spin' : ''}`} />
            {seedingStatus === 'loading' ? 'Seeding Database...' : 'Refresh Data'}
          </button>
          
          {stats && stats.total_customers === 0 && seedingStatus === 'idle' && (
            <span className="text-xs font-bold text-red-500 animate-pulse">❌ Data not available</span>
          )}
          {seedingStatus === 'loading' && (
            <span className="text-xs font-bold text-[#138B7B] animate-pulse">
              ⏳ Data loading... (Est: ~15-30s | Elapsed: {elapsedTime >= 60 ? `${Math.floor(elapsedTime / 60)}m ${elapsedTime % 60}s` : `${elapsedTime}s`})
            </span>
          )}
          {seedingStatus === 'success' && (
            <span className="text-xs font-bold text-emerald-600">✅ Data Loaded! Refreshing dashboard...</span>
          )}
          {seedingStatus === 'error' && (
            <span className="text-xs font-bold text-red-600">⚠️ Seeding failed. Please check backend connection.</span>
          )}
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Total Database"
            value={stats.total_customers.toLocaleString()}
            icon={<Users className="w-6 h-6" />}
            colorClass="text-[#00543B]"
            bgClass="bg-[#00543B]/10"
          />
          <StatCard
            label="Qualified Leads"
            value={stats.total_leads.toLocaleString()}
            icon={<TrendingUp className="w-6 h-6" />}
            colorClass="text-[#138B7B]"
            bgClass="bg-[#138B7B]/10"
            subtext={`${((stats.total_leads / stats.total_customers) * 100).toFixed(0)}% qualified rate`}
          />
          <StatCard
            label="Risk Suppressed"
            value={stats.suppressed_count.toLocaleString()}
            icon={<ShieldAlert className="w-6 h-6" />}
            colorClass="text-red-600"
            bgClass="bg-red-50"
            subtext="Over-leveraged / stress flagged"
          />
          <StatCard
            label="Average Score"
            value={`${(stats.avg_composite_score * 100).toFixed(0)}%`}
            icon={<BarChart3 className="w-6 h-6" />}
            colorClass="text-[#F37021]"
            bgClass="bg-[#F37021]/10"
          />
        </div>
      )}

      {/* Analytics Visualizations Section */}
      {stats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Risk Tiers Distribution */}
          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <h3 className="text-sm font-bold text-gray-800 uppercase tracking-wider mb-4">Lending Risk Distribution</h3>
            <div className="h-60 flex flex-col justify-between">
              <ResponsiveContainer width="100%" height="85%">
                <PieChart>
                  <Pie
                    data={riskChartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={75}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {riskChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${value} customers`, 'Count']} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-6 text-xs font-semibold">
                {riskChartData.map((entry, idx) => (
                  <div key={idx} className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                    <span className="text-gray-600">{entry.name}: {entry.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Product Offer Distribution */}
          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <h3 className="text-sm font-bold text-gray-800 uppercase tracking-wider mb-4">Target Product Suitability</h3>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart data={productChartData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748B', fontWeight: 600 }} stroke="#CBD5E1" />
                  <YAxis tick={{ fontSize: 10, fill: '#64748B', fontWeight: 600 }} stroke="#CBD5E1" />
                  <Tooltip formatter={(value) => [`${value} leads`, 'Eligible Leads']} />
                  <RechartsBar dataKey="value" name="Eligible Leads" fill="#00543B" radius={[4, 4, 0, 0]} />
                </RechartsBarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <FilterPanel
        minScore={minScore}
        setMinScore={setMinScore}
        productType={productType}
        setProductType={setProductType}
        showSuppressed={showSuppressed}
        setShowSuppressed={setShowSuppressed}
      />

      {/* Lead Table */}
      <LeadTable leads={leads} loading={loading} />

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-4 border-t border-gray-200">
          <p className="text-sm text-gray-600">
            Showing {((page - 1) * pageSize) + 1}–{Math.min(page * pageSize, total)} of {total} leads
          </p>
          <div className="flex items-center gap-4">
            <button
              disabled={page <= 1}
              onClick={() => setPage(p => p - 1)}
              className="px-3.5 py-1.5 text-sm font-semibold bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-40 disabled:hover:bg-white rounded-lg transition-colors text-gray-700 shadow-sm"
            >
              Previous
            </button>
            
            <div className="flex items-center gap-1.5">
              <span className="text-sm text-gray-600 font-medium">Page</span>
              <input
                type="text"
                value={pageInput}
                onChange={(e) => setPageInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    submitPage();
                  }
                }}
                onBlur={submitPage}
                className="w-12 px-2 py-1 text-sm font-bold text-center border border-gray-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-[#00543B] text-gray-800"
              />
              <span className="text-sm text-gray-600 font-medium">of {totalPages}</span>
            </div>

            <button
              disabled={page >= totalPages}
              onClick={() => setPage(p => p + 1)}
              className="px-3.5 py-1.5 text-sm font-semibold bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-40 disabled:hover:bg-white rounded-lg transition-colors text-gray-700 shadow-sm"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="bg-gray-50 rounded-xl p-4 border border-gray-200/60 text-center">
        <p className="text-xs text-gray-500 font-medium">
          ⚠️ Demo Environment Disclaimer: All customer names, details, and UPI/AA transactions are synthetically generated.
          In a production deployment, this engine directly consumes secure, consent-based bank data streams via Sahamati-compliant Account Aggregator API integrations.
        </p>
      </div>
    </div>
  );
}
