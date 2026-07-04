import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, TrendingUp, Shield, AlertTriangle } from 'lucide-react';

function getScoreBadge(score, tier) {
  if (tier === 'Suppress') {
    return 'bg-red-50 text-red-700 border border-red-200';
  }
  if (tier === 'Watch') {
    return 'bg-amber-50 text-amber-755 text-amber-800 border border-amber-200';
  }
  if (score >= 0.7) return 'bg-emerald-50 text-emerald-700 border border-emerald-200';
  if (score >= 0.4) return 'bg-blue-50 text-blue-700 border border-blue-200';
  return 'bg-gray-50 text-gray-600 border border-gray-200';
}

function getTierIcon(tier) {
  switch (tier) {
    case 'Suppress': return <AlertTriangle className="w-4 h-4 text-red-500" />;
    case 'Watch': return <Shield className="w-4 h-4 text-amber-500" />;
    default: return <TrendingUp className="w-4 h-4 text-emerald-500" />;
  }
}

export default function LeadTable({ leads, loading }) {
  const navigate = useNavigate();

  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-12 shadow-sm text-center">
        <div className="flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 border-3 border-[#00543B]/20 border-t-[#00543B] rounded-full animate-spin" />
          <span className="text-gray-500 text-sm font-semibold">Analyzing transaction database...</span>
        </div>
      </div>
    );
  }

  if (!leads || leads.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-12 shadow-sm text-center">
        <p className="text-gray-500 font-medium">No lending leads match the selected filter criteria.</p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="px-6 py-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Customer Profile</th>
              <th className="px-6 py-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Composite Score</th>
              <th className="px-6 py-4 text-xs font-bold text-gray-500 uppercase tracking-wider hidden lg:table-cell">Lending Risk Tier</th>
              <th className="px-6 py-4 text-xs font-bold text-gray-500 uppercase tracking-wider hidden md:table-cell">AI Attribution Summary</th>
              <th className="px-6 py-4 text-xs font-bold text-gray-500 uppercase tracking-wider hidden md:table-cell">Target Loan Product</th>
              <th className="px-6 py-4 text-xs font-bold text-gray-500 uppercase tracking-wider text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {leads.map((lead) => (
              <tr
                key={lead.customer_id}
                className="hover:bg-gray-50/80 transition-colors cursor-pointer"
                onClick={() => navigate(`/customer/${lead.customer_id}`)}
              >
                {/* Customer Info */}
                <td className="px-6 py-4">
                  <div>
                    <div className="font-bold text-gray-900 text-sm">{lead.name}</div>
                    <div className="text-xs text-gray-500 font-medium mt-0.5">
                      {lead.customer_id} · <span className="capitalize">{lead.persona_type.replace(/_/g, ' ')}</span>
                    </div>
                  </div>
                </td>

                {/* Score Badge */}
                <td className="px-6 py-4">
                  <span className={`score-badge ${getScoreBadge(lead.composite_score, lead.guardrail_tier)}`}>
                    {(lead.composite_score * 100).toFixed(0)}%
                  </span>
                </td>

                {/* Risk Tier */}
                <td className="px-6 py-4 hidden lg:table-cell">
                  <div className="flex items-center gap-2">
                    {getTierIcon(lead.guardrail_tier)}
                    <span className={`text-xs font-bold ${
                      lead.guardrail_tier === 'Suppress' ? 'text-red-600' :
                      lead.guardrail_tier === 'Watch' ? 'text-amber-600' :
                      'text-emerald-700'
                    }`}>
                      {lead.guardrail_tier}
                    </span>
                  </div>
                </td>

                {/* AI Attribution */}
                <td className="px-6 py-4 hidden md:table-cell">
                  <p className="text-xs text-gray-600 max-w-xs truncate font-medium">
                    {lead.explanation?.split('.')[0] || 'No signals registered'}
                  </p>
                </td>

                {/* Target Product */}
                <td className="px-6 py-4 hidden md:table-cell">
                  <span className="text-xs font-bold text-[#00543B] bg-[#00543B]/10 px-2.5 py-1 rounded-md">
                    {lead.suggested_product}
                  </span>
                </td>

                {/* Action button */}
                <td className="px-6 py-4 text-right">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(`/customer/${lead.customer_id}`);
                    }}
                    className="inline-flex items-center gap-1.5 text-xs font-bold text-[#00543B] hover:text-[#003c2a] hover:underline transition-all"
                  >
                    <Eye className="w-4 h-4" />
                    <span>View Analytics</span>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
