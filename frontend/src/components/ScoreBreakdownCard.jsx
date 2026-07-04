import React from 'react';
import { TrendingUp, Shield, Zap, AlertTriangle, CheckCircle } from 'lucide-react';

function ScoreBar({ label, score, icon, colorClass, barColorClass }) {
  const percentage = Math.round(score * 100);
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-bold text-gray-700">{label}</span>
        </div>
        <span className={`text-sm font-extrabold ${colorClass}`}>{percentage}%</span>
      </div>
      <div className="w-full h-2.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${barColorClass}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default function ScoreBreakdownCard({ score }) {
  if (!score) return null;

  const tierConfig = {
    Safe: {
      color: 'text-emerald-700',
      bg: 'bg-emerald-50 border-emerald-200',
      icon: <CheckCircle className="w-4.5 h-4.5 text-emerald-600" />
    },
    Watch: {
      color: 'text-amber-800',
      bg: 'bg-amber-50 border-amber-200',
      icon: <Shield className="w-4.5 h-4.5 text-amber-600" />
    },
    Suppress: {
      color: 'text-red-700',
      bg: 'bg-red-50 border-red-200',
      icon: <AlertTriangle className="w-4.5 h-4.5 text-red-600" />
    },
  };

  const tier = tierConfig[score.guardrail_tier] || tierConfig.Safe;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-100 pb-4">
        <h3 className="text-md font-extrabold text-gray-800 uppercase tracking-wider">Score Breakdown</h3>
        <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border ${tier.bg}`}>
          {tier.icon}
          <span className={`text-xs font-extrabold uppercase ${tier.color}`}>{score.guardrail_tier}</span>
        </div>
      </div>

      {/* Composite Score Circle/Display */}
      <div className="text-center py-2 flex flex-col items-center">
        <div className="w-28 h-28 rounded-full border-4 border-[#00543B]/10 flex flex-col items-center justify-center bg-gray-50 shadow-inner">
          <span className="text-4xl font-extrabold text-[#00543B]">
            {(score.composite_score * 100).toFixed(0)}%
          </span>
        </div>
        <div className="text-xs font-bold text-gray-500 uppercase tracking-widest mt-3">Composite Lead Score</div>
      </div>

      {/* Sub-scores */}
      <div className="space-y-4">
        <ScoreBar
          label="Intent Signal Score"
          score={score.intent_score}
          icon={<Zap className="w-4.5 h-4.5 text-[#138B7B]" />}
          colorClass="text-[#138B7B]"
          barColorClass="bg-[#138B7B]"
        />
        <ScoreBar
          label="Repayment Capacity Score"
          score={score.capacity_score}
          icon={<TrendingUp className="w-4.5 h-4.5 text-[#00543B]" />}
          colorClass="text-[#00543B]"
          barColorClass="bg-[#00543B]"
        />
        <ScoreBar
          label="Guardrail Integrity Score"
          score={1 - score.guardrail_score}
          icon={<Shield className="w-4.5 h-4.5 text-[#F37021]" />}
          colorClass="text-[#F37021]"
          barColorClass="bg-[#F37021]"
        />
      </div>

      {/* Capacity Amount */}
      <div className="bg-gray-50 border border-gray-150 rounded-xl p-4 shadow-sm">
        <div className="text-xs font-bold text-gray-500 uppercase tracking-wider">Estimated Monthly Repayment Capacity</div>
        <div className="text-2xl font-extrabold text-gray-900 mt-1">
          ₹{(score.capacity_amount || 0).toLocaleString('en-IN')}
          <span className="text-xs font-semibold text-gray-500 ml-2">
            ±₹{(score.capacity_confidence || 0).toLocaleString('en-IN')} confidence interval
          </span>
        </div>
      </div>

      {/* Suggested Product */}
      <div className="flex items-center justify-between bg-[#00543B]/5 border border-[#00543B]/10 rounded-xl p-4">
        <span className="text-xs font-bold text-gray-600 uppercase tracking-wider">Suggested Loan Product</span>
        <span className="text-sm font-extrabold text-[#00543B] bg-white border border-[#00543B]/20 px-3 py-1 rounded-lg">
          {score.suggested_product}
        </span>
      </div>

      {/* Intent Event details */}
      {score.intent_event_type && (
        <div className="bg-[#138B7B]/5 border border-[#138B7B]/10 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-1.5">
            <Zap className="w-4 h-4 text-[#138B7B]" />
            <span className="text-xs font-bold text-[#138B7B] uppercase tracking-wider">Life Event Signal</span>
          </div>
          <p className="text-sm font-semibold text-gray-800 capitalize">
            {score.intent_event_type.replace(/_/g, ' ')}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">
            Detected {score.intent_event_recency_days} days ago via transaction change-point model
          </p>
        </div>
      )}

      {/* Guardrail Reasons */}
      {score.guardrail_reasons && score.guardrail_reasons.length > 0 && (
        <div className={`rounded-xl p-4 border ${
          score.guardrail_tier === 'Suppress' ? 'bg-red-50 border-red-200' : 'bg-amber-50 border-amber-200'
        }`}>
          <div className="text-xs font-bold uppercase tracking-wider mb-2 text-gray-700">Lending Risk Warning Flags</div>
          <ul className="space-y-1.5">
            {score.guardrail_reasons.map((reason, i) => (
              <li key={i} className="text-xs font-semibold text-gray-700 flex items-start gap-2">
                <span className="text-red-500 mt-0.5">•</span>
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
