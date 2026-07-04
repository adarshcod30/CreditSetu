import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Area, AreaChart
} from 'recharts';

export default function TransactionTimeline({ transactions, changePoints }) {
  const dailyData = useMemo(() => {
    if (!transactions || transactions.length === 0) return [];

    // Aggregate to daily net cash flow
    const dailyMap = {};
    transactions.forEach(txn => {
      const date = txn.date;
      if (!dailyMap[date]) {
        dailyMap[date] = { date, credits: 0, debits: 0, net: 0 };
      }
      if (txn.type === 'credit') {
        dailyMap[date].credits += txn.amount;
      } else {
        dailyMap[date].debits += txn.amount;
      }
      dailyMap[date].net = dailyMap[date].credits - dailyMap[date].debits;
    });

    // Sort by date and compute rolling 7-day average
    const sorted = Object.values(dailyMap).sort((a, b) => a.date.localeCompare(b.date));

    // Compute rolling average for smoother visualization
    for (let i = 0; i < sorted.length; i++) {
      const window = sorted.slice(Math.max(0, i - 6), i + 1);
      sorted[i].rolling_avg = window.reduce((sum, d) => sum + d.net, 0) / window.length;
    }

    // Subsample if too many data points (keep every Nth for readability)
    if (sorted.length > 180) {
      const step = Math.ceil(sorted.length / 180);
      return sorted.filter((_, i) => i % step === 0);
    }

    return sorted;
  }, [transactions]);

  if (dailyData.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-lg font-bold text-gray-800 mb-4">Cash Flow Timeline</h3>
        <p className="text-gray-500 text-sm">No transaction data available.</p>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload) return null;
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-3 shadow-xl text-gray-800">
        <p className="text-xs font-bold text-gray-400 mb-1">{label}</p>
        {payload.map((entry, i) => (
          <p key={i} className="text-sm font-extrabold" style={{ color: entry.color }}>
            {entry.name}: ₹{Math.round(entry.value).toLocaleString('en-IN')}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
      <h3 className="text-md font-extrabold text-gray-800 mb-4 uppercase tracking-wider">
        Daily Net Cash Flow
        <span className="text-xs font-medium text-gray-500 ml-2">(7-day rolling average)</span>
      </h3>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={dailyData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
            <defs>
              <linearGradient id="colorNet" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#138B7B" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#138B7B" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: '#64748B', fontWeight: 600 }}
              interval="preserveStartEnd"
              tickFormatter={(d) => {
                const date = new Date(d);
                return `${date.getDate()}/${date.getMonth() + 1}`;
              }}
              stroke="#CBD5E1"
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#64748B', fontWeight: 600 }}
              tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`}
              stroke="#CBD5E1"
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0} stroke="#94A3B8" strokeDasharray="3 3" />

            {/* Mark change points */}
            {changePoints && changePoints.map((cp, i) => (
              <ReferenceLine
                key={i}
                x={cp.date}
                stroke="#F37021"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{
                  value: cp.event_type?.replace(/_/g, ' ') || 'event',
                  position: 'top',
                  fill: '#F37021',
                  fontSize: 10,
                  fontWeight: 'bold',
                }}
              />
            ))}

            <Area
              type="monotone"
              dataKey="rolling_avg"
              name="Net Cash Flow (7d avg)"
              stroke="#138B7B"
              strokeWidth={2.5}
              fill="url(#colorNet)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      {changePoints && changePoints.length > 0 && (
        <div className="mt-4 flex items-center gap-4 text-xs text-gray-500 font-semibold">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-0.5" style={{borderTop: '2.5px dashed #F37021'}} />
            <span>Attributed Life Event (Change-Point Detector)</span>
          </div>
        </div>
      )}
    </div>
  );
}
