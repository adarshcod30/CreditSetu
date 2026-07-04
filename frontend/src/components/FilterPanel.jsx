import React from 'react';

export default function FilterPanel({
  minScore,
  setMinScore,
  productType,
  setProductType,
  showSuppressed,
  setShowSuppressed,
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
      <div className="flex flex-wrap items-center gap-8">
        {/* Min Score Slider */}
        <div className="flex items-center gap-3">
          <label className="text-sm font-bold text-gray-700 whitespace-nowrap">
            Minimum Score
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={minScore}
            onChange={(e) => setMinScore(parseFloat(e.target.value))}
            className="w-36 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#00543B]"
          />
          <span className="text-sm font-extrabold text-[#00543B] w-12 text-center bg-[#00543B]/10 px-2 py-0.5 rounded">
            {(minScore * 100).toFixed(0)}%
          </span>
        </div>

        {/* Product Type Dropdown */}
        <div className="flex items-center gap-3">
          <label className="text-sm font-bold text-gray-700 whitespace-nowrap">
            Product Category
          </label>
          <select
            value={productType}
            onChange={(e) => setProductType(e.target.value)}
            className="bg-white border border-gray-300 text-gray-800 text-sm rounded-lg px-3 py-2 focus:ring-2 focus:ring-[#00543B] focus:border-[#00543B] outline-none font-medium shadow-sm transition-all"
          >
            <option value="">All Products</option>
            <option value="Micro-Credit Line">Micro-Credit Line</option>
            <option value="Auto Loan">Auto Loan</option>
            <option value="Personal Loan">Personal Loan</option>
            <option value="Home Loan">Home Loan</option>
            <option value="Retail Credit Card">Retail Credit Card</option>
          </select>
        </div>

        {/* Show Suppressed Toggle */}
        <div className="flex items-center gap-3">
          <label className="text-sm font-bold text-gray-700 whitespace-nowrap">
            Show Suppressed Risk
          </label>
          <button
            onClick={() => setShowSuppressed(!showSuppressed)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 outline-none ${
              showSuppressed ? 'bg-[#00543B]' : 'bg-gray-300'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ${
                showSuppressed ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
          <span className={`text-xs font-bold ${showSuppressed ? 'text-[#00543B]' : 'text-gray-500'}`}>
            {showSuppressed ? 'Visible' : 'Hidden'}
          </span>
        </div>
      </div>
    </div>
  );
}
