import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, BarChart3, Wallet } from 'lucide-react';
import LeadDashboard from './pages/LeadDashboard';
import CustomerDetail from './pages/CustomerDetail';
import BenchmarkView from './pages/BenchmarkView';

function NavItem({ to, icon, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-150 ${
          isActive
            ? 'bg-[#00543B] text-white shadow-sm'
            : 'text-gray-700 hover:text-[#00543B] hover:bg-gray-100'
        }`
      }
    >
      {icon}
      <span>{label}</span>
    </NavLink>
  );
}

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#F7F8FA] text-gray-900 flex flex-col">
        {/* Navigation Header */}
        <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
          {/* Top Bar for IDBI Logo and Branding */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo / Brand */}
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  {/* Styled IDBI Bank-like icon logo */}
                  <div className="w-10 h-10 rounded-full bg-[#00543B] flex items-center justify-center border-2 border-[#F37021]">
                    <Wallet className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex flex-col leading-none">
                    <span className="text-xl font-extrabold tracking-tight text-[#00543B]">
                      IDBI <span className="text-[#F37021]">BANK</span>
                    </span>
                    <span className="text-[10px] uppercase font-bold tracking-widest text-[#138B7B] mt-0.5">
                      CreditSetu AI Engine
                    </span>
                  </div>
                </div>
              </div>

              {/* Nav Links */}
              <nav className="flex items-center gap-2">
                <NavItem to="/" icon={<LayoutDashboard className="w-4.5 h-4.5" />} label="Lead Dashboard" />
                <NavItem to="/benchmarks" icon={<BarChart3 className="w-4.5 h-4.5" />} label="Benchmarks" />
              </nav>

              {/* System status tags */}
              <div className="flex items-center gap-3">
                <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-emerald-50 rounded-full border border-emerald-200">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-xs font-bold text-emerald-700">API: Connected</span>
                </div>
                <div className="px-3 py-1 bg-[#00543B]/10 rounded-full border border-[#00543B]/20 text-xs font-bold text-[#00543B]">
                  Database: Seeded
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full">
          <Routes>
            <Route path="/" element={<LeadDashboard />} />
            <Route path="/customer/:customerId" element={<CustomerDetail />} />
            <Route path="/benchmarks" element={<BenchmarkView />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 py-4 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs text-gray-500">
            IDBI Innovate 2026 Hackathon - Track 02 | Lead Generation, Behavioural Analytics, Retail Lending
          </div>
        </footer>
      </div>
    </Router>
  );
}
