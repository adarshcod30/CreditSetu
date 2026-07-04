import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions
export const api = {
  // Customers
  getCustomers: (params = {}) =>
    client.get('/api/customers', { params }),

  getCustomer: (customerId) =>
    client.get(`/api/customers/${customerId}`),

  // Leads
  getLeads: (params = {}) =>
    client.get('/api/leads', { params }),

  getDashboardStats: () =>
    client.get('/api/leads/stats'),

  // Scores
  getScore: (customerId) =>
    client.get(`/api/score/${customerId}`),

  // Data generation
  generateData: (data = { n_customers: 1000, seed: 42 }) =>
    client.post('/api/data/generate', data),

  // Benchmarks
  runBenchmark: () =>
    client.post('/api/benchmark/run'),

  getLatestBenchmark: () =>
    client.get('/api/benchmark/latest'),

  // Health
  healthCheck: () =>
    client.get('/api/health'),
};

export default api;
