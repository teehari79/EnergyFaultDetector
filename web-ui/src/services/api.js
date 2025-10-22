import axios from 'axios';
import { API_BASE_URL } from '../config.js';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000
});

export const setAuthToken = (token) => {
  if (token) {
    apiClient.defaults.headers.common['X-Auth-Token'] = token;
  }
};

export const clearAuthToken = () => {
  delete apiClient.defaults.headers.common['X-Auth-Token'];
};

export const login = async ({ organizationId, username, password, seedToken }) => {
  const response = await apiClient.post('/api/auth/login', {
    organizationId,
    username,
    password,
    seedToken
  });
  return response.data;
};

export const createPredictionJob = async (file, metadata = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  Object.entries(metadata).forEach(([key, value]) => {
    if (value !== undefined && value !== null && `${value}`.trim() !== '') {
      formData.append(key, value);
    }
  });

  const response = await apiClient.post('/api/jobs', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return response.data;
};

export const fetchJobs = async (filters = {}) => {
  const response = await apiClient.get('/api/jobs', {
    params: filters
  });
  return response.data;
};

export const fetchJobStatus = async (jobId) => {
  const response = await apiClient.get(`/api/jobs/${jobId}`);
  return response.data;
};

export default apiClient;
