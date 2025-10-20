import axios from 'axios';
import { API_BASE_URL, NARRATIVE_ENDPOINT } from '../config.js';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000
});

export const uploadDataset = async (file, metadata = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  Object.entries(metadata).forEach(([key, value]) => {
    formData.append(key, value);
  });

  const { data } = await client.post('/api/predictions', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return data;
};

export const fetchNarrative = async (predictionId) => {
  const { data } = await client.post(NARRATIVE_ENDPOINT, { prediction_id: predictionId });
  return data;
};

export const downloadReport = async (predictionId) => {
  const response = await client.get(`/api/predictions/${predictionId}/report`, {
    responseType: 'blob'
  });

  return response.data;
};

export default client;
