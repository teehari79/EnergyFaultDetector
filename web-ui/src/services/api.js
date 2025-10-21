import axios from 'axios';
import { API_BASE_URL, ASYNC_API_BASE_URL } from '../config.js';

const uiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000
});

const predictionClient = axios.create({
  baseURL: ASYNC_API_BASE_URL,
  timeout: 60000
});

export const uploadDataset = async (file, metadata = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  Object.entries(metadata).forEach(([key, value]) => {
    formData.append(key, value);
  });

  const { data } = await uiClient.post('/api/predictions', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return data;
};

export const authenticate = async ({ organizationId, credentialsEncrypted, nonce }) => {
  const response = await predictionClient.post('/auth', {
    organization_id: organizationId,
    credentials_encrypted: credentialsEncrypted,
    nonce
  });
  return response.data;
};

export const submitAsyncPrediction = async ({ authToken, authHash, payloadEncrypted }) => {
  const response = await predictionClient.post('/predict', {
    auth_token: authToken,
    auth_hash: authHash,
    payload_encrypted: payloadEncrypted
  });
  return response.data;
};

export const fetchJobStatus = async (jobId, authToken) => {
  const response = await predictionClient.get(`/jobs/${jobId}`, {
    params: { auth_token: authToken }
  });
  return response.data;
};

export default uiClient;
