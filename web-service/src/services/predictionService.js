import axios from 'axios';
import FormData from 'form-data';
import { config } from '../config.js';
import { AUTH_CONTEXT, encryptPayload, hashAuthToken } from '../utils/crypto.js';

const datasetClient = axios.create({
  baseURL: config.datasetApiBaseUrl,
  timeout: 120000
});

const predictionClient = axios.create({
  baseURL: config.predictionApiBaseUrl,
  timeout: 120000
});

function normaliseError(error, defaultMessage) {
  if (error.response?.data?.detail) {
    return new Error(error.response.data.detail);
  }
  if (error.response?.data?.error) {
    const err = new Error(error.response.data.error);
    err.status = error.response.status;
    return err;
  }
  if (error.response?.status) {
    const err = new Error(defaultMessage);
    err.status = error.response.status;
    return err;
  }
  return new Error(defaultMessage);
}

export async function authenticate({ organizationId, username, password, seedToken }) {
  try {
    const encrypted = encryptPayload(seedToken, { username, password }, AUTH_CONTEXT);
    const response = await predictionClient.post('/auth', {
      organization_id: organizationId,
      credentials_encrypted: encrypted,
      nonce: null
    });
    return response.data;
  } catch (error) {
    throw normaliseError(error, 'Authentication request failed.');
  }
}

export async function uploadDataset(file, metadata) {
  try {
    const formData = new FormData();
    formData.append('file', file.buffer, {
      filename: file.originalname,
      contentType: file.mimetype
    });
    Object.entries(metadata).forEach(([key, value]) => {
      if (value !== undefined && value !== null && `${value}`.trim() !== '') {
        formData.append(key, `${value}`);
      }
    });

    const response = await datasetClient.post('/api/predictions', formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    return response.data;
  } catch (error) {
    throw normaliseError(error, 'Dataset upload failed.');
  }
}

export async function submitPrediction({
  authToken,
  seedToken,
  payload
}) {
  try {
    const authHash = hashAuthToken(authToken, seedToken);
    const encryptedPayload = encryptPayload(seedToken, payload, authHash);
    const response = await predictionClient.post('/predict', {
      auth_token: authToken,
      auth_hash: authHash,
      payload_encrypted: encryptedPayload
    });
    return response.data;
  } catch (error) {
    throw normaliseError(error, 'Prediction submission failed.');
  }
}

export async function fetchJobStatus(jobId, authToken) {
  try {
    const response = await predictionClient.get(`/jobs/${jobId}`, {
      params: { auth_token: authToken }
    });
    return response.data;
  } catch (error) {
    throw normaliseError(error, 'Fetching job status failed.');
  }
}
