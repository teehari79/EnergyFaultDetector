const DEFAULT_PORT = process.env.PORT ? Number(process.env.PORT) : 4000;

export const config = {
  port: Number.isNaN(DEFAULT_PORT) ? 4000 : DEFAULT_PORT,
  mongoUri: process.env.MONGO_URI || 'mongodb://localhost:27017',
  mongoDbName: process.env.MONGO_DB_NAME || 'energy_fault_detector',
  datasetApiBaseUrl: process.env.DATASET_API_BASE_URL || 'http://localhost:8000',
  predictionApiBaseUrl: process.env.PREDICTION_API_BASE_URL || 'http://localhost:8001',
  corsOrigin: process.env.CORS_ORIGIN || '*',
  sessionTtlSeconds: process.env.SESSION_TTL_SECONDS
    ? Number(process.env.SESSION_TTL_SECONDS)
    : null
};
