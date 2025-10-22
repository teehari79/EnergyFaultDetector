import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import { config } from './config.js';
import { connectToDatabase } from './db.js';
import authRoutes from './routes/auth.js';
import jobsRoutes from './routes/jobs.js';
import { errorHandler } from './middleware/errorHandler.js';

async function startServer() {
  await connectToDatabase();

  const app = express();
  app.use(cors({ origin: config.corsOrigin === '*' ? true : config.corsOrigin, credentials: true }));
  app.use(express.json({ limit: '2mb' }));
  app.use(morgan('dev'));

  app.use('/api/auth', authRoutes);
  app.use('/api/jobs', jobsRoutes);

  app.use(errorHandler);

  app.listen(config.port, () => {
    console.log(`Node service listening on port ${config.port}`);
  });
}

startServer().catch((error) => {
  console.error('Failed to start Node service:', error);
  process.exit(1);
});
