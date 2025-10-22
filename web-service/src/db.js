import { MongoClient } from 'mongodb';
import { config } from './config.js';

let client;
let database;

const COLLECTION_NAMES = {
  jobs: 'jobs',
  sessions: 'sessions'
};

export async function connectToDatabase() {
  if (database) {
    return database;
  }

  client = new MongoClient(config.mongoUri, {
    maxPoolSize: 10
  });
  await client.connect();
  database = client.db(config.mongoDbName);

  await Promise.all([
    database.collection(COLLECTION_NAMES.jobs).createIndex({ jobId: 1 }, { unique: true }),
    database
      .collection(COLLECTION_NAMES.jobs)
      .createIndex({ organizationId: 1, username: 1, status: 1, updatedAt: -1 }),
    database.collection(COLLECTION_NAMES.sessions).createIndex({ authToken: 1 }, { unique: true }),
    database
      .collection(COLLECTION_NAMES.sessions)
      .createIndex({ expiresAt: 1 }, { expireAfterSeconds: 0 })
  ]).catch((error) => {
    // Index creation is best-effort; log and continue.
    console.warn('Failed to ensure MongoDB indexes:', error);
  });

  return database;
}

export function getDatabase() {
  if (!database) {
    throw new Error('Database connection has not been initialised.');
  }
  return database;
}

export function getJobsCollection() {
  return getDatabase().collection(COLLECTION_NAMES.jobs);
}

export function getSessionsCollection() {
  return getDatabase().collection(COLLECTION_NAMES.sessions);
}

export async function closeDatabase() {
  if (client) {
    await client.close();
    client = null;
    database = null;
  }
}
