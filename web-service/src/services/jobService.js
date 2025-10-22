import { ObjectId } from 'mongodb';
import { getJobsCollection } from '../db.js';
import { fetchJobStatus, submitPrediction, uploadDataset } from './predictionService.js';

const FINAL_STATES = new Set(['completed', 'failed']);
const DEFAULT_TIMESTAMP_COLUMN = 'time_stamp';

function computeAnomalyCounts(status) {
  if (!status) {
    return { anomaliesCount: null, criticalAnomaliesCount: null };
  }
  const anomaliesSource =
    status.steps?.event_detection?.payload?.events ?? status.result?.events ?? null;
  const anomaliesCount = Array.isArray(anomaliesSource) ? anomaliesSource.length : null;

  const criticalSource = status.steps?.criticality_analysis?.payload?.events ?? null;
  const criticalAnomaliesCount = Array.isArray(criticalSource) ? criticalSource.length : null;

  return { anomaliesCount, criticalAnomaliesCount };
}

function normaliseNumber(value) {
  if (value === undefined || value === null || value === '') {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isNaN(parsed) ? undefined : parsed;
}

function buildPredictionPayload({
  organizationId,
  dataPath,
  metadata
}) {
  const payload = {
    organization_id: organizationId,
    request: {
      model_name: metadata.model_name,
      data_path: dataPath,
      timestamp_column: metadata.timestamp_column || DEFAULT_TIMESTAMP_COLUMN
    }
  };

  if (metadata.asset_name) {
    payload.request.asset_name = metadata.asset_name;
  }
  if (metadata.farm_name) {
    payload.request.farm_name = metadata.farm_name;
  }
  if (metadata.model_version) {
    payload.request.model_version = metadata.model_version;
  }

  const minEventLength = normaliseNumber(metadata.min_event_length);
  if (minEventLength !== undefined) {
    payload.request.min_event_length = minEventLength;
  }

  if (metadata.min_event_duration !== undefined && metadata.min_event_duration !== null) {
    const trimmed = `${metadata.min_event_duration}`.trim();
    if (trimmed) {
      payload.request.min_event_duration = trimmed;
    }
  }

  if (metadata.enable_narrative === 'false' || metadata.enable_narrative === false) {
    payload.request.enable_narrative = false;
  }

  return payload;
}

function createJobDocument({
  jobId,
  predictionId,
  session,
  metadata,
  file,
  notes
}) {
  const now = new Date();
  return {
    _id: new ObjectId(),
    jobId,
    predictionId,
    organizationId: session.organizationId,
    username: session.username,
    status: 'queued',
    submittedAt: now,
    updatedAt: now,
    lastSyncedAt: now,
    assetName: metadata.asset_name?.trim() || '',
    farmName: metadata.farm_name?.trim() || '',
    modelName: metadata.model_name?.trim() || '',
    batchName: metadata.batch_name?.trim() || '',
    fileName: file?.originalname || '',
    notes: notes || '',
    anomaliesCount: null,
    criticalAnomaliesCount: null,
    metadata: {
      timestamp_column: metadata.timestamp_column || DEFAULT_TIMESTAMP_COLUMN,
      model_version: metadata.model_version || null,
      min_event_length: metadata.min_event_length ?? null,
      min_event_duration: metadata.min_event_duration ?? null,
      enable_narrative:
        metadata.enable_narrative === 'false' || metadata.enable_narrative === false
          ? false
          : true
    },
    lastSnapshot: null
  };
}

export async function startPredictionJob({ session, file, metadata }) {
  if (!file) {
    const error = new Error('Prediction dataset is required.');
    error.status = 400;
    throw error;
  }
  if (!metadata?.model_name) {
    const error = new Error('Model name must be provided.');
    error.status = 400;
    throw error;
  }

  const uploadResponse = await uploadDataset(file, metadata);
  if (!uploadResponse?.data_path || !uploadResponse?.prediction_id) {
    throw new Error('Dataset upload response missing prediction identifier.');
  }

  const payload = buildPredictionPayload({
    organizationId: session.organizationId,
    dataPath: uploadResponse.data_path,
    metadata
  });
  const predictionResponse = await submitPrediction({
    authToken: session.authToken,
    seedToken: session.seedToken,
    payload
  });

  const jobsCollection = getJobsCollection();
  const existing = await jobsCollection.findOne({ jobId: predictionResponse.job_id });
  if (existing) {
    await jobsCollection.updateOne(
      { _id: existing._id },
      {
        $set: {
          status: 'queued',
          updatedAt: new Date(),
          lastSyncedAt: new Date(),
          predictionId: uploadResponse.prediction_id,
          organizationId: session.organizationId,
          username: session.username,
          assetName: metadata.asset_name?.trim() || existing.assetName,
          farmName: metadata.farm_name?.trim() || existing.farmName,
          modelName: metadata.model_name?.trim() || existing.modelName,
          batchName: metadata.batch_name?.trim() || existing.batchName,
          fileName: file?.originalname || existing.fileName,
          notes: metadata.notes?.trim() || existing.notes || '',
          metadata: {
            ...existing.metadata,
            timestamp_column: metadata.timestamp_column || existing.metadata?.timestamp_column,
            model_version: metadata.model_version ?? existing.metadata?.model_version,
            min_event_length: metadata.min_event_length ?? existing.metadata?.min_event_length,
            min_event_duration: metadata.min_event_duration ?? existing.metadata?.min_event_duration,
            enable_narrative:
              metadata.enable_narrative === 'false' || metadata.enable_narrative === false
                ? false
                : existing.metadata?.enable_narrative
          }
        }
      }
    );
    const refreshed = await jobsCollection.findOne({ _id: existing._id });
    return { job: sanitiseJobDocument(refreshed) };
  }

  const jobDocument = createJobDocument({
    jobId: predictionResponse.job_id,
    predictionId: uploadResponse.prediction_id,
    session,
    metadata,
    file,
    notes: metadata.notes?.trim() || ''
  });

  await jobsCollection.insertOne(jobDocument);
  return { job: sanitiseJobDocument(jobDocument) };
}

export async function synchroniseJob(jobDocument, session) {
  if (!jobDocument) {
    return null;
  }
  if (FINAL_STATES.has(jobDocument.status)) {
    return jobDocument;
  }

  try {
    const status = await fetchJobStatus(jobDocument.jobId, session.authToken);
    const now = new Date();
    const updatedAt = status.updated_at ? new Date(status.updated_at) : now;
    const { anomaliesCount, criticalAnomaliesCount } = computeAnomalyCounts(status);

    const jobsCollection = getJobsCollection();
    await jobsCollection.updateOne(
      { _id: jobDocument._id },
      {
        $set: {
          status: status.status,
          updatedAt,
          lastSyncedAt: now,
          lastSnapshot: status,
          anomaliesCount,
          criticalAnomaliesCount,
          resultPath: status.result_path ?? jobDocument.resultPath ?? null,
          error: status.error ?? null
        }
      }
    );

    return {
      ...jobDocument,
      status: status.status,
      updatedAt,
      lastSyncedAt: now,
      lastSnapshot: status,
      anomaliesCount,
      criticalAnomaliesCount,
      resultPath: status.result_path ?? jobDocument.resultPath ?? null,
      error: status.error ?? null
    };
  } catch (error) {
    // Surface prediction API errors but keep stored record intact.
    if (error.status === 404) {
      const jobsCollection = getJobsCollection();
      await jobsCollection.updateOne(
        { _id: jobDocument._id },
        {
          $set: {
            status: 'unknown',
            error: 'Prediction job no longer available from upstream service.',
            lastSyncedAt: new Date()
          }
        }
      );
    }
    throw error;
  }
}

export function sanitiseJobDocument(jobDocument) {
  if (!jobDocument) {
    return null;
  }
  const {
    _id,
    jobId,
    predictionId,
    organizationId,
    username,
    status,
    submittedAt,
    updatedAt,
    assetName,
    farmName,
    modelName,
    batchName,
    fileName,
    notes,
    anomaliesCount,
    criticalAnomaliesCount,
    metadata,
    lastSnapshot,
    resultPath
  } = jobDocument;

  return {
    id: _id?.toString?.() ?? null,
    jobId,
    predictionId,
    organizationId,
    username,
    status,
    submittedAt,
    updatedAt,
    assetName,
    farmName,
    modelName,
    batchName,
    fileName,
    notes,
    anomaliesCount,
    criticalAnomaliesCount,
    metadata,
    snapshot: lastSnapshot ?? null,
    resultPath: resultPath ?? null
  };
}

export function buildJobQuery(filters, session) {
  const query = buildOwnershipQuery(session);

  if (filters.status === 'completed') {
    query.status = 'completed';
  } else if (filters.status === 'in_progress') {
    query.status = { $nin: Array.from(FINAL_STATES) };
  } else if (filters.status === 'failed') {
    query.status = 'failed';
  }

  if (filters.startDate || filters.endDate) {
    query.submittedAt = {};
    if (filters.startDate) {
      query.submittedAt.$gte = new Date(filters.startDate);
    }
    if (filters.endDate) {
      query.submittedAt.$lte = new Date(filters.endDate);
    }
    if (Object.keys(query.submittedAt).length === 0) {
      delete query.submittedAt;
    }
  }

  if (filters.assetName) {
    query.assetName = { $regex: filters.assetName, $options: 'i' };
  }

  if (filters.farmName) {
    query.farmName = { $regex: filters.farmName, $options: 'i' };
  }

  if (filters.minAnomalies !== undefined || filters.maxAnomalies !== undefined) {
    query.anomaliesCount = {};
    if (filters.minAnomalies !== undefined) {
      query.anomaliesCount.$gte = Number(filters.minAnomalies);
    }
    if (filters.maxAnomalies !== undefined) {
      query.anomaliesCount.$lte = Number(filters.maxAnomalies);
    }
  }

  if (filters.minCritical !== undefined || filters.maxCritical !== undefined) {
    query.criticalAnomaliesCount = {};
    if (filters.minCritical !== undefined) {
      query.criticalAnomaliesCount.$gte = Number(filters.minCritical);
    }
    if (filters.maxCritical !== undefined) {
      query.criticalAnomaliesCount.$lte = Number(filters.maxCritical);
    }
  }

  return query;
}

export async function listJobs({ filters, session }) {
  const jobsCollection = getJobsCollection();
  const query = buildJobQuery(filters, session);
  const cursor = jobsCollection
    .find(query)
    .sort({ updatedAt: -1 });
  const jobs = await cursor.toArray();
  return jobs;
}

export function buildOwnershipQuery(session) {
  return {
    organizationId: session.organizationId,
    username: session.username
  };
}
