import express from 'express';
import multer from 'multer';
import { authenticateRequest } from '../middleware/authenticate.js';
import {
  buildJobQuery,
  buildOwnershipQuery,
  listJobs,
  sanitiseJobDocument,
  startPredictionJob,
  synchroniseJob
} from '../services/jobService.js';
import { getJobsCollection } from '../db.js';

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 200 * 1024 * 1024 } });

const parseOptionalNumber = (value) => {
  if (value === undefined) {
    return undefined;
  }
  const trimmed = `${value}`.trim();
  if (trimmed === '') {
    return undefined;
  }
  const parsed = Number(trimmed);
  return Number.isNaN(parsed) ? undefined : parsed;
};

router.use(authenticateRequest);

router.post('/', upload.single('file'), async (req, res, next) => {
  try {
    const metadata = req.body || {};
    const file = req.file;
    const { job } = await startPredictionJob({
      session: req.session,
      file,
      metadata
    });
    res.status(202).json({ job });
  } catch (error) {
    next(error);
  }
});

router.get('/', async (req, res, next) => {
  try {
    const filters = {
      status: req.query.status ?? 'in_progress',
      startDate: req.query.startDate,
      endDate: req.query.endDate,
      assetName: req.query.assetName,
      farmName: req.query.farmName,
      minAnomalies: parseOptionalNumber(req.query.minAnomalies),
      maxAnomalies: parseOptionalNumber(req.query.maxAnomalies),
      minCritical: parseOptionalNumber(req.query.minCritical),
      maxCritical: parseOptionalNumber(req.query.maxCritical)
    };

    const jobs = await listJobs({ filters, session: req.session });

    const refreshedJobs = [];
    for (const job of jobs) {
      try {
        const synced = await synchroniseJob(job, req.session);
        refreshedJobs.push(sanitiseJobDocument(synced ?? job));
      } catch (syncError) {
        if (syncError.status && syncError.status >= 400 && syncError.status < 500) {
          refreshedJobs.push(sanitiseJobDocument(job));
        } else {
          throw syncError;
        }
      }
    }

    res.json({ jobs: refreshedJobs });
  } catch (error) {
    next(error);
  }
});

router.get('/:jobId', async (req, res, next) => {
  try {
    const jobsCollection = getJobsCollection();
    const ownershipQuery = buildOwnershipQuery(req.session);
    const job = await jobsCollection.findOne({
      jobId: req.params.jobId,
      ...ownershipQuery
    });
    if (!job) {
      return res.status(404).json({ error: 'Prediction job not found.' });
    }

    let syncedJob = job;
    try {
      syncedJob = await synchroniseJob(job, req.session);
    } catch (syncError) {
      if (!syncError.status || syncError.status >= 500) {
        throw syncError;
      }
    }

    const sessionJob = await jobsCollection.findOne({ _id: syncedJob._id });
    const latest = sessionJob || syncedJob;

    res.json({
      job: sanitiseJobDocument(latest),
      status: latest.lastSnapshot || null
    });
  } catch (error) {
    next(error);
  }
});

export default router;
