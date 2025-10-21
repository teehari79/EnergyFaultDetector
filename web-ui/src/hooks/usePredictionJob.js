import { useEffect, useRef, useState } from 'react';
import { fetchJobStatus } from '../services/api.js';

const DEFAULT_INTERVAL = 5000;

export const usePredictionJob = (jobId, authToken, { onUpdate, interval = DEFAULT_INTERVAL } = {}) => {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!jobId || !authToken) {
      setJob(null);
      setError(null);
      return undefined;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const status = await fetchJobStatus(jobId, authToken);
        if (cancelled) {
          return;
        }
        setJob(status);
        setError(null);
        onUpdate?.(status);
        if (status.status === 'completed' || status.status === 'failed') {
          return;
        }
      } catch (err) {
        if (cancelled) {
          return;
        }
        setError(err);
      }
      timerRef.current = window.setTimeout(poll, interval);
    };

    poll();

    return () => {
      cancelled = true;
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, [jobId, authToken, interval, onUpdate]);

  return { job, error };
};
