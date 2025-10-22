import { useEffect, useRef, useState } from 'react';
import { fetchJobStatus } from '../services/api.js';

const DEFAULT_INTERVAL = 5000;

export const usePredictionJob = (jobId, { onUpdate, interval = DEFAULT_INTERVAL } = {}) => {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setError(null);
      return undefined;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const response = await fetchJobStatus(jobId);
        if (cancelled) {
          return;
        }
        const status = response?.status ?? null;
        setJob(status);
        setError(null);
        if (status) {
          onUpdate?.(status, response?.job ?? null);
        }
        if (status?.status === 'completed' || status?.status === 'failed') {
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
  }, [jobId, interval, onUpdate]);

  return { job, error };
};
