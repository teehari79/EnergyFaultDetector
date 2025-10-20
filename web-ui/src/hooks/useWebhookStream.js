import { useEffect, useMemo, useRef, useState } from 'react';
import dayjs from 'dayjs';
import { API_BASE_URL, WEBHOOK_ENDPOINTS } from '../config.js';

const buildUrl = (path, predictionId) => {
  const base = `${API_BASE_URL}${path}`;
  const url = new URL(base);
  if (predictionId) {
    url.searchParams.set('prediction_id', predictionId);
  }
  return url.toString();
};

const createInitialStatus = () =>
  WEBHOOK_ENDPOINTS.reduce((acc, hook) => {
    acc[hook.key] = { state: 'waiting', updatedAt: null };
    return acc;
  }, {});

export const useWebhookStream = (predictionId, { onEvent } = {}) => {
  const [status, setStatus] = useState(createInitialStatus);
  const eventBufferRef = useRef({ anomalies: [], critical: [], rca: [] });

  useEffect(() => {
    if (!predictionId || typeof window === 'undefined') {
      return undefined;
    }

    const controllers = [];
    const nextStatus = createInitialStatus();

    WEBHOOK_ENDPOINTS.forEach((hook) => {
      const endpoint = buildUrl(hook.path, predictionId);
      try {
        const source = new EventSource(endpoint, { withCredentials: true });
        source.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            eventBufferRef.current[hook.key] = [
              ...(eventBufferRef.current[hook.key] || []),
              payload
            ];
            nextStatus[hook.key] = {
              state: 'ready',
              updatedAt: dayjs().toISOString()
            };
            setStatus({ ...nextStatus });
            onEvent?.(hook, payload, {
              allEvents: { ...eventBufferRef.current }
            });
          } catch (error) {
            console.error('Failed to parse webhook event', error);
            nextStatus[hook.key] = {
              state: 'error',
              updatedAt: dayjs().toISOString(),
              error
            };
            setStatus({ ...nextStatus });
          }
        };
        source.onerror = () => {
          nextStatus[hook.key] = {
            state: 'error',
            updatedAt: dayjs().toISOString()
          };
          setStatus({ ...nextStatus });
        };

        controllers.push(() => source.close());
      } catch (error) {
        console.warn('Falling back to polling for', hook.path, error);
        let stopped = false;
        const poll = async () => {
          if (stopped) return;
          try {
            const response = await fetch(endpoint);
            if (!response.ok) {
              throw new Error(`Failed to poll ${hook.label}`);
            }
            const payload = await response.json();
            if (Array.isArray(payload) ? payload.length : payload) {
              eventBufferRef.current[hook.key] = Array.isArray(payload)
                ? payload
                : [payload];
              nextStatus[hook.key] = {
                state: 'ready',
                updatedAt: dayjs().toISOString()
              };
              setStatus({ ...nextStatus });
              onEvent?.(hook, eventBufferRef.current[hook.key], {
                allEvents: { ...eventBufferRef.current }
              });
            }
          } catch (err) {
            nextStatus[hook.key] = {
              state: 'error',
              updatedAt: dayjs().toISOString(),
              error: err
            };
            setStatus({ ...nextStatus });
          }
          setTimeout(poll, 5000);
        };
        poll();
        controllers.push(() => {
          stopped = true;
        });
      }
    });

    return () => {
      controllers.forEach((dispose) => dispose());
      setStatus(createInitialStatus());
      eventBufferRef.current = { anomalies: [], critical: [], rca: [] };
    };
  }, [predictionId, onEvent]);

  const hasErrors = useMemo(
    () => Object.values(status).some((entry) => entry.state === 'error'),
    [status]
  );

  return {
    status,
    hasErrors
  };
};
