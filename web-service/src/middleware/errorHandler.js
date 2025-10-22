export function errorHandler(err, req, res, _next) {
  const status = err.status || err.statusCode || 500;
  const payload = {
    error: err.message || 'Internal server error'
  };
  if (process.env.NODE_ENV !== 'production' && err.stack) {
    payload.details = err.details || undefined;
  }
  if (status >= 500) {
    console.error('Unhandled error:', err);
  }
  res.status(status).json(payload);
}
