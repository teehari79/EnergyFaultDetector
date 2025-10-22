import { getSessionsCollection } from '../db.js';

export async function authenticateRequest(req, res, next) {
  try {
    const token = req.header('x-auth-token');
    if (!token) {
      return res.status(401).json({ error: 'Authentication token is required.' });
    }

    const sessions = getSessionsCollection();
    const session = await sessions.findOne({ authToken: token.trim() });
    if (!session) {
      return res.status(401).json({ error: 'Session expired. Please authenticate again.' });
    }

    if (session.expiresAt && new Date(session.expiresAt) <= new Date()) {
      await sessions.deleteOne({ _id: session._id });
      return res.status(401).json({ error: 'Session expired. Please authenticate again.' });
    }

    await sessions.updateOne(
      { _id: session._id },
      { $set: { lastSeenAt: new Date() } }
    );

    req.session = session;
    return next();
  } catch (error) {
    return next(error);
  }
}
