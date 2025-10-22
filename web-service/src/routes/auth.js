import express from 'express';
import { authenticate } from '../services/predictionService.js';
import { getSessionsCollection } from '../db.js';

const router = express.Router();

router.post('/login', async (req, res, next) => {
  try {
    const { organizationId, username, password, seedToken } = req.body || {};
    if (!organizationId || !username || !password || !seedToken) {
      return res
        .status(400)
        .json({ error: 'organizationId, username, password, and seedToken are required.' });
    }

    const response = await authenticate({ organizationId, username, password, seedToken });
    const expiresAt = response.expires_at ? new Date(response.expires_at) : null;

    const sessions = getSessionsCollection();
    await sessions.updateOne(
      { authToken: response.auth_token },
      {
        $set: {
          authToken: response.auth_token,
          organizationId,
          username,
          seedToken,
          expiresAt,
          createdAt: new Date(),
          lastSeenAt: new Date()
        }
      },
      { upsert: true }
    );

    res.json({
      authToken: response.auth_token,
      expiresAt: expiresAt ? expiresAt.toISOString() : null,
      user: {
        organizationId,
        username
      }
    });
  } catch (error) {
    if (!error.status) {
      error.status = 401;
    }
    next(error);
  }
});

export default router;
