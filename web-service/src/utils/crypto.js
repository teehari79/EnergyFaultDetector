import crypto from 'node:crypto';

const AUTH_CONTEXT = 'auth_credentials';

function sha256Bytes(value) {
  return crypto.createHash('sha256').update(value).digest();
}

function xorBuffers(data, key) {
  const output = Buffer.allocUnsafe(data.length);
  for (let index = 0; index < data.length; index += 1) {
    output[index] = data[index] ^ key[index % key.length];
  }
  return output;
}

export function encryptPayload(seedToken, content, context) {
  const raw = Buffer.from(JSON.stringify(content), 'utf8');
  const key = sha256Bytes(`${seedToken}:${context}`);
  const encrypted = xorBuffers(raw, key);
  return encrypted.toString('base64');
}

export function hashAuthToken(authToken, seedToken) {
  return sha256Bytes(`${authToken}:${seedToken}`).toString('hex');
}

export { AUTH_CONTEXT };
