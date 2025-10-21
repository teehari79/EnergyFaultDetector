/* global Buffer */

const getSubtleCrypto = () => {
  if (globalThis.crypto?.subtle) {
    return globalThis.crypto.subtle;
  }
  if (globalThis.crypto?.webcrypto?.subtle) {
    return globalThis.crypto.webcrypto.subtle;
  }
  throw new Error('Web Crypto API is not available in this environment.');
};

const subtle = getSubtleCrypto();
const textEncoder = new TextEncoder();

const sha256Bytes = async (value) => {
  const encoded = textEncoder.encode(value);
  const digest = await subtle.digest('SHA-256', encoded);
  return new Uint8Array(digest);
};

const bytesToHex = (bytes) =>
  Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');

const xorBytes = (data, key) => {
  const output = new Uint8Array(data.length);
  for (let index = 0; index < data.length; index += 1) {
    output[index] = data[index] ^ key[index % key.length];
  }
  return output;
};

const bytesToBase64 = (bytes) => {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes).toString('base64');
  }
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
};

const deriveKey = async (seedToken, context) => sha256Bytes(`${seedToken}:${context}`);

export const encryptPayload = async (seedToken, content, context) => {
  const raw = textEncoder.encode(JSON.stringify(content));
  const key = await deriveKey(seedToken, context);
  const encrypted = xorBytes(raw, key);
  return bytesToBase64(encrypted);
};

export const hashAuthToken = async (authToken, seedToken) => {
  const digest = await sha256Bytes(`${authToken}:${seedToken}`);
  return bytesToHex(digest);
};

export const AUTH_CONTEXT = 'auth_credentials';
