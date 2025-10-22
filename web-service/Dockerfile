# syntax=docker/dockerfile:1.5

FROM node:20-alpine AS base

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY . .

ENV NODE_ENV=production
EXPOSE 4000

CMD ["node", "src/index.js"]
