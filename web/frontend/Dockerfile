# Stage 1: React uygulamasını build et
FROM node:18-alpine AS builder

WORKDIR /app
COPY web/frontend/package*.json ./
RUN npm ci
COPY web/frontend/ ./
RUN npm run build

# Stage 2: Nginx ile static dosyaları sun
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY web/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
