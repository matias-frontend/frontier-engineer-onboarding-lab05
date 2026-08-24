FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src ./src
COPY scripts ./scripts

# Typecheck at build time — tsx transpiles without checking types, so a type
# error would otherwise ship and only surface at runtime.
RUN npx tsc --noEmit

ENV NODE_ENV=production
ENV PORT=8000
ENV HOST=0.0.0.0
EXPOSE 8000

CMD ["npx", "tsx", "src/server.ts"]
