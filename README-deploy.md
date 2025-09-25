## Backend deployment (Render)

- Root directory: `backend`
- Build command: `npm ci`
- Start command: `npm start`
- Health check: `GET /api/health`
- Required env vars:
  - `OPENAI_API_KEY`
  - `CORS_ORIGIN` (e.g., `http://localhost:3000` or your deployed frontend origin)
  - Optional: `ML_INFER_URL`, `RATE_LIMIT_WINDOW_MS`, `RATE_LIMIT_MAX_REQUESTS`

Notes:
- Render provides `PORT` automatically; `backend/server.js` reads it.
- File uploads are handled in-memory; max 10MB per request.

## Frontend configuration

- Create a file `.env` in the repo root with:

```
REACT_APP_API_URL=https://your-backend.onrender.com/api
```

- Then install and run locally:

```
npm ci
npm start
```

## Local development

- Backend: from `backend/`
```
npm ci
npm run dev
```
- Frontend: from repo root
```
npm ci
npm start
```

## What is ignored from git
- Dependencies: `node_modules/`, `backend/node_modules/`
- Python venv: `backend/ml/venv/`
- Large datasets and models under `backend/ml/data/*` and `backend/ml/models/`
- Any `.env` files 