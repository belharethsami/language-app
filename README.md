# Language Learning App

A web application for language learning with text-to-speech and translation capabilities.

## Project Structure

```
.
├── backend/             # FastAPI backend
│   ├── main.py         # Main API code
│   ├── requirements.txt # Python dependencies
│   └── .env            # Environment variables (not in git)
└── frontend/           # Static frontend
    ├── index.html      # Main frontend code
    └── server.py       # Development server
```

## Local Development

1. Install dependencies:
```bash
# Install backend dependencies
npm run install:all

# Install frontend development dependencies
npm install
```

2. Set up environment variables:
```bash
# Copy example .env and fill in your values
cp backend/.env.example backend/.env
```

3. Run the development servers:
```bash
# Run both frontend and backend
npm run dev

# Or run them separately:
npm run frontend:dev
npm run backend:dev
```

## Deployment

### Backend (Render.com)

1. Create a new Web Service on Render.com
2. Connect your GitHub repository
3. Configure:
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && gunicorn main:app -k uvicorn.workers.UvicornWorker`
4. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `FRONTEND_URL`: Your GitHub Pages URL

### Frontend (GitHub Pages)

1. Update the API URL in frontend/index.html to point to your Render.com backend
2. Deploy to GitHub Pages:
```bash
npm run deploy:frontend
```

## Development Notes

- The frontend is a simple static site using HTML, CSS, and JavaScript
- The backend uses FastAPI and integrates with OpenAI's API
- Environment variables are used for configuration
- CORS is configured to allow GitHub Pages domains
