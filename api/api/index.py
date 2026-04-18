from backend.main import app

# Explicitly export for Vercel's ASGI handler detection
handler = app
