
// env.js is loaded (if present) before index.js, and is silently skipped (harmless 404) if you haven't created it.

window.ENV = {
    //must match Backend API PORT from backend .env or cloud deployment URL

    API_BASE: 'http://localhost:xxxx' || 'https://Render-or-Vercel.app',
};