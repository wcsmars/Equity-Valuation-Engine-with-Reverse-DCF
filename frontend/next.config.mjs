/** @type {import('next').NextConfig} */
const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";

const nextConfig = {
  experimental: {
    // The rewrite proxy below times out after 30s by default. AI digests,
    // research notes and cold valuations can take minutes, so allow up to
    // 10 minutes (the Anthropic SDK's own default). Must be > 0.
    proxyTimeout: 600_000,
    // Proxied request bodies are cut off at 10MB by default. PDFs are sent
    // base64-encoded inside the JSON body, so raise the cap to the backend's
    // 50 MB /api/ai/* limit. Past it the proxy drops the connection instead of
    // answering 413, so lib/api.ts checks request sizes before sending.
    middlewareClientMaxBodySize: "50mb",
  },
  // Same-origin proxy: the browser calls /api/*, Next forwards to FastAPI.
  // Avoids CORS and keeps the API base configurable for deploys.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
