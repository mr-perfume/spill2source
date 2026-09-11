import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend only ever talks to its own origin. Vite proxies /api and the
// socket handshake to the gateway, which keeps CORS out of the picture during
// development and means the deployed and dev URLs are identical.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:4000", changeOrigin: true },
      "/socket.io": { target: "http://127.0.0.1:4000", ws: true, changeOrigin: true },
    },
  },
});
