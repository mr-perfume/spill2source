/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        abyss: "#04121C",
        hull: "#0A1F2E",
        rule: "#153B4E",
        haze: "#7FA6B8",
        chart: "#CFE6F0",
        slick: "#FFA43D",
        posterior: "#35E0D2",
        origin: "#F2FBFF",
        alert: "#FF5A5F",
        cleared: "#4E7A8C",
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
