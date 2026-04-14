/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#1F2937",
        paper: "#FFFFFF",
        mist: "#F9FAFB",
        ether: "#6D28D9",
        lavender: "#8B5CF6",
      },
    },
  },
  plugins: [],
}

