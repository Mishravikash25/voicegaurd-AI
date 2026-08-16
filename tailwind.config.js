/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          DEFAULT: '#020617', // Deep Navy
          darker: '#000000',  // Black
          accent: '#1e1b4b',  // Dark Purple
        },
        foreground: '#f8fafc',
        neon: {
          indigo: '#6366F1',
          green: '#22C55E',
          red: '#EF4444',
        },
        surface: 'rgba(15, 23, 42, 0.6)',
        border: 'rgba(255, 255, 255, 0.1)',
      },
      borderRadius: {
        '2xl': '1rem',
      },
      boxShadow: {
        'neon-indigo': '0 0 20px rgba(99, 102, 241, 0.4)',
        'neon-green': '0 0 20px rgba(34, 197, 94, 0.4)',
        'neon-red': '0 0 20px rgba(239, 68, 68, 0.4)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      },
      dropShadow: {
        'neon-indigo': '0 0 10px rgba(99, 102, 241, 0.6)',
        'neon-green': '0 0 10px rgba(34, 197, 94, 0.6)',
        'neon-red': '0 0 10px rgba(239, 68, 68, 0.6)',
      },
      animation: {
        'background-animate': 'gradient 15s ease infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        gradient: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      fontFamily: {
        sans: ['Outfit', 'Inter', 'sans-serif'],
        display: ['Outfit', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
