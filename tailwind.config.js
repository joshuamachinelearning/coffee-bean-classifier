/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './templates/**/*.html',   // scan all HTML files in templates
    './static/js/**/*.js'      // scan JS if you have dynamic classes
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
