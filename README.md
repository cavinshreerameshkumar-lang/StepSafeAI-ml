# StepSafeAI — Heat Stroke Prediction Platform

A production-ready React + Vite frontend for heat stroke risk prediction.

## 🚀 Quick Start

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

## 📦 Build for Production

```bash
npm run build
npm run preview
```

## 🗂 Project Structure

```
src/
├── components/
│   ├── layout/
│   │   ├── Navbar.jsx / .module.css
│   │   └── Footer.jsx / .module.css
│   ├── sections/               # Landing page scrollable sections
│   │   ├── HeroSection.jsx / .module.css
│   │   ├── UVSection.jsx / .module.css
│   │   ├── AQISection.jsx / .module.css
│   │   └── DehydrationSection.jsx / .module.css
│   └── tabs/                   # Dashboard tab content
│       ├── ChecklistTab.jsx / .module.css
│       ├── RisksTab.jsx / .module.css
│       ├── SafetyScoreTab.jsx / .module.css
│       └── AuthTab.jsx / .module.css
├── pages/
│   ├── HomePage.jsx
│   ├── DashboardPage.jsx / .module.css
│   └── AboutPage.jsx / .module.css
├── hooks/
│   └── useScrollReveal.js      # IntersectionObserver hook
├── data/
│   └── index.js                # Static data (UV, AQI, dehydration)
├── styles/
│   └── globals.css             # CSS variables + utility classes
├── App.jsx                     # Router + layout
└── main.jsx                    # Entry point
```

## 🔌 Connecting Real Data

Replace mock values in `src/data/index.js` and add API calls in each section component.

Suggested APIs:
- **UV Index**: OpenUV API, EPA / NOAA
- **AQI**: AirVisual (IQAir), OpenAQ
- **Weather / Heat Index**: OpenWeatherMap, WeatherAPI

## 🚢 Deployment

Deploy the `dist/` folder to:
- **Vercel**: `vercel --prod`
- **Netlify**: drag & drop `dist/`
- **GitHub Pages**: configure `vite.config.js` base path

> Note: React Router uses client-side routing. Configure your host to redirect all routes to `index.html`.
