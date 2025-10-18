<h1 align="center">🌍 Orbit Visualizer 🌌</h1>

<p align="center">
  <strong>A beginner-friendly Python project that simulates 2D Keplerian orbits of planets around the Sun with real elliptical geometry, beautiful plots, and Excel + GIF exports.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/Rishi943/orbit-visualizer/stargazers"><img src="https://img.shields.io/github/stars/Rishi943/orbit-visualizer?color=yellow" alt="GitHub Stars"></a>
</p>

---

## 🪐 Overview

This project simulates **elliptical orbits** for multiple bodies around the Sun using **Kepler’s equation**, rendered with `matplotlib`.  
It’s fully offline, non-interactive (Agg backend), and exports plots, animation (optional GIF), and tidy ephemeris data in Excel.

---

## ⚙️ Features

- Simulates orbits for Mercury → Saturn and a high-eccentricity comet ☄️  
- Solves Kepler’s equation with Newton–Raphson method  
- Produces two static plots (full system + inner planets)  
- Black background with colored bodies (yellow Sun, blue/green Earth, white comet)  
- Optional animated GIF using `imageio`  
- Exports orbital data to Excel (`orbit_ephemeris.xlsx`)

---

## 🧩 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Rishi943/orbit-visualizer.git
cd orbit-visualizer
