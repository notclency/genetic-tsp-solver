# 🧬 Genetic TSP Solver (Traveling Salesman Problem)

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Algorithms](https://img.shields.io/badge/Algorithm-Evolutionary-emerald?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

An evolutionary algorithm visualization that solves the **Traveling Salesman Problem (TSP)**. It uses biological concepts like Natural Selection, Crossover, and Mutation to find the shortest possible route between a set of cities, visiting each city exactly once and returning to the start.

**[🌐 View Live Simulation](https://clencytabe.vercel.app/projects/genetic-tsp)**

---

## 🚀 Features

### 1. Interactive Visualization
- **Real-Time Evolution:** Watch the algorithm improve the path generation by generation.
- **Dynamic Complexity:** Switch between 10, 50, or 250 cities to see how the algorithm scales.
- **Performance Metrics:** Tracks Generation count, Best Distance found, and Execution time live.

### 2. Core Genetic Logic
The solver implements a standard Genetic Algorithm (GA) pipeline:
- **Selection:** "Fittest" paths (shortest distance) are selected to breed the next generation (Elitism).
- **Crossover (OX1):** Combines two parent paths to create a child path while maintaining a valid tour (no duplicate cities).
- **Mutation:** Randomly swaps two cities in a path to introduce variation and prevent getting stuck in local optima.

---

## 🧠 Algorithm Breakdown

### The Challenge
Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin? This is an NP-Hard problem, meaning brute-forcing it becomes impossible as the number of cities grows ($N!$ complexity).

### The Solution (Evolutionary Approach)

1. **Initialization:**
   - Generate a population of 50 random paths.
   
2. **Evaluation (Fitness Function):**
   - Calculate the total distance of each path.
   - Shorter distance = Higher fitness.

3. **Breeding (Crossover - Order 1):**
   - Take a sub-slice of cities from **Parent A**.
   - Fill the remaining empty spots with cities from **Parent B** in the order they appear.
   - *Why?* Standard crossover would create invalid paths with duplicate cities. OX1 ensures validity.

4. **Mutation:**
   - With a 15% probability, swap two random cities in a child's path.
   - This prevents the population from becoming identical too quickly.

---

## 🛠️ Tech Stack

### Frontend (Visualization)
- **Framework:** React + Vite
- **Rendering:** HTML5 Canvas API (for high-performance drawing of 250+ nodes)
- **Styling:** Tailwind CSS
- **Logic:** TypeScript (Custom Genetic Engine implementation)

### Backend / Core Logic (Python Version)
- **Library:** PyGAD (Genetic Algorithm library)
- **Optimization:** NumPy for vectorized distance calculations

---

## 📸 Screenshots

| 50 Cities (Initial) | 50 Cities (Optimized) |
|:---:|:---:|
| <img src="https://via.placeholder.com/400x200?text=Random+Chaos" alt="Initial State" width="400"/> | <img src="https://via.placeholder.com/400x200?text=Optimized+Path" alt="Optimized State" width="400"/> |

---

## 💻 Running Locally

### Prerequisites
- Node.js (v16+)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/genetic-tsp-solver.git
