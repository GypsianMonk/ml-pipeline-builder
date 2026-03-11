# ⬡ ML Pipeline Builder

> A step-by-step interactive ML pipeline builder for developers and data scientists.
> Build scikit-learn pipelines visually — export clean, production-ready Python code instantly.

![ML Pipeline Builder](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat-square&logo=vite)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## ✨ Features

- **7-step guided pipeline** — walk through every stage of an ML workflow
- **3 task types** — Classification, Regression, and Clustering
- **Smart suggestions** — preprocessing steps, models, and metrics adapt to your chosen task
- **Live code generation** — Python/sklearn code updates in real-time as you configure
- **One-click copy** — grab the generated code instantly
- **Dark, dev-focused UI** — built for data scientists, not marketers

---

## 🧠 Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Task | Choose Classification, Regression, or Clustering |
| 2. Data | Select your data source (CSV, JSON, SQL, API, sklearn) |
| 3. Preprocessing | Pick transformers: scalers, encoders, imputers, PCA, etc. |
| 4. Model | Choose your algorithm with tagged categories |
| 5. Training | Configure test split, CV folds, and random state |
| 6. Evaluation | Select task-appropriate metrics |
| 7. Export | Save as Pickle, Joblib, ONNX, MLflow, and more |

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v18+
- npm or yarn

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ml-pipeline-builder.git
cd ml-pipeline-builder

# Install dependencies
npm install

# Start the dev server
npm run dev
```

Visit `http://localhost:5173` in your browser.

### Build for Production

```bash
npm run build
# Output is in the /dist folder
```

---

## ☁️ Deploy

### Deploy to Vercel (recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

Or connect your GitHub repo to [vercel.com](https://vercel.com) for automatic deploys on every push.

### Deploy to Netlify

```bash
# Install Netlify CLI
npm i -g netlify-cli

# Build & deploy
npm run build
netlify deploy --prod --dir=dist
```

---

## 🛠️ Tech Stack

- **React 18** — UI framework
- **Vite 5** — build tool & dev server
- **scikit-learn** — the generated code targets the sklearn API
- **Google Fonts** — DM Mono + Space Grotesk

---

## 📦 Generated Code Example

The builder outputs clean, commented Python:

```python
# ML Pipeline — Classification
# Model: RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"accuracy_score: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 📁 Project Structure

```
ml-pipeline-builder/
├── public/
│   └── favicon.svg
├── src/
│   ├── App.jsx        # Main pipeline builder component
│   └── main.jsx       # React entry point
├── index.html
├── vite.config.js
├── vercel.json
├── package.json
└── README.md
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT — feel free to use, modify, and distribute.

---

*Built with ⬡ ML Pipeline Builder*
