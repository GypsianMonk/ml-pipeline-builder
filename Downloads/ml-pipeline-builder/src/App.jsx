import { useState, useEffect, useRef } from "react";

const STEPS = ["Task", "Data", "Preprocess", "Model", "Training", "Evaluate", "Export"];

const TASK_OPTIONS = [
  { id: "classification", label: "Classification", icon: "◈", desc: "Predict discrete categories", color: "#00f5ff", glow: "#00f5ff44" },
  { id: "regression", label: "Regression", icon: "◉", desc: "Predict continuous values", color: "#ff6b35", glow: "#ff6b3544" },
  { id: "clustering", label: "Clustering", icon: "◎", desc: "Discover hidden groupings", color: "#a855f7", glow: "#a855f744" },
];

const DATA_SOURCES = [
  { id: "csv", label: "CSV / Excel", icon: "⬡", desc: "Local file upload" },
  { id: "json", label: "JSON", icon: "⬢", desc: "Structured data format" },
  { id: "sql", label: "SQL Database", icon: "◈", desc: "Query directly" },
  { id: "api", label: "API Endpoint", icon: "⟳", desc: "Live data stream" },
  { id: "sklearn", label: "Sklearn Dataset", icon: "◉", desc: "Built-in datasets" },
];

const PREPROCESSING = {
  classification: ["StandardScaler","MinMaxScaler","LabelEncoder","OneHotEncoder","SimpleImputer (mean)","SimpleImputer (median)","SMOTE","PCA","SelectKBest"],
  regression: ["StandardScaler","MinMaxScaler","PolynomialFeatures","SimpleImputer (mean)","SimpleImputer (median)","Log Transform","PCA","SelectKBest"],
  clustering: ["StandardScaler","MinMaxScaler","PCA","UMAP","SimpleImputer (mean)","Normalizer"],
};

const MODELS = {
  classification: [
    { id: "LogisticRegression", label: "Logistic Regression", tag: "Linear", color: "#00f5ff" },
    { id: "RandomForestClassifier", label: "Random Forest", tag: "Ensemble", color: "#10b981" },
    { id: "GradientBoostingClassifier", label: "Gradient Boosting", tag: "Ensemble", color: "#10b981" },
    { id: "XGBClassifier", label: "XGBoost", tag: "Boosting", color: "#f59e0b" },
    { id: "SVC", label: "Support Vector Machine", tag: "Kernel", color: "#a855f7" },
    { id: "KNeighborsClassifier", label: "K-Nearest Neighbors", tag: "Instance", color: "#ff6b35" },
    { id: "DecisionTreeClassifier", label: "Decision Tree", tag: "Tree", color: "#06b6d4" },
    { id: "MLPClassifier", label: "Neural Network (MLP)", tag: "Neural", color: "#f43f5e" },
  ],
  regression: [
    { id: "LinearRegression", label: "Linear Regression", tag: "Linear", color: "#00f5ff" },
    { id: "Ridge", label: "Ridge Regression", tag: "Regularized", color: "#06b6d4" },
    { id: "Lasso", label: "Lasso Regression", tag: "Regularized", color: "#06b6d4" },
    { id: "RandomForestRegressor", label: "Random Forest", tag: "Ensemble", color: "#10b981" },
    { id: "GradientBoostingRegressor", label: "Gradient Boosting", tag: "Ensemble", color: "#10b981" },
    { id: "XGBRegressor", label: "XGBoost", tag: "Boosting", color: "#f59e0b" },
    { id: "SVR", label: "Support Vector Regressor", tag: "Kernel", color: "#a855f7" },
    { id: "MLPRegressor", label: "Neural Network (MLP)", tag: "Neural", color: "#f43f5e" },
  ],
  clustering: [
    { id: "KMeans", label: "K-Means", tag: "Centroid", color: "#f59e0b" },
    { id: "DBSCAN", label: "DBSCAN", tag: "Density", color: "#10b981" },
    { id: "AgglomerativeClustering", label: "Agglomerative", tag: "Hierarchical", color: "#a855f7" },
    { id: "GaussianMixture", label: "Gaussian Mixture", tag: "Probabilistic", color: "#ff6b35" },
    { id: "SpectralClustering", label: "Spectral Clustering", tag: "Graph", color: "#f43f5e" },
    { id: "MeanShift", label: "Mean Shift", tag: "Density", color: "#10b981" },
  ],
};

const METRICS = {
  classification: ["accuracy_score","f1_score","precision_score","recall_score","roc_auc_score","confusion_matrix","classification_report"],
  regression: ["mean_squared_error","mean_absolute_error","r2_score","root_mean_squared_error","mean_absolute_percentage_error"],
  clustering: ["silhouette_score","davies_bouldin_score","calinski_harabasz_score","adjusted_rand_score"],
};

const EXPORT_FORMATS = [
  { id: "joblib", label: "Joblib", ext: ".joblib", icon: "⬡" },
  { id: "pickle", label: "Pickle", ext: ".pkl", icon: "⬢" },
  { id: "onnx", label: "ONNX", ext: ".onnx", icon: "◈" },
  { id: "mlflow", label: "MLflow", ext: "", icon: "◉" },
  { id: "pmml", label: "PMML", ext: ".pmml", icon: "◎" },
  { id: "script", label: "Python Script", ext: ".py", icon: "{}" },
];

function generateCode(pipeline) {
  const { task, dataSource, preprocessing, model, training, evaluation, exportFormat } = pipeline;
  if (!task || !model) return "# ⬡ Complete the pipeline to generate code\n# Select a task and model to get started";
  const isClustering = task === "clustering";
  const testSplit = training.testSplit || 0.2;
  const cvFolds = training.cvFolds || 5;
  const randomState = training.randomState || 42;
  const preSteps = (preprocessing || []).map((p) => {
    const stepMap = {
      "StandardScaler":`    ('scaler', StandardScaler())`,
      "MinMaxScaler":`    ('minmax', MinMaxScaler())`,
      "LabelEncoder":`    ('le', LabelEncoder())`,
      "OneHotEncoder":`    ('ohe', OneHotEncoder(handle_unknown='ignore'))`,
      "SimpleImputer (mean)":`    ('imputer', SimpleImputer(strategy='mean'))`,
      "SimpleImputer (median)":`    ('imputer', SimpleImputer(strategy='median'))`,
      "SMOTE":`    ('smote', SMOTE(random_state=${randomState}))`,
      "PCA":`    ('pca', PCA(n_components=0.95))`,
      "SelectKBest":`    ('selector', SelectKBest(k=10))`,
      "PolynomialFeatures":`    ('poly', PolynomialFeatures(degree=2))`,
      "Log Transform":`    ('log', FunctionTransformer(np.log1p))`,
      "Normalizer":`    ('normalizer', Normalizer())`,
      "UMAP":`    ('umap', UMAP(n_components=2))`,
    };
    return stepMap[p] || `    ('step', ${p}())`;
  });
  const modelLine = (() => {
    if (model==="XGBClassifier"||model==="XGBRegressor") return `    ('model', ${model}(use_label_encoder=False, eval_metric='logloss', random_state=${randomState}))`;
    if (model==="DBSCAN") return `    ('model', DBSCAN(eps=0.5, min_samples=5))`;
    if (model==="KMeans") return `    ('model', KMeans(n_clusters=3, random_state=${randomState}))`;
    if (model==="GaussianMixture") return `    ('model', GaussianMixture(n_components=3, random_state=${randomState}))`;
    if (model==="AgglomerativeClustering") return `    ('model', AgglomerativeClustering(n_clusters=3))`;
    if (model==="SpectralClustering") return `    ('model', SpectralClustering(n_clusters=3, random_state=${randomState}))`;
    if (model==="MeanShift") return `    ('model', MeanShift())`;
    if (model.includes("Forest")||model.includes("Boosting")) return `    ('model', ${model}(n_estimators=100, random_state=${randomState}))`;
    if (model==="SVC"||model==="SVR") return `    ('model', ${model}(kernel='rbf', C=1.0))`;
    if (model.includes("MLP")) return `    ('model', ${model}(hidden_layer_sizes=(100,50), max_iter=200, random_state=${randomState}))`;
    return `    ('model', ${model}(random_state=${randomState}))`;
  })();
  const allSteps = [...preSteps, modelLine].join(",\n");
  const metricsCode = (evaluation||[]).map(m => {
    if (m==="confusion_matrix") return `print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))`;
    if (m==="classification_report") return `print("Classification Report:\\n", classification_report(y_test, y_pred))`;
    if (m==="root_mean_squared_error") return `print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")`;
    if (["silhouette_score","davies_bouldin_score","calinski_harabasz_score"].includes(m)) return `print(f"${m}: {${m}(X, labels):.4f}")`;
    if (m==="adjusted_rand_score") return `print(f"Adjusted Rand: {adjusted_rand_score(y_true, labels):.4f}")`;
    return `print(f"${m}: {${m}(y_test, y_pred):.4f}")`;
  }).join("\n");
  const exportCode = (() => {
    if (exportFormat==="pickle") return `import pickle\nwith open('model.pkl', 'wb') as f:\n    pickle.dump(pipeline, f)`;
    if (exportFormat==="mlflow") return `import mlflow\nwith mlflow.start_run():\n    mlflow.sklearn.log_model(pipeline, "model")`;
    return `import joblib\njoblib.dump(pipeline, 'model.joblib')`;
  })();
  return `# ╔══════════════════════════════════════════════════╗
# ║  ML Pipeline  ·  ${task.charAt(0).toUpperCase()+task.slice(1).padEnd(14)} ·  ${(model||"").padEnd(22)}║
# ╚══════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder, Normalizer, FunctionTransformer)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
${task==="classification"
  ?`from sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.metrics import ${(evaluation||[]).join(", ")||"accuracy_score"}`
  :task==="regression"
  ?`from sklearn.linear_model import LinearRegression, Ridge, Lasso\nfrom sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\nfrom sklearn.svm import SVR\nfrom sklearn.neural_network import MLPRegressor\nfrom sklearn.metrics import ${(evaluation||[]).filter(e=>e!=="root_mean_squared_error").join(", ")||"mean_squared_error, r2_score"}`
  :`from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift\nfrom sklearn.mixture import GaussianMixture\nfrom sklearn.metrics import ${(evaluation||[]).join(", ")||"silhouette_score"}`}

# ── Load Data ─────────────────────────────────────────────
df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1)
${!isClustering?`y = df["target"]

# ── Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=${testSplit}, random_state=${randomState}
)`:""}

# ── Pipeline ──────────────────────────────────────────────
pipeline = Pipeline(steps=[
${allSteps}
])

# ── Train ─────────────────────────────────────────────────
${!isClustering
  ?`pipeline.fit(X_train, y_train)

# ── Cross Validate ────────────────────────────────────────
scores = cross_val_score(pipeline, X_train, y_train, cv=${cvFolds})
print(f"CV: {scores.mean():.4f} ± {scores.std():.4f}")

# ── Predict & Evaluate ────────────────────────────────────
y_pred = pipeline.predict(X_test)
${metricsCode||"# add metrics above"}`
  :`labels = pipeline.fit_predict(X)
print(f"Clusters: {np.unique(labels)}")
${metricsCode||"# add metrics above"}`}

# ── Export ────────────────────────────────────────────────
${exportCode}
`;
}

function ParticleField() {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const setSize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    setSize();
    const particles = Array.from({ length: 55 }, () => ({
      x: Math.random() * canvas.width, y: Math.random() * canvas.height,
      r: Math.random() * 1.2 + 0.3, dx: (Math.random()-.5)*.25, dy: (Math.random()-.5)*.25,
      opacity: Math.random() * 0.4 + 0.1,
    }));
    let raf;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(p => {
        p.x += p.dx; p.y += p.dy;
        if (p.x<0||p.x>canvas.width) p.dx*=-1;
        if (p.y<0||p.y>canvas.height) p.dy*=-1;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
        ctx.fillStyle = `rgba(0,245,255,${p.opacity})`; ctx.fill();
      });
      particles.forEach((a,i) => particles.slice(i+1).forEach(b => {
        const d = Math.hypot(a.x-b.x, a.y-b.y);
        if (d < 110) {
          ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y);
          ctx.strokeStyle = `rgba(0,245,255,${0.05*(1-d/110)})`; ctx.lineWidth=.5; ctx.stroke();
        }
      }));
      raf = requestAnimationFrame(draw);
    };
    draw();
    window.addEventListener("resize", setSize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", setSize); };
  }, []);
  return <canvas ref={canvasRef} style={{ position:"fixed", inset:0, zIndex:0, pointerEvents:"none" }} />;
}

export default function MLPipelineBuilder() {
  const [step, setStep] = useState(0);
  const [pipeline, setPipeline] = useState({
    task: null, dataSource: null, preprocessing: [],
    model: null, training: { testSplit: 0.2, cvFolds: 5, randomState: 42 },
    evaluation: [], exportFormat: null,
  });
  const [activeTab, setActiveTab] = useState("builder");
  const [copied, setCopied] = useState(false);
  const [animKey, setAnimKey] = useState(0);

  const update = (key, val) => setPipeline(p => ({ ...p, [key]: val }));
  const toggleList = (key, val) => setPipeline(p => {
    const arr = p[key]||[];
    return { ...p, [key]: arr.includes(val) ? arr.filter(x=>x!==val) : [...arr, val] };
  });
  const goStep = (s) => { setStep(s); setAnimKey(k=>k+1); };
  const canAdvance = () => {
    if (step===0) return !!pipeline.task;
    if (step===1) return !!pipeline.dataSource;
    if (step===3) return !!pipeline.model;
    return true;
  };
  const isStepDone = (i) => {
    if (i===0) return !!pipeline.task;
    if (i===1) return !!pipeline.dataSource;
    if (i===2) return pipeline.preprocessing.length>0;
    if (i===3) return !!pipeline.model;
    if (i===4) return true;
    if (i===5) return pipeline.evaluation.length>0;
    if (i===6) return !!pipeline.exportFormat;
    return false;
  };
  const completedCount = () => STEPS.filter((_,i)=>isStepDone(i)).length;

  const taskColor = pipeline.task ? TASK_OPTIONS.find(t=>t.id===pipeline.task)?.color : "#00f5ff";
  const modelColor = pipeline.model && pipeline.task ? (MODELS[pipeline.task]||[]).find(m=>m.id===pipeline.model)?.color : null;
  const accent = modelColor || taskColor || "#00f5ff";

  const copyCode = () => {
    navigator.clipboard.writeText(generateCode(pipeline));
    setCopied(true); setTimeout(()=>setCopied(false), 2000);
  };

  return (
    <div style={{ minHeight:"100vh", background:"#020408", color:"#e2e8f0", fontFamily:"'Rajdhani',sans-serif", position:"relative", overflow:"hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:3px;}
        ::-webkit-scrollbar-thumb{background:#00f5ff22;border-radius:2px;}
        .grid-bg{
          background-image:linear-gradient(rgba(0,245,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,255,0.025) 1px,transparent 1px);
          background-size:44px 44px;
        }
        .hov{transition:all .22s ease;cursor:pointer;}
        .hov:hover{transform:translateY(-2px) scale(1.015);}
        .hov-row{transition:all .2s ease;cursor:pointer;}
        .hov-row:hover{transform:translateX(4px);}
        @keyframes fadeSlide{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
        @keyframes scanline{0%{top:-2px;}100%{top:100vh;}}
        @keyframes blink{0%,100%{opacity:1;}50%{opacity:0;}}
        @keyframes glow{0%,100%{opacity:.6;}50%{opacity:1;}}
        .anim{animation:fadeSlide .45s cubic-bezier(.16,1,.3,1) forwards;}
        .pulse{animation:glow 2.5s ease-in-out infinite;}
        input[type=range]{-webkit-appearance:none;height:2px;background:rgba(255,255,255,0.08);border-radius:2px;outline:none;cursor:pointer;width:100%;}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;cursor:pointer;transition:transform .15s;}
        input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.3);}
        pre{white-space:pre;overflow-x:auto;font-family:'Share Tech Mono',monospace !important;}
      `}</style>

      <div className="grid-bg" style={{ position:"fixed", inset:0, zIndex:0 }} />
      <div style={{ position:"fixed", inset:0, zIndex:0, pointerEvents:"none", overflow:"hidden" }}>
        <div style={{ position:"absolute", left:0, right:0, height:"1px", background:"linear-gradient(90deg,transparent,rgba(0,245,255,0.12),transparent)", animation:"scanline 9s linear infinite" }} />
      </div>
      <div style={{ position:"fixed", inset:0, background:`radial-gradient(ellipse 60% 50% at 70% 30%, ${accent}08, transparent)`, zIndex:0, transition:"background 0.5s", pointerEvents:"none" }} />
      <ParticleField />

      <div style={{ position:"relative", zIndex:10, display:"flex", flexDirection:"column", minHeight:"100vh" }}>

        {/* ── HEADER ── */}
        <header style={{ padding:"14px 32px", borderBottom:"1px solid rgba(0,245,255,0.08)", background:"rgba(2,4,8,0.85)", backdropFilter:"blur(24px)", display:"flex", alignItems:"center", justifyContent:"space-between", flexShrink:0 }}>
          <div style={{ display:"flex", alignItems:"center", gap:14 }}>
            <div style={{ width:38, height:38, border:`1.5px solid ${accent}`, borderRadius:7, display:"flex", alignItems:"center", justifyContent:"center", fontSize:18, color:accent, boxShadow:`0 0 14px ${accent}55, inset 0 0 10px ${accent}11`, transition:"all .4s" }}>⬡</div>
            <div>
              <div style={{ fontFamily:"'Orbitron',sans-serif", fontWeight:900, fontSize:13, letterSpacing:".22em", color:"#f1f5f9" }}>ML PIPELINE BUILDER</div>
              <div style={{ fontSize:9, color:`${accent}88`, letterSpacing:".3em", marginTop:2 }}>SKLEARN · PYTHON · v1.0</div>
            </div>
          </div>
          <div style={{ display:"flex", alignItems:"center", gap:28 }}>
            <div style={{ display:"flex", alignItems:"center", gap:10 }}>
              <div style={{ width:100, height:2, background:"rgba(255,255,255,0.06)", borderRadius:2, overflow:"hidden" }}>
                <div style={{ height:"100%", background:`linear-gradient(90deg,${accent},${accent}77)`, borderRadius:2, width:`${(completedCount()/STEPS.length)*100}%`, transition:"width .5s ease", boxShadow:`0 0 6px ${accent}` }} />
              </div>
              <span style={{ fontSize:10, color:accent, fontFamily:"'Share Tech Mono'", letterSpacing:".05em" }}>{completedCount()}/{STEPS.length}</span>
            </div>
            <div style={{ display:"flex", gap:2, background:"rgba(255,255,255,0.025)", padding:3, borderRadius:7, border:"1px solid rgba(255,255,255,0.06)" }}>
              {[["builder","◈ BUILDER"],["code","⟨/⟩ CODE"]].map(([id,lbl])=>(
                <button key={id} onClick={()=>setActiveTab(id)} style={{ padding:"7px 20px", border:"none", borderRadius:5, background:activeTab===id?accent:"transparent", color:activeTab===id?"#020408":"#334155", fontFamily:"'Orbitron',sans-serif", fontSize:9, fontWeight:700, letterSpacing:".12em", boxShadow:activeTab===id?`0 0 14px ${accent}88`:"none", cursor:"pointer", transition:"all .2s" }}>{lbl}</button>
              ))}
            </div>
          </div>
        </header>

        {activeTab==="builder" ? (
          <div style={{ display:"flex", flex:1 }}>

            {/* ── SIDEBAR ── */}
            <aside style={{ width:210, borderRight:"1px solid rgba(0,245,255,0.07)", background:"rgba(2,4,8,0.65)", backdropFilter:"blur(20px)", padding:"32px 0", flexShrink:0, display:"flex", flexDirection:"column" }}>
              <div style={{ padding:"0 18px", flex:1 }}>
                <div style={{ fontSize:8, color:"#1e293b", letterSpacing:".25em", marginBottom:20, fontFamily:"'Orbitron',sans-serif" }}>PIPELINE STEPS</div>
                {STEPS.map((s,i)=>{
                  const done=isStepDone(i), active=step===i;
                  const nodeColor = active ? accent : done ? "#10b981" : "#1e293b";
                  return (
                    <div key={i} style={{ display:"flex", alignItems:"flex-start", gap:0 }}>
                      <div style={{ display:"flex", flexDirection:"column", alignItems:"center" }}>
                        <div onClick={()=>goStep(i)} style={{ width:30, height:30, borderRadius:"50%", border:`1.5px solid ${nodeColor}`, background: active?`${nodeColor}18`:done?`${nodeColor}10`:"transparent", display:"flex", alignItems:"center", justifyContent:"center", boxShadow:active?`0 0 14px ${nodeColor},0 0 28px ${nodeColor}44`:done?`0 0 6px ${nodeColor}55`:"none", cursor:"pointer", fontSize:10, color:nodeColor, fontFamily:"'Share Tech Mono'", transition:"all .3s", flexShrink:0 }}>
                          {done&&!active?"✓":i+1}
                        </div>
                        {i<STEPS.length-1&&<div style={{ width:1, height:22, background:done?`linear-gradient(${nodeColor},${isStepDone(i+1)?"#10b981":"#1e293b"})`:"rgba(255,255,255,0.04)" }} />}
                      </div>
                      <div onClick={()=>goStep(i)} style={{ marginLeft:14, paddingTop:5, marginBottom:22, cursor:"pointer" }}>
                        <div style={{ fontSize:10, fontFamily:"'Orbitron',sans-serif", fontWeight:active?700:400, letterSpacing:".1em", color:active?accent:done?"#4b5563":"#1e293b", transition:"color .3s" }}>{s.toUpperCase()}</div>
                        {active&&<div style={{ width:18, height:1.5, background:accent, marginTop:4, boxShadow:`0 0 5px ${accent}`, borderRadius:1 }} />}
                      </div>
                    </div>
                  );
                })}
              </div>
              {pipeline.task&&(
                <div style={{ margin:"8px 12px 0", padding:"12px 14px", border:"1px solid rgba(255,255,255,0.05)", borderRadius:7, background:"rgba(0,0,0,0.3)" }}>
                  <div style={{ fontSize:8, color:"#1e293b", letterSpacing:".2em", marginBottom:10, fontFamily:"'Orbitron',sans-serif" }}>CONFIG</div>
                  {pipeline.task&&<SRow label="task" val={pipeline.task} c={taskColor} />}
                  {pipeline.model&&<SRow label="model" val={pipeline.model.replace(/Classifier|Regressor/,"")} c={modelColor||"#94a3b8"} />}
                  {pipeline.preprocessing.length>0&&<SRow label="steps" val={pipeline.preprocessing.length+" transforms"} c="#10b981" />}
                </div>
              )}
            </aside>

            {/* ── MAIN CONTENT ── */}
            <main style={{ flex:1, overflow:"auto", padding:"44px 56px" }}>
              <div className="anim" key={animKey}>
                {/* Step header */}
                <div style={{ marginBottom:38 }}>
                  <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
                    <span style={{ fontSize:9, color:accent, fontFamily:"'Share Tech Mono'", letterSpacing:".2em" }}>[STEP {step+1}/{STEPS.length}]</span>
                    <div style={{ flex:1, height:1, background:`linear-gradient(90deg,${accent}55,transparent)` }} />
                  </div>
                  <h1 style={{ fontFamily:"'Orbitron',sans-serif", fontWeight:900, fontSize:30, color:"#f8fafc", letterSpacing:"-.01em", marginBottom:8 }}>{STEPS[step]}</h1>
                  <p style={{ fontSize:14, color:"#334155", letterSpacing:".02em" }}>{["Select your ML task type","Choose your data source","Pick preprocessing transforms","Select your algorithm","Configure training parameters","Choose evaluation metrics","Set export format"][step]}</p>
                </div>

                {/* STEP 0 */}
                {step===0&&(
                  <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:20, maxWidth:760 }}>
                    {TASK_OPTIONS.map((t,i)=>(
                      <div key={t.id} className="hov" onClick={()=>{update("task",t.id);update("model",null);update("preprocessing",[]);update("evaluation",[]);}}
                        style={{ padding:"34px 24px", border:`1px solid ${pipeline.task===t.id?t.color:"rgba(255,255,255,0.05)"}`, borderRadius:12, background:pipeline.task===t.id?`${t.color}0d`:"rgba(255,255,255,0.015)", textAlign:"center", boxShadow:pipeline.task===t.id?`0 0 32px ${t.color}22,inset 0 0 24px ${t.color}06`:"none", animationDelay:`${i*.07}s` }}>
                        <div style={{ fontSize:38, marginBottom:14, color:t.color, textShadow:`0 0 14px ${t.color}` }}>{t.icon}</div>
                        <div style={{ fontFamily:"'Orbitron',sans-serif", fontWeight:700, fontSize:11, color:pipeline.task===t.id?t.color:"#64748b", marginBottom:10, letterSpacing:".12em" }}>{t.label.toUpperCase()}</div>
                        <div style={{ fontSize:12, color:"#334155", lineHeight:1.65 }}>{t.desc}</div>
                        {pipeline.task===t.id&&<div style={{ margin:"16px auto 0", width:28, height:2, background:t.color, boxShadow:`0 0 8px ${t.color}`, borderRadius:1 }} />}
                      </div>
                    ))}
                  </div>
                )}

                {/* STEP 1 */}
                {step===1&&(
                  <div style={{ display:"flex", flexDirection:"column", gap:10, maxWidth:480 }}>
                    {DATA_SOURCES.map((d,i)=>(
                      <div key={d.id} className="hov-row" onClick={()=>update("dataSource",d.id)}
                        style={{ padding:"15px 20px", border:`1px solid ${pipeline.dataSource===d.id?accent:"rgba(255,255,255,0.05)"}`, borderRadius:9, background:pipeline.dataSource===d.id?`${accent}0a`:"rgba(255,255,255,0.015)", display:"flex", alignItems:"center", gap:16, boxShadow:pipeline.dataSource===d.id?`0 0 22px ${accent}22`:"none" }}>
                        <div style={{ width:38, height:38, border:`1px solid ${pipeline.dataSource===d.id?accent:"rgba(255,255,255,0.07)"}`, borderRadius:7, display:"flex", alignItems:"center", justifyContent:"center", fontSize:16, color:pipeline.dataSource===d.id?accent:"#1e293b", flexShrink:0, boxShadow:pipeline.dataSource===d.id?`0 0 10px ${accent}44`:"none" }}>{d.icon}</div>
                        <div style={{ flex:1 }}>
                          <div style={{ fontSize:14, fontWeight:600, color:pipeline.dataSource===d.id?accent:"#94a3b8", letterSpacing:".04em" }}>{d.label}</div>
                          <div style={{ fontSize:11, color:"#1e293b", marginTop:2, fontFamily:"'Share Tech Mono'" }}>{d.desc}</div>
                        </div>
                        {pipeline.dataSource===d.id&&<div style={{ color:accent, fontSize:12, textShadow:`0 0 8px ${accent}` }}>◈</div>}
                      </div>
                    ))}
                  </div>
                )}

                {/* STEP 2 */}
                {step===2&&pipeline.task&&(
                  <div>
                    <div style={{ display:"flex", flexWrap:"wrap", gap:10, maxWidth:680 }}>
                      {PREPROCESSING[pipeline.task].map(p=>{
                        const sel=pipeline.preprocessing.includes(p);
                        return (
                          <div key={p} className="hov" onClick={()=>toggleList("preprocessing",p)}
                            style={{ padding:"10px 18px", border:`1px solid ${sel?accent:"rgba(255,255,255,0.05)"}`, borderRadius:6, background:sel?`${accent}0e`:"rgba(255,255,255,0.015)", fontSize:12, fontFamily:"'Share Tech Mono'", color:sel?accent:"#334155", boxShadow:sel?`0 0 12px ${accent}33`:"none", display:"flex", alignItems:"center", gap:8 }}>
                            <span style={{ fontSize:9, color:sel?accent:"#1e293b" }}>{sel?"▣":"▢"}</span>{p}
                          </div>
                        );
                      })}
                    </div>
                    <div style={{ marginTop:14, fontSize:10, color:"#1e293b", fontFamily:"'Share Tech Mono'" }}>// optional — skip to proceed without preprocessing</div>
                  </div>
                )}

                {/* STEP 3 */}
                {step===3&&pipeline.task&&(
                  <div style={{ display:"grid", gridTemplateColumns:"repeat(2,1fr)", gap:12, maxWidth:700 }}>
                    {MODELS[pipeline.task].map((m,i)=>(
                      <div key={m.id} className="hov" onClick={()=>update("model",m.id)}
                        style={{ padding:"16px 18px", border:`1px solid ${pipeline.model===m.id?m.color:"rgba(255,255,255,0.05)"}`, borderRadius:9, background:pipeline.model===m.id?`${m.color}0c`:"rgba(255,255,255,0.015)", display:"flex", alignItems:"center", justifyContent:"space-between", boxShadow:pipeline.model===m.id?`0 0 22px ${m.color}22`:"none", animationDelay:`${i*.04}s` }}>
                        <div>
                          <div style={{ fontSize:13, fontWeight:600, color:pipeline.model===m.id?m.color:"#94a3b8", letterSpacing:".02em" }}>{m.label}</div>
                          {pipeline.model===m.id&&<div style={{ width:14, height:1.5, background:m.color, marginTop:5, boxShadow:`0 0 4px ${m.color}`, borderRadius:1 }} />}
                        </div>
                        <span style={{ fontSize:9, padding:"3px 9px", borderRadius:3, border:`1px solid ${m.color}44`, color:m.color, background:`${m.color}0f`, fontFamily:"'Share Tech Mono'", letterSpacing:".06em" }}>{m.tag}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* STEP 4 */}
                {step===4&&(
                  <div style={{ display:"flex", flexDirection:"column", gap:20, maxWidth:500 }}>
                    {[
                      { label:"TEST SPLIT", key:"testSplit", min:.1, max:.4, step:.05, fmt:v=>`${Math.round(v*100)}%` },
                      { label:"CV FOLDS",   key:"cvFolds",   min:2,  max:10, step:1,   fmt:v=>`${v}-fold` },
                      { label:"RANDOM STATE", key:"randomState", min:0, max:100, step:1, fmt:v=>v },
                    ].map(({label,key,min,max,step:s,fmt})=>(
                      <div key={key} style={{ padding:"22px 24px", border:"1px solid rgba(255,255,255,0.05)", borderRadius:10, background:"rgba(255,255,255,0.015)" }}>
                        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:18, alignItems:"center" }}>
                          <span style={{ fontSize:9, fontFamily:"'Orbitron',sans-serif", color:"#334155", letterSpacing:".18em" }}>{label}</span>
                          <span style={{ fontSize:22, fontFamily:"'Share Tech Mono'", color:accent, textShadow:`0 0 10px ${accent}88` }}>{fmt(pipeline.training[key])}</span>
                        </div>
                        <input type="range" min={min} max={max} step={s} value={pipeline.training[key]}
                          onChange={e=>update("training",{...pipeline.training,[key]:parseFloat(e.target.value)})}
                          style={{ accentColor:accent }} />
                        <div style={{ display:"flex", justifyContent:"space-between", fontSize:9, color:"#1e293b", marginTop:8, fontFamily:"'Share Tech Mono'" }}>
                          <span>{fmt(min)}</span><span>{fmt(max)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* STEP 5 */}
                {step===5&&pipeline.task&&(
                  <div style={{ display:"flex", flexWrap:"wrap", gap:10, maxWidth:680 }}>
                    {METRICS[pipeline.task].map(m=>{
                      const sel=pipeline.evaluation.includes(m);
                      return (
                        <div key={m} className="hov" onClick={()=>toggleList("evaluation",m)}
                          style={{ padding:"10px 18px", border:`1px solid ${sel?"#10b981":"rgba(255,255,255,0.05)"}`, borderRadius:6, background:sel?"rgba(16,185,129,0.09)":"rgba(255,255,255,0.015)", fontSize:12, fontFamily:"'Share Tech Mono'", color:sel?"#10b981":"#334155", boxShadow:sel?"0 0 12px rgba(16,185,129,0.28)":"none", display:"flex", alignItems:"center", gap:8 }}>
                          <span style={{ fontSize:9, color:sel?"#10b981":"#1e293b" }}>{sel?"▣":"▢"}</span>{m}
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* STEP 6 */}
                {step===6&&(
                  <div>
                    <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14, maxWidth:580, marginBottom:36 }}>
                      {EXPORT_FORMATS.map((f,i)=>(
                        <div key={f.id} className="hov" onClick={()=>update("exportFormat",f.id)}
                          style={{ padding:"20px 16px", border:`1px solid ${pipeline.exportFormat===f.id?accent:"rgba(255,255,255,0.05)"}`, borderRadius:9, background:pipeline.exportFormat===f.id?`${accent}0a`:"rgba(255,255,255,0.015)", textAlign:"center", boxShadow:pipeline.exportFormat===f.id?`0 0 22px ${accent}22`:"none", animationDelay:`${i*.05}s` }}>
                          <div style={{ fontSize:22, color:pipeline.exportFormat===f.id?accent:"#1e293b", marginBottom:10 }}>{f.icon}</div>
                          <div style={{ fontSize:12, fontWeight:600, color:pipeline.exportFormat===f.id?accent:"#4b5563", letterSpacing:".04em" }}>{f.label}</div>
                          {f.ext&&<div style={{ fontSize:9, color:"#1e293b", fontFamily:"'Share Tech Mono'", marginTop:3 }}>{f.ext}</div>}
                        </div>
                      ))}
                    </div>

                    {/* Terminal summary */}
                    <div style={{ padding:"22px 26px", border:"1px solid rgba(0,245,255,0.1)", borderRadius:10, background:"rgba(0,0,0,0.45)", maxWidth:560, backdropFilter:"blur(12px)" }}>
                      <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:16 }}>
                        {["#f43f5e","#f59e0b","#10b981"].map(c=><div key={c} style={{ width:9,height:9,borderRadius:"50%",background:c }} />)}
                        <span style={{ marginLeft:10, fontSize:9, color:"#1e293b", fontFamily:"'Share Tech Mono'" }}>pipeline.summary()</span>
                      </div>
                      {[["task",pipeline.task],["data_source",pipeline.dataSource],["preprocessing",pipeline.preprocessing.length?pipeline.preprocessing.join(", "):"none"],["model",pipeline.model],["test_split",`${Math.round(pipeline.training.testSplit*100)}%`],["cv_folds",pipeline.training.cvFolds],["metrics",pipeline.evaluation.join(", ")||"none"],["export",pipeline.exportFormat]].map(([k,v])=>(
                        <div key={k} style={{ display:"flex", gap:10, marginBottom:7, fontSize:11, fontFamily:"'Share Tech Mono'" }}>
                          <span style={{ color:accent, width:110, flexShrink:0 }}>{k}</span>
                          <span style={{ color:"#334155" }}>→</span>
                          <span style={{ color:"#64748b" }}>{v||"—"}</span>
                        </div>
                      ))}
                      <div style={{ marginTop:6, fontSize:11, fontFamily:"'Share Tech Mono'", color:"#10b981", display:"flex", alignItems:"center", gap:4 }}>
                        <span>▶</span><span style={{ animation:"blink 1s infinite" }}>_</span>
                      </div>
                    </div>

                    <div onClick={()=>setActiveTab("code")} className="hov"
                      style={{ marginTop:24, display:"inline-flex", alignItems:"center", gap:10, padding:"12px 28px", border:`1px solid ${accent}`, borderRadius:7, background:`${accent}12`, color:accent, fontFamily:"'Orbitron',sans-serif", fontSize:10, fontWeight:700, letterSpacing:".16em", boxShadow:`0 0 22px ${accent}33`, cursor:"pointer" }}>
                      ⟨/⟩ VIEW GENERATED CODE →
                    </div>
                  </div>
                )}

                {/* Nav buttons */}
                <div style={{ display:"flex", gap:12, marginTop:46 }}>
                  {step>0&&(
                    <button onClick={()=>goStep(step-1)} style={{ padding:"10px 26px", border:"1px solid rgba(255,255,255,0.07)", borderRadius:7, background:"transparent", color:"#334155", cursor:"pointer", fontFamily:"'Orbitron',sans-serif", fontSize:9, letterSpacing:".12em", transition:"all .2s" }}>← BACK</button>
                  )}
                  {step<STEPS.length-1&&(
                    <button onClick={()=>canAdvance()&&goStep(step+1)}
                      style={{ padding:"10px 30px", border:`1px solid ${canAdvance()?accent:"rgba(255,255,255,0.05)"}`, borderRadius:7, background:canAdvance()?`${accent}14`:"transparent", color:canAdvance()?accent:"#1e293b", cursor:canAdvance()?"pointer":"not-allowed", fontFamily:"'Orbitron',sans-serif", fontSize:9, fontWeight:700, letterSpacing:".16em", boxShadow:canAdvance()?`0 0 18px ${accent}33`:"none", transition:"all .25s" }}>
                      {step===2||step===4||step===5?"CONTINUE →":"NEXT →"}
                    </button>
                  )}
                </div>
              </div>
            </main>
          </div>
        ) : (
          /* ── CODE TAB ── */
          <div style={{ flex:1, padding:"36px 52px", overflow:"auto" }}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:26 }}>
              <div>
                <h2 style={{ fontFamily:"'Orbitron',sans-serif", fontWeight:700, fontSize:19, color:"#f8fafc", letterSpacing:".06em" }}>GENERATED CODE</h2>
                <div style={{ fontSize:10, color:"#1e293b", marginTop:5, fontFamily:"'Share Tech Mono'" }}>Python · scikit-learn · copy & run</div>
              </div>
              <div style={{ display:"flex", gap:10 }}>
                <button onClick={()=>setActiveTab("builder")} style={{ padding:"9px 22px", border:"1px solid rgba(255,255,255,0.07)", borderRadius:6, background:"transparent", color:"#334155", cursor:"pointer", fontFamily:"'Orbitron',sans-serif", fontSize:9, letterSpacing:".1em" }}>← BUILDER</button>
                <button onClick={copyCode} style={{ padding:"9px 26px", border:`1px solid ${copied?"#10b981":accent}`, borderRadius:6, background:copied?"rgba(16,185,129,0.1)":`${accent}12`, color:copied?"#10b981":accent, cursor:"pointer", fontFamily:"'Orbitron',sans-serif", fontSize:9, fontWeight:700, letterSpacing:".14em", boxShadow:`0 0 14px ${copied?"#10b981":accent}33`, transition:"all .3s" }}>
                  {copied?"✓ COPIED":"⎘ COPY"}
                </button>
              </div>
            </div>
            <div style={{ border:"1px solid rgba(0,245,255,0.09)", borderRadius:12, background:"rgba(0,0,0,0.55)", overflow:"hidden", backdropFilter:"blur(12px)" }}>
              <div style={{ padding:"10px 18px", borderBottom:"1px solid rgba(0,245,255,0.06)", display:"flex", alignItems:"center", gap:8, background:"rgba(0,0,0,0.3)" }}>
                {["#f43f5e","#f59e0b","#10b981"].map(c=><div key={c} style={{ width:9,height:9,borderRadius:"50%",background:c }} />)}
                <span style={{ marginLeft:10, fontSize:10, color:"#1e293b", fontFamily:"'Share Tech Mono'" }}>pipeline.py</span>
                <div style={{ marginLeft:"auto", fontSize:9, color:`${accent}88`, fontFamily:"'Share Tech Mono'" }}>sklearn</div>
              </div>
              <div style={{ padding:"26px 30px", overflowX:"auto" }}>
                <pre style={{ margin:0, fontSize:12, lineHeight:1.85 }}>
                  {generateCode(pipeline).split("\n").map((line,i)=>{
                    let c="#334155";
                    if(line.trim().startsWith("#")) c="#1e293b";
                    else if(line.startsWith("import")||line.startsWith("from")) c="#60a5fa";
                    else if(line.includes("pipeline")&&line.includes("=")&&!line.includes("==")) c=accent;
                    else if(line.includes("Pipeline(")||line.includes(".fit(")||line.includes(".predict(")||line.includes(".fit_predict(")) c="#a855f7";
                    else if(line.includes("print(")) c="#10b981";
                    else if(line.includes("train_test_split")||line.includes("cross_val_score")) c="#f59e0b";
                    else if(line.includes("===")||line.includes("('")) c="#94a3b8";
                    return <span key={i} style={{ color:c, display:"block" }}>{line||" "}</span>;
                  })}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SRow({ label, val, c }) {
  return (
    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6, fontSize:9 }}>
      <span style={{ color:"#1e293b", fontFamily:"'Share Tech Mono'" }}>{label}</span>
      <span style={{ color:c, fontFamily:"'Share Tech Mono'", maxWidth:90, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{val}</span>
    </div>
  );
}
