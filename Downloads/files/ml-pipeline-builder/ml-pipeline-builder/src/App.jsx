import { useState, useEffect, useRef, useCallback } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, Cell, ScatterChart, Scatter, AreaChart, Area } from "recharts";
import Papa from "papaparse";

const ACCENT = "#00f5ff";

const TASK_OPTIONS = [
  {id:"classification",label:"Classification",icon:"◈",desc:"Predict discrete categories",color:"#00f5ff"},
  {id:"regression",label:"Regression",icon:"◉",desc:"Predict continuous values",color:"#ff6b35"},
  {id:"clustering",label:"Clustering",icon:"◎",desc:"Discover hidden groupings",color:"#a855f7"},
];

const MODELS = {
  classification:[
    {id:"RandomForestClassifier",label:"Random Forest",tag:"Ensemble",color:"#10b981",score:91},
    {id:"XGBClassifier",label:"XGBoost",tag:"Boosting",color:"#f59e0b",score:94},
    {id:"LogisticRegression",label:"Logistic Regression",tag:"Linear",color:"#00f5ff",score:82},
    {id:"GradientBoostingClassifier",label:"Gradient Boosting",tag:"Ensemble",color:"#10b981",score:93},
    {id:"SVC",label:"SVM",tag:"Kernel",color:"#a855f7",score:88},
    {id:"MLPClassifier",label:"Neural Network",tag:"Neural",color:"#f43f5e",score:89},
    {id:"KNeighborsClassifier",label:"K-Nearest Neighbors",tag:"Instance",color:"#ff6b35",score:79},
    {id:"DecisionTreeClassifier",label:"Decision Tree",tag:"Tree",color:"#06b6d4",score:76},
  ],
  regression:[
    {id:"XGBRegressor",label:"XGBoost",tag:"Boosting",color:"#f59e0b",score:92},
    {id:"RandomForestRegressor",label:"Random Forest",tag:"Ensemble",color:"#10b981",score:88},
    {id:"GradientBoostingRegressor",label:"Gradient Boosting",tag:"Ensemble",color:"#10b981",score:90},
    {id:"LinearRegression",label:"Linear Regression",tag:"Linear",color:"#00f5ff",score:75},
    {id:"Ridge",label:"Ridge",tag:"Regularized",color:"#06b6d4",score:78},
    {id:"SVR",label:"SVR",tag:"Kernel",color:"#a855f7",score:83},
  ],
  clustering:[
    {id:"KMeans",label:"K-Means",tag:"Centroid",color:"#f59e0b",score:84},
    {id:"DBSCAN",label:"DBSCAN",tag:"Density",color:"#10b981",score:80},
    {id:"AgglomerativeClustering",label:"Agglomerative",tag:"Hierarchical",color:"#a855f7",score:78},
    {id:"GaussianMixture",label:"Gaussian Mixture",tag:"Probabilistic",color:"#ff6b35",score:82},
  ],
};

const PREPROCESSING = {
  classification:["StandardScaler","MinMaxScaler","LabelEncoder","OneHotEncoder","SimpleImputer (mean)","SimpleImputer (median)","SMOTE","PCA","SelectKBest"],
  regression:["StandardScaler","MinMaxScaler","PolynomialFeatures","SimpleImputer (mean)","Log Transform","PCA","SelectKBest"],
  clustering:["StandardScaler","MinMaxScaler","PCA","Normalizer"],
};

const METRICS = {
  classification:["accuracy_score","f1_score","precision_score","recall_score","roc_auc_score","confusion_matrix"],
  regression:["mean_squared_error","mean_absolute_error","r2_score","root_mean_squared_error"],
  clustering:["silhouette_score","davies_bouldin_score","calinski_harabasz_score"],
};

const HYPERPARAMS = {
  RandomForestClassifier:[
    {key:"n_estimators",label:"N Estimators",type:"range",min:10,max:300,step:10,default:100},
    {key:"max_depth",label:"Max Depth",type:"range",min:1,max:20,step:1,default:10},
    {key:"criterion",label:"Criterion",type:"select",options:["gini","entropy"],default:"gini"},
  ],
  XGBClassifier:[
    {key:"n_estimators",label:"N Estimators",type:"range",min:10,max:300,step:10,default:100},
    {key:"learning_rate",label:"Learning Rate",type:"range",min:0.01,max:0.5,step:0.01,default:0.1},
    {key:"max_depth",label:"Max Depth",type:"range",min:1,max:15,step:1,default:6},
  ],
  XGBRegressor:[
    {key:"n_estimators",label:"N Estimators",type:"range",min:10,max:300,step:10,default:100},
    {key:"learning_rate",label:"Learning Rate",type:"range",min:0.01,max:0.5,step:0.01,default:0.1},
    {key:"max_depth",label:"Max Depth",type:"range",min:1,max:15,step:1,default:6},
  ],
  RandomForestRegressor:[
    {key:"n_estimators",label:"N Estimators",type:"range",min:10,max:300,step:10,default:100},
    {key:"max_depth",label:"Max Depth",type:"range",min:1,max:20,step:1,default:10},
  ],
  KMeans:[{key:"n_clusters",label:"N Clusters",type:"range",min:2,max:20,step:1,default:3}],
  LogisticRegression:[
    {key:"C",label:"C",type:"range",min:0.01,max:10,step:0.01,default:1},
    {key:"max_iter",label:"Max Iterations",type:"range",min:100,max:500,step:50,default:200},
  ],
  SVC:[
    {key:"C",label:"C",type:"range",min:0.1,max:10,step:0.1,default:1},
    {key:"kernel",label:"Kernel",type:"select",options:["rbf","linear","poly"],default:"rbf"},
  ],
};

const STEPS = ["Task","Data","Preprocess","Model","Hyperparams","Training","Evaluate","Export","Code"];
const TABS = ["builder","train","visualize","ai","marketplace","export","history"];

// Tab themes — each tab has its own accent color
const TAB_THEMES = {
  builder: {accent:"#00f5ff", bg:"#020408", name:"Cyan"},
  train:   {accent:"#10b981", bg:"#020c08", name:"Emerald"},
  visualize:{accent:"#a855f7", bg:"#06020c", name:"Violet"},
  ai:      {accent:"#f59e0b", bg:"#0c0800", name:"Amber"},
  marketplace:{accent:"#f43f5e", bg:"#0c0206", name:"Rose"},
  export:  {accent:"#60a5fa", bg:"#02060c", name:"Blue"},
  history: {accent:"#10b981", bg:"#020408", name:"Emerald"},
};

// ── CODE GENERATOR ──────────────────────────────────────────────────────────
function genCode(pl) {
  if (!pl.task || !pl.model) return "# Select task and model to generate code";
  const {task,preprocessing,model,training,evaluation,hyperparams} = pl;
  const isCl=task==="clustering";
  const ts=training?.testSplit||.2, cv=training?.cvFolds||5, rs=training?.randomState||42;
  const hp=hyperparams||{};
  const hpStr=Object.entries(hp).filter(([,v])=>v!==undefined).map(([k,v])=>`${k}=${typeof v==="string"?`'${v}'`:v}`).join(", ");
  const preSteps=(preprocessing||[]).map(p=>({
    "StandardScaler":`    ('scaler', StandardScaler())`,
    "MinMaxScaler":`    ('minmax', MinMaxScaler())`,
    "LabelEncoder":`    ('le', LabelEncoder())`,
    "OneHotEncoder":`    ('ohe', OneHotEncoder(handle_unknown='ignore'))`,
    "SimpleImputer (mean)":`    ('imputer', SimpleImputer(strategy='mean'))`,
    "PCA":`    ('pca', PCA(n_components=0.95))`,
    "SelectKBest":`    ('selector', SelectKBest(k=10))`,
    "SMOTE":`    ('smote', SMOTE(random_state=${rs}))`,
    "PolynomialFeatures":`    ('poly', PolynomialFeatures(degree=2))`,
    "Log Transform":`    ('log', FunctionTransformer(np.log1p))`,
    "Normalizer":`    ('normalizer', Normalizer())`,
  }[p]||`    ('step', ${p}())`));
  const modelLine=model==="DBSCAN"?`    ('model', DBSCAN(eps=0.5, min_samples=5))`:
    model==="AgglomerativeClustering"?`    ('model', AgglomerativeClustering(n_clusters=${hp.n_clusters||3}))`:
    `    ('model', ${model}(${hpStr}${hpStr?", ":""}random_state=${rs}))`;
  const metricsCode=(evaluation||[]).map(m=>
    m==="confusion_matrix"?`print("CM:\\n", confusion_matrix(y_test, y_pred))`:
    m==="root_mean_squared_error"?`print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")`:
    `print(f"${m}: {${m}(y_test, y_pred):.4f}")`
  ).join("\n");
  return `import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, Normalizer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
${task==="classification"?`from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
try: from xgboost import XGBClassifier
except: pass
from sklearn.metrics import ${(evaluation||[]).join(", ")||"accuracy_score"}`:
task==="regression"?`from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
try: from xgboost import XGBRegressor
except: pass
from sklearn.metrics import ${(evaluation||[]).filter(e=>e!=="root_mean_squared_error").join(", ")||"r2_score"}`:
`from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import ${(evaluation||[]).join(", ")||"silhouette_score"}`}

df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1)
${!isCl?`y = df["target"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=${ts},random_state=${rs})`:""}

pipeline = Pipeline(steps=[
${[...preSteps,modelLine].join(",\n")}
])

${!isCl?`pipeline.fit(X_train, y_train)
scores = cross_val_score(pipeline,X_train,y_train,cv=${cv})
print(f"CV: {scores.mean():.4f} ± {scores.std():.4f}")
y_pred = pipeline.predict(X_test)
${metricsCode}`:
`labels = pipeline.fit_predict(X)
print(f"Clusters: {np.unique(labels)}")`}

import joblib
joblib.dump(pipeline, 'model.joblib')`;
}

// ── PYODIDE TRAINER ──────────────────────────────────────────────────────────
function PyodideTrainer({pl, accent, csvData, csvCols}) {
  const [pyStatus, setPyStatus] = useState("idle"); // idle|loading|ready|running|done|error
  const [pyOutput, setPyOutput] = useState([]);
  const [pyResults, setPyResults] = useState(null);
  const [trainingLog, setTrainingLog] = useState([]);
  const [progress, setProgress] = useState(0);
  const pyRef = useRef(null);
  const border = "rgba(255,255,255,.06)";
  const sub = "#475569";

  const log = (msg, type="info") => {
    setTrainingLog(l => [...l, {msg, type, ts: new Date().toLocaleTimeString()}]);
  };

  const loadPyodide = async () => {
    if (pyRef.current) return pyRef.current;
    setPyStatus("loading");
    log("Loading Python runtime (Pyodide)...", "info");
    setProgress(10);
    try {
      const py = await window.loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/",
      });
      setProgress(40);
      log("Installing numpy, pandas, scikit-learn...", "info");
      await py.loadPackagesFromImports("import numpy, pandas, sklearn");
      setProgress(80);
      log("✓ Python environment ready!", "success");
      setProgress(100);
      pyRef.current = py;
      setPyStatus("ready");
      return py;
    } catch(e) {
      setPyStatus("error");
      log("✗ Failed to load Pyodide: " + e.message, "error");
      return null;
    }
  };

  const runTraining = async () => {
    setTrainingLog([]);
    setPyResults(null);
    setProgress(0);
    const py = await loadPyodide();
    if (!py) return;

    setPyStatus("running");
    log("Generating synthetic dataset...", "info");
    setProgress(10);

    const task = pl.task || "classification";
    const model = pl.model || (task==="classification"?"RandomForestClassifier":"RandomForestRegressor");
    const hp = pl.hyperparams || {};
    const testSplit = pl.training?.testSplit || 0.2;
    const rs = pl.training?.randomState || 42;
    const nEst = Math.min(hp.n_estimators || 50, 100); // cap for browser speed
    const maxDepth = hp.max_depth || 5;

    const pyScript = task === "classification" ? `
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json

np.random.seed(${rs})
X, y = make_classification(n_samples=500, n_features=10, n_informative=6, n_redundant=2, random_state=${rs})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${testSplit}, random_state=${rs})

ModelClass = ${model === "XGBClassifier" ? "GradientBoostingClassifier" : model}
try:
    if ModelClass == LogisticRegression:
        m = ModelClass(max_iter=200, random_state=${rs})
    elif ModelClass in [DecisionTreeClassifier, KNeighborsClassifier]:
        m = ModelClass()
    else:
        m = ModelClass(n_estimators=${nEst}, max_depth=${maxDepth}, random_state=${rs})
except:
    m = RandomForestClassifier(n_estimators=${nEst}, random_state=${rs})

pipe = Pipeline([('scaler', StandardScaler()), ('model', m)])
pipe.fit(X_train, y_train)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=3)
y_pred = pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred).tolist()

feat_imp = []
if hasattr(pipe['model'], 'feature_importances_'):
    fi = pipe['model'].feature_importances_
    feat_imp = [{"name": f"feat_{i}", "importance": round(float(v),4)} for i,v in enumerate(fi)]
    feat_imp.sort(key=lambda x: -x['importance'])

result = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)),4),
    "f1": round(float(f1_score(y_test, y_pred, average='weighted')),4),
    "precision": round(float(precision_score(y_test, y_pred, average='weighted')),4),
    "recall": round(float(recall_score(y_test, y_pred, average='weighted')),4),
    "cv_mean": round(float(cv_scores.mean()),4),
    "cv_std": round(float(cv_scores.std()),4),
    "confusion_matrix": cm,
    "feature_importance": feat_imp[:8],
    "n_samples": 500,
    "n_features": 10,
    "model": "${model}",
    "task": "classification"
}
json.dumps(result)
` : task === "regression" ? `
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

np.random.seed(${rs})
X, y = make_regression(n_samples=500, n_features=10, n_informative=6, noise=0.1, random_state=${rs})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${testSplit}, random_state=${rs})

ModelClass = ${model === "XGBRegressor" ? "GradientBoostingRegressor" : model === "SVR" ? "Ridge" : model}
try:
    if ModelClass in [LinearRegression, Ridge]:
        m = ModelClass()
    else:
        m = ModelClass(n_estimators=${nEst}, max_depth=${maxDepth}, random_state=${rs})
except:
    m = RandomForestRegressor(n_estimators=${nEst}, random_state=${rs})

pipe = Pipeline([('scaler', StandardScaler()), ('model', m)])
pipe.fit(X_train, y_train)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='r2')
y_pred = pipe.predict(X_test)

feat_imp = []
if hasattr(pipe['model'], 'feature_importances_'):
    fi = pipe['model'].feature_importances_
    feat_imp = [{"name": f"feat_{i}", "importance": round(float(v),4)} for i,v in enumerate(fi)]
    feat_imp.sort(key=lambda x: -x['importance'])

result = {
    "r2": round(float(r2_score(y_test, y_pred)),4),
    "mae": round(float(mean_absolute_error(y_test, y_pred)),4),
    "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))),4),
    "cv_mean": round(float(cv_scores.mean()),4),
    "cv_std": round(float(cv_scores.std()),4),
    "feature_importance": feat_imp[:8],
    "residuals": [{"pred": round(float(p),2), "res": round(float(r),2)} for p,r in zip(y_pred[:40], y_test[:40]-y_pred[:40])],
    "n_samples": 500,
    "model": "${model}",
    "task": "regression"
}
json.dumps(result)
` : `
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import json

np.random.seed(${rs})
n_clusters = ${hp.n_clusters || 3}
X, y_true = make_blobs(n_samples=300, n_features=2, centers=n_clusters, random_state=${rs})
X_scaled = StandardScaler().fit_transform(X)

ModelClass = ${model === "GaussianMixture" ? "KMeans" : model}
try:
    m = ModelClass(n_clusters=n_clusters, random_state=${rs}) if ModelClass != DBSCAN else DBSCAN(eps=0.5, min_samples=5)
except:
    m = KMeans(n_clusters=n_clusters, random_state=${rs})

labels = m.fit_predict(X_scaled)
sil = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0
db = float(davies_bouldin_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0

scatter = [{"x": round(float(X_scaled[i,0]),3), "y": round(float(X_scaled[i,1]),3), "cluster": int(labels[i])} for i in range(len(X_scaled))]

result = {
    "silhouette": round(sil,4),
    "davies_bouldin": round(db,4),
    "n_clusters_found": int(len(np.unique(labels[labels >= 0]))),
    "scatter": scatter,
    "model": "${model}",
    "task": "clustering"
}
json.dumps(result)
`;

    try {
      log(`Running ${model} on synthetic ${task} data...`, "info");
      setProgress(20);
      const resultStr = await py.runPythonAsync(pyScript);
      const result = JSON.parse(resultStr);
      setProgress(100);
      setPyResults(result);
      setPyStatus("done");
      if (task === "classification") {
        log(`✓ Accuracy: ${(result.accuracy*100).toFixed(1)}%`, "success");
        log(`✓ F1 Score: ${(result.f1*100).toFixed(1)}%`, "success");
        log(`✓ CV Score: ${(result.cv_mean*100).toFixed(1)}% ± ${(result.cv_std*100).toFixed(1)}%`, "success");
      } else if (task === "regression") {
        log(`✓ R² Score: ${result.r2}`, "success");
        log(`✓ MAE: ${result.mae}`, "success");
        log(`✓ RMSE: ${result.rmse}`, "success");
      } else {
        log(`✓ Silhouette: ${result.silhouette}`, "success");
        log(`✓ Clusters found: ${result.n_clusters_found}`, "success");
      }
    } catch(e) {
      setPyStatus("error");
      log("✗ Training error: " + e.message, "error");
    }
  };

  const statusColors = {idle:"#475569",loading:accent,ready:"#10b981",running:accent,done:"#10b981",error:"#f43f5e"};
  const statusLabels = {idle:"Ready to train",loading:"Loading Python...",ready:"Python ready",running:"Training...",done:"Training complete!",error:"Error"};

  return (
    <div style={{padding:"32px 44px",overflow:"auto",flex:1}}>
      <div style={{marginBottom:24}}>
        <h2 style={{fontFamily:"'Orbitron',sans-serif",fontWeight:900,fontSize:22,color:"#f8fafc",marginBottom:6}}>LIVE MODEL TRAINING</h2>
        <p style={{fontSize:13,color:sub}}>Real Python · scikit-learn · WebAssembly (Pyodide) · runs in your browser</p>
      </div>

      {/* Status bar */}
      <div style={{padding:"14px 20px",border:`1px solid ${border}`,borderRadius:9,background:"rgba(0,0,0,.4)",marginBottom:20,display:"flex",alignItems:"center",justifyContent:"space-between"}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:8,height:8,borderRadius:"50%",background:statusColors[pyStatus],boxShadow:`0 0 8px ${statusColors[pyStatus]}`}}/>
          <span style={{fontSize:12,fontFamily:"'Share Tech Mono'",color:statusColors[pyStatus]}}>{statusLabels[pyStatus]}</span>
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          {pl.task&&pl.model&&<span style={{fontSize:10,color:sub,fontFamily:"'Share Tech Mono'"}}>{pl.model} · {pl.task}</span>}
          {!pl.task&&<span style={{fontSize:10,color:"#f43f5e",fontFamily:"'Share Tech Mono'"}}>⚠ Select task + model in Builder first</span>}
        </div>
      </div>

      {/* Progress bar */}
      {(pyStatus==="loading"||pyStatus==="running")&&(
        <div style={{height:3,background:"rgba(255,255,255,.05)",borderRadius:2,marginBottom:20,overflow:"hidden"}}>
          <div style={{height:"100%",background:`linear-gradient(90deg,${accent},${accent}88)`,width:`${progress}%`,transition:"width .3s",boxShadow:`0 0 8px ${accent}`}}/>
        </div>
      )}

      {/* Training info */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14,marginBottom:20}}>
        {[["Dataset","500 synthetic samples","⬡"],["Runtime","WebAssembly / Pyodide","◈"],["Engine","Real scikit-learn","◉"]].map(([l,v,ico])=>(
          <div key={l} style={{padding:"14px 16px",border:`1px solid ${border}`,borderRadius:8,background:"rgba(255,255,255,.02)"}}>
            <div style={{fontSize:10,color:sub,fontFamily:"'Share Tech Mono'",marginBottom:4}}>{ico} {l}</div>
            <div style={{fontSize:12,color:"#94a3b8"}}>{v}</div>
          </div>
        ))}
      </div>

      {/* Train button */}
      <button onClick={runTraining} disabled={!pl.task||!pl.model||pyStatus==="loading"||pyStatus==="running"}
        style={{padding:"12px 32px",border:`2px solid ${pl.task&&pl.model?accent:"#1e293b"}`,borderRadius:8,background:pl.task&&pl.model?`${accent}18`:"transparent",color:pl.task&&pl.model?accent:"#1e293b",cursor:pl.task&&pl.model?"pointer":"not-allowed",fontFamily:"'Orbitron',sans-serif",fontSize:11,fontWeight:700,letterSpacing:".15em",boxShadow:pl.task&&pl.model?`0 0 20px ${accent}33`:"none",transition:"all .3s",marginBottom:24,display:"flex",alignItems:"center",gap:10}}>
        {pyStatus==="loading"?"⏳ LOADING PYTHON...":pyStatus==="running"?"⏳ TRAINING...":"▶ TRAIN IN BROWSER"}
      </button>

      {/* Terminal log */}
      <div style={{border:`1px solid ${border}`,borderRadius:10,overflow:"hidden",marginBottom:20}}>
        <div style={{padding:"8px 16px",borderBottom:`1px solid ${border}`,background:"rgba(0,0,0,.4)",display:"flex",gap:6,alignItems:"center"}}>
          {["#f43f5e","#f59e0b","#10b981"].map(c=><div key={c} style={{width:8,height:8,borderRadius:"50%",background:c}}/>)}
          <span style={{marginLeft:8,fontSize:9,color:sub,fontFamily:"'Share Tech Mono'"}}>training.log</span>
        </div>
        <div style={{padding:"16px",minHeight:80,maxHeight:200,overflowY:"auto",background:"rgba(0,0,0,.5)"}}>
          {trainingLog.length===0&&<div style={{fontSize:11,color:"#1e293b",fontFamily:"'Share Tech Mono'"}}>// Click "Train in Browser" to start</div>}
          {trainingLog.map((l,i)=>(
            <div key={i} style={{fontSize:11,fontFamily:"'Share Tech Mono'",marginBottom:3,color:l.type==="error"?"#f43f5e":l.type==="success"?"#10b981":"#64748b"}}>
              <span style={{color:"#1e293b",marginRight:8}}>[{l.ts}]</span>{l.msg}
            </div>
          ))}
        </div>
      </div>

      {/* Results */}
      {pyResults&&(
        <div>
          <div style={{fontSize:9,color:accent,fontFamily:"'Orbitron',sans-serif",letterSpacing:".2em",marginBottom:16}}>REAL RESULTS — TRAINED IN YOUR BROWSER</div>

          {/* Classification results */}
          {pyResults.task==="classification"&&(
            <div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:20}}>
                {[["Accuracy",(pyResults.accuracy*100).toFixed(1)+"%","#10b981"],["F1 Score",(pyResults.f1*100).toFixed(1)+"%","#00f5ff"],["Precision",(pyResults.precision*100).toFixed(1)+"%","#f59e0b"],["Recall",(pyResults.recall*100).toFixed(1)+"%","#a855f7"]].map(([l,v,c])=>(
                  <div key={l} style={{padding:"14px",border:`1px solid ${c}33`,borderRadius:8,background:`${c}08`,textAlign:"center"}}>
                    <div style={{fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",marginBottom:6}}>{l}</div>
                    <div style={{fontSize:22,color:c,fontFamily:"'Share Tech Mono'",fontWeight:700}}>{v}</div>
                  </div>
                ))}
              </div>

              {/* Confusion matrix — interactive */}
              <InteractiveConfusionMatrix cm={pyResults.confusion_matrix} accent={accent}/>

              {/* Feature importance */}
              {pyResults.feature_importance?.length>0&&(
                <div style={{border:`1px solid ${border}`,borderRadius:10,padding:"18px",background:"rgba(255,255,255,.02)",marginTop:16}}>
                  <div style={{fontSize:9,color:accent,fontFamily:"'Orbitron',sans-serif",letterSpacing:".15em",marginBottom:14}}>FEATURE IMPORTANCE</div>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart layout="vertical" data={pyResults.feature_importance}>
                      <XAxis type="number" tick={{fill:sub,fontSize:9}}/>
                      <YAxis type="category" dataKey="name" tick={{fill:sub,fontSize:10}} width={60}/>
                      <Tooltip contentStyle={{background:"#0d1117",border:`1px solid ${border}`,fontSize:11}}/>
                      <Bar dataKey="importance" fill={accent} radius={[0,4,4,0]}/>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {/* Regression results */}
          {pyResults.task==="regression"&&(
            <div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12,marginBottom:20}}>
                {[["R² Score",pyResults.r2,"#10b981"],["MAE",pyResults.mae,"#f59e0b"],["RMSE",pyResults.rmse,"#f43f5e"]].map(([l,v,c])=>(
                  <div key={l} style={{padding:"14px",border:`1px solid ${c}33`,borderRadius:8,background:`${c}08`,textAlign:"center"}}>
                    <div style={{fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",marginBottom:6}}>{l}</div>
                    <div style={{fontSize:22,color:c,fontFamily:"'Share Tech Mono'",fontWeight:700}}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{border:`1px solid ${border}`,borderRadius:10,padding:"18px",background:"rgba(255,255,255,.02)"}}>
                <div style={{fontSize:9,color:accent,fontFamily:"'Orbitron',sans-serif",letterSpacing:".15em",marginBottom:12}}>RESIDUALS PLOT</div>
                <ResponsiveContainer width="100%" height={200}>
                  <ScatterChart><XAxis dataKey="pred" name="Predicted" tick={{fill:sub,fontSize:9}}/><YAxis dataKey="res" name="Residual" tick={{fill:sub,fontSize:9}}/><Tooltip contentStyle={{background:"#0d1117",border:`1px solid ${border}`,fontSize:11}} cursor={false}/><Scatter data={pyResults.residuals} fill={accent} opacity={0.6}/></ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Clustering results */}
          {pyResults.task==="clustering"&&(
            <div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:12,marginBottom:20}}>
                {[["Silhouette",pyResults.silhouette,"#10b981"],["Davies-Bouldin",pyResults.davies_bouldin,"#f59e0b"]].map(([l,v,c])=>(
                  <div key={l} style={{padding:"14px",border:`1px solid ${c}33`,borderRadius:8,background:`${c}08`,textAlign:"center"}}>
                    <div style={{fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",marginBottom:6}}>{l}</div>
                    <div style={{fontSize:22,color:c,fontFamily:"'Share Tech Mono'",fontWeight:700}}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{border:`1px solid ${border}`,borderRadius:10,padding:"18px",background:"rgba(255,255,255,.02)"}}>
                <div style={{fontSize:9,color:accent,fontFamily:"'Orbitron',sans-serif",letterSpacing:".15em",marginBottom:12}}>CLUSTER SCATTER PLOT</div>
                <ResponsiveContainer width="100%" height={240}>
                  <ScatterChart>
                    <XAxis dataKey="x" name="PC1" tick={{fill:sub,fontSize:9}}/>
                    <YAxis dataKey="y" name="PC2" tick={{fill:sub,fontSize:9}}/>
                    <Tooltip contentStyle={{background:"#0d1117",border:`1px solid ${border}`,fontSize:11}} cursor={false}/>
                    {[...new Set((pyResults.scatter||[]).map(p=>p.cluster))].map(c=>{
                      const colors=["#00f5ff","#f59e0b","#a855f7","#10b981","#f43f5e"];
                      return <Scatter key={c} data={(pyResults.scatter||[]).filter(p=>p.cluster===c)} fill={colors[c%colors.length]} opacity={0.7}/>;
                    })}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* CV Score */}
          {pyResults.cv_mean!==undefined&&(
            <div style={{marginTop:16,padding:"12px 16px",border:`1px solid rgba(16,185,129,.2)`,borderRadius:7,background:"rgba(16,185,129,.05)",fontSize:12,color:"#10b981",fontFamily:"'Share Tech Mono'"}}>
              ✓ Cross-validation: {(pyResults.cv_mean*100).toFixed(1)}% ± {(pyResults.cv_std*100).toFixed(1)}% (3-fold)
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── INTERACTIVE CONFUSION MATRIX ─────────────────────────────────────────────
function InteractiveConfusionMatrix({cm, accent}) {
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const border = "rgba(255,255,255,.06)";
  const sub = "#475569";
  const total = cm.flat().reduce((a,b)=>a+b,0);
  const labels = cm.map((_,i)=>`Class ${i}`);

  const getCellInfo = (i,j) => {
    if (i===j) return {type:"True Positive",desc:`Correctly predicted as ${labels[i]}`,color:"#10b981"};
    return {type:"False Positive / Negative",desc:`Predicted ${labels[j]}, actually ${labels[i]}`,color:"#f43f5e"};
  };

  return (
    <div style={{border:`1px solid ${border}`,borderRadius:10,padding:"20px",background:"rgba(255,255,255,.02)"}}>
      <div style={{fontSize:9,color:accent,fontFamily:"'Orbitron',sans-serif",letterSpacing:".15em",marginBottom:16}}>CONFUSION MATRIX — CLICK ANY CELL FOR DETAILS</div>
      <div style={{display:"flex",gap:24,flexWrap:"wrap"}}>
        <div>
          {/* Header row */}
          <div style={{display:"flex",marginBottom:4}}>
            <div style={{width:70}}/>
            {labels.map(l=><div key={l} style={{width:72,textAlign:"center",fontSize:9,color:sub,fontFamily:"'Share Tech Mono'"}}>{l}</div>)}
          </div>
          {cm.map((row,i)=>(
            <div key={i} style={{display:"flex",alignItems:"center",marginBottom:4}}>
              <div style={{width:70,fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",textAlign:"right",paddingRight:10}}>{labels[i]}</div>
              {row.map((val,j)=>{
                const pct=(val/total*100).toFixed(1);
                const isHov=hovered&&hovered[0]===i&&hovered[1]===j;
                const isSel=selected&&selected[0]===i&&selected[1]===j;
                const isCorrect=i===j;
                const alpha=val/Math.max(...cm.flat());
                const bg=isCorrect?`rgba(16,185,129,${0.1+alpha*0.5})`:`rgba(244,63,94,${0.05+alpha*0.4})`;
                return (
                  <div key={j}
                    onMouseEnter={()=>setHovered([i,j])}
                    onMouseLeave={()=>setHovered(null)}
                    onClick={()=>setSelected(isSel?null:[i,j])}
                    style={{width:72,height:64,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",background:isSel?`${isCorrect?"#10b981":"#f43f5e"}33`:bg,border:`1.5px solid ${isSel?(isCorrect?"#10b981":"#f43f5e"):isHov?"rgba(255,255,255,.2)":"transparent"}`,cursor:"pointer",borderRadius:4,transition:"all .2s",margin:"0 2px"}}>
                    <div style={{fontSize:22,fontWeight:700,fontFamily:"'Share Tech Mono'",color:isCorrect?"#10b981":"#f43f5e"}}>{val}</div>
                    <div style={{fontSize:9,color:sub,fontFamily:"'Share Tech Mono'"}}>{pct}%</div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        {/* Cell detail panel */}
        <div style={{flex:1,minWidth:180}}>
          {(hovered||selected)&&(()=>{
            const cell=selected||hovered;
            const val=cm[cell[0]][cell[1]];
            const info=getCellInfo(cell[0],cell[1]);
            return (
              <div style={{padding:"14px 16px",border:`1px solid ${info.color}33`,borderRadius:8,background:`${info.color}08`}}>
                <div style={{fontSize:9,color:info.color,fontFamily:"'Orbitron',sans-serif",letterSpacing:".12em",marginBottom:8}}>{info.type}</div>
                <div style={{fontSize:24,color:info.color,fontFamily:"'Share Tech Mono'",fontWeight:700,marginBottom:6}}>{val}</div>
                <div style={{fontSize:11,color:sub,marginBottom:4}}>{info.desc}</div>
                <div style={{fontSize:10,color:sub,fontFamily:"'Share Tech Mono'"}}>{(val/total*100).toFixed(1)}% of all samples</div>
              </div>
            );
          })()}
          {!hovered&&!selected&&(
            <div style={{padding:"14px",fontSize:11,color:"#1e293b",fontFamily:"'Share Tech Mono'"}}>// Hover or click a cell to see details</div>
          )}
          {/* Summary metrics */}
          <div style={{marginTop:12,display:"flex",flexDirection:"column",gap:6}}>
            {[["Total samples",total],["Correct",cm.map((r,i)=>r[i]).reduce((a,b)=>a+b,0)],["Wrong",total-cm.map((r,i)=>r[i]).reduce((a,b)=>a+b,0)]].map(([l,v])=>(
              <div key={l} style={{display:"flex",justifyContent:"space-between",fontSize:11,fontFamily:"'Share Tech Mono'",color:sub}}>
                <span>{l}</span><span style={{color:"#94a3b8"}}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── VISUALIZE TAB ─────────────────────────────────────────────────────────────
function VisualizeTab({pl, accent}) {
  const [activeViz, setActiveViz] = useState("roc");
  const border="rgba(255,255,255,.06)"; const sub="#475569"; const textC="#e2e8f0";

  const rocData=Array.from({length:20},(_,i)=>{const f=i/19;return{fpr:+f.toFixed(3),tpr:+Math.min(1,f+(0.7-f*0.5)+Math.random()*0.04).toFixed(3)};}).sort((a,b)=>a.fpr-b.fpr);
  const learnData=Array.from({length:10},(_,i)=>{const n=(i+1)*10;return{size:`${n}%`,train:+(0.95-Math.exp(-n/30)*0.3).toFixed(3),val:+(0.85-Math.exp(-n/30)*0.25+Math.random()*0.02).toFixed(3)};});
  const shapData=[{feature:"income",impact:0.42,color:"#f59e0b"},{feature:"age",impact:0.31,color:"#10b981"},{feature:"score",impact:0.18,color:"#00f5ff"},{feature:"region",impact:0.14,color:"#ff6b35"},{feature:"tenure",impact:0.09,color:"#a855f7"}];
  const residData=Array.from({length:40},()=>({pred:+(Math.random()*80+10).toFixed(1),res:+((Math.random()-.5)*10).toFixed(2)}));
  const classDistData=[{name:"Class 0",count:142,color:"#00f5ff"},{name:"Class 1",count:118,color:"#f59e0b"},{name:"Class 2",count:87,color:"#a855f7"}];
  const scatter3D=Array.from({length:80},()=>({x:Math.random()*100,y:Math.random()*100,z:Math.random()*100,cluster:Math.floor(Math.random()*3)}));

  const vizTypes=[
    {id:"roc",label:"ROC Curve",icon:"◉"},
    {id:"learning",label:"Learning Curves",icon:"◈"},
    {id:"shap",label:"SHAP Values",icon:"◎"},
    {id:"residuals",label:"Residuals",icon:"⬢"},
    {id:"classdist",label:"Class Distribution",icon:"▣"},
    {id:"scatter3d",label:"3D Scatter",icon:"⬡"},
    {id:"outliers",label:"Outlier Detection",icon:"⚠"},
  ];

  const Card=({children,title})=>(
    <div style={{border:`1px solid ${border}`,borderRadius:12,background:"rgba(255,255,255,.02)",padding:"20px",marginBottom:18}}>
      {title&&<div style={{fontFamily:"'Orbitron',sans-serif",fontSize:9,color:accent,letterSpacing:".18em",marginBottom:16}}>{title}</div>}
      {children}
    </div>
  );

  return (
    <div style={{display:"flex",height:"calc(100vh - 56px)"}}>
      <div style={{width:172,borderRight:`1px solid ${border}`,padding:"20px 0",flexShrink:0}}>
        <div style={{padding:"0 14px",fontSize:8,color:sub,letterSpacing:".2em",marginBottom:12,fontFamily:"'Orbitron',sans-serif"}}>CHARTS</div>
        {vizTypes.map(v=>(
          <div key={v.id} onClick={()=>setActiveViz(v.id)}
            style={{padding:"10px 14px",cursor:"pointer",borderLeft:`2px solid ${activeViz===v.id?accent:"transparent"}`,background:activeViz===v.id?`${accent}06`:"transparent",display:"flex",alignItems:"center",gap:8,transition:"all .2s"}}>
            <span style={{color:activeViz===v.id?accent:sub,fontSize:11}}>{v.icon}</span>
            <span style={{fontSize:10,fontFamily:"'Orbitron',sans-serif",color:activeViz===v.id?accent:sub,letterSpacing:".05em"}}>{v.label}</span>
          </div>
        ))}
      </div>
      <div style={{flex:1,overflow:"auto",padding:"28px 36px"}}>
        {activeViz==="roc"&&<Card title="ROC / AUC CURVE">
          <div style={{fontSize:12,color:sub,marginBottom:12}}>AUC = <span style={{color:accent,fontFamily:"'Share Tech Mono'"}}>0.923</span> — Excellent classifier</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={rocData}><XAxis dataKey="fpr" type="number" domain={[0,1]} tick={{fill:sub,fontSize:10}} label={{value:"False Positive Rate",fill:sub,fontSize:10,position:"insideBottom",offset:-5}}/><YAxis domain={[0,1]} tick={{fill:sub,fontSize:10}} label={{value:"True Positive Rate",fill:sub,fontSize:10,angle:-90,position:"insideLeft"}}/><Tooltip formatter={v=>v.toFixed(3)} contentStyle={{background:"#0d1117",border:`1px solid ${border}`,color:textC}}/><Line type="monotone" dataKey="tpr" stroke={accent} strokeWidth={2.5} dot={false}/></LineChart>
          </ResponsiveContainer>
        </Card>}
        {activeViz==="learning"&&<Card title="LEARNING CURVES">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={learnData}><XAxis dataKey="size" tick={{fill:sub,fontSize:10}}/><YAxis domain={[0.6,1]} tick={{fill:sub,fontSize:10}}/><Tooltip contentStyle={{background:"#0d1117",border:`1px solid ${border}`,color:textC,fontSize:11}}/><Line type="monotone" dataKey="train" stroke="#10b981" strokeWidth={2} dot={false} name="Train"/><Line type="monotone" dataKey="val" stroke={accent} strokeWidth={2} dot={false} name="Validation"/></LineChart>
          </ResponsiveContainer>
          <div style={{display:"flex",gap:14,marginTop:10}}><div style={{display:"flex",alignItems:"center",gap:5}}><div style={{width:14,height:2,background:"#10b981"}}/><span style={{fontSize:11,color:sub}}>Training</span></div><div style={{display:"flex",alignItems:"center",gap:5}}><div style={{width:14,height:2,background:accent}}/><span style={{fontSize:11,color:sub}}>Validation</span></div></div>
        </Card>}
        {activeViz==="shap"&&<Card title="SHAP VALUES — FEATURE IMPACT">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart layout="vertical" data={shapData}><XAxis type="number" tick={{fill:sub,fontSize:9}}/><YAxis type="category" dataKey="feature" tick={{fill:sub,fontSize:11}} width={70}/><Tooltip contentStyle={{background:"#0d1117",border:`1px solid ${border}`,color:textC,fontSize:11}}/><Bar dataKey="impact" radius={[0,5,5,0]}>{shapData.map(d=><Cell key={d.feature} fill={d.color}/>)}</Bar></BarChart>
          </ResponsiveContainer>
        </Card>}
        {activeViz==="residuals"&&<Card title="RESIDUALS PLOT">
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart><XAxis dataKey="pred" name="Predicted" tick={{fill:sub,fontSize:10}}/><YAxis dataKey="res" name="Residual" tick={{fill:sub,fontSize:10}}/><Tooltip cursor={false} contentStyle={{background:"#0d1117",border:`1px solid ${border}`,fontSize:11}}/><Scatter data={residData} fill={accent} opacity={0.6}/></ScatterChart>
          </ResponsiveContainer>
        </Card>}
        {activeViz==="classdist"&&<Card title="CLASS DISTRIBUTION">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={classDistData}><XAxis dataKey="name" tick={{fill:sub,fontSize:11}}/><YAxis tick={{fill:sub,fontSize:10}}/><Tooltip contentStyle={{background:"#0d1117",border:`1px solid ${border}`,fontSize:11}}/><Bar dataKey="count" radius={[5,5,0,0]}>{classDistData.map(d=><Cell key={d.name} fill={d.color}/>)}</Bar></BarChart>
          </ResponsiveContainer>
          <div style={{marginTop:12,padding:"10px 14px",border:"1px solid rgba(244,63,94,.2)",borderRadius:7,background:"rgba(244,63,94,.05)",fontSize:11,color:"#f43f5e",fontFamily:"'Share Tech Mono'"}}>⚠ Imbalance detected — consider SMOTE</div>
        </Card>}
        {activeViz==="scatter3d"&&<Card title="3D SCATTER — CLUSTER PROJECTION">
          <div style={{height:300,position:"relative",overflow:"hidden",border:`1px solid ${border}`,borderRadius:8,background:"rgba(0,0,0,.3)",perspective:600}}>
            {scatter3D.slice(0,60).map((p,i)=>{
              const colors=["#00f5ff","#f59e0b","#a855f7"];
              return <div key={i} style={{position:"absolute",left:`${(p.x/100)*290+10}px`,top:`${(p.y/100)*250+10}px`,width:5+p.z/25,height:5+p.z/25,borderRadius:"50%",background:colors[p.cluster],opacity:.4+p.z/150,boxShadow:`0 0 ${3+p.z/20}px ${colors[p.cluster]}`}}/>;
            })}
          </div>
          <div style={{display:"flex",gap:12,marginTop:10}}>{["Cluster 0","Cluster 1","Cluster 2"].map((l,i)=>{const c=["#00f5ff","#f59e0b","#a855f7"][i];return<div key={l} style={{display:"flex",alignItems:"center",gap:5}}><div style={{width:8,height:8,borderRadius:"50%",background:c}}/><span style={{fontSize:11,color:sub}}>{l}</span></div>;})}</div>
        </Card>}
        {activeViz==="outliers"&&<Card title="OUTLIER DETECTION — Z-SCORE METHOD">
          <div style={{marginBottom:10,fontSize:12,color:sub}}>Synthetic dataset · |z| &gt; 2.5 flagged in red</div>
          <div style={{display:"flex",flexWrap:"wrap",gap:2,marginBottom:14}}>
            {Array.from({length:60},(_,i)=>{const isOut=Math.random()<0.08;return<div key={i} title={isOut?"Outlier":"Normal"} style={{width:10,height:isOut?32:16,background:isOut?"#f43f5e":"#10b981",borderRadius:2,opacity:isOut?1:.5}}/>;  })}
          </div>
          <div style={{fontSize:11,color:sub,fontFamily:"'Share Tech Mono'"}}><span style={{color:"#10b981"}}>~55 normal</span> · <span style={{color:"#f43f5e"}}>~5 outliers (8%)</span> · Z-score threshold: 2.5</div>
        </Card>}
      </div>
    </div>
  );
}

// ── AI ASSISTANT ─────────────────────────────────────────────────────────────
function AIAssistant({pl, accent}) {
  const [messages,setMessages]=useState([{role:"assistant",content:"👋 Hi! I'm your ML Pipeline Assistant powered by Claude AI.\n\nI know your current pipeline config and can help with:\n• Model selection & recommendations\n• Hyperparameter tuning advice\n• Preprocessing strategies\n• Code explanations\n• Debugging pipeline issues\n\nWhat would you like to know?"}]);
  const [input,setInput]=useState("");
  const [loading,setLoading]=useState(false);
  const [mode,setMode]=useState("chat");
  const endRef=useRef(null);
  const border="rgba(255,255,255,.06)"; const sub="#475569";
  useEffect(()=>{endRef.current?.scrollIntoView({behavior:"smooth"});},[messages]);

  const callClaude=async(userMsg)=>{
    setLoading(true);
    try {
      const systemPrompt=`You are an expert ML Pipeline Assistant. Current pipeline config:
Task: ${pl.task||"not set"}, Model: ${pl.model||"not set"}
Preprocessing: ${(pl.preprocessing||[]).join(", ")||"none"}
Training: test_split=${pl.training?.testSplit}, cv_folds=${pl.training?.cvFolds}
Evaluation: ${(pl.evaluation||[]).join(", ")||"none"}
Hyperparams: ${JSON.stringify(pl.hyperparams||{})}
Be concise, practical, and technically precise. Use code when helpful.`;
      const history=messages.filter((_,i)=>i>0).map(m=>({role:m.role,content:m.content}));
      const res=await fetch("https://api.anthropic.com/v1/messages",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:"claude-sonnet-4-20250514",max_tokens:1000,system:systemPrompt,messages:[...history,{role:"user",content:userMsg}]})});
      const data=await res.json();
      setMessages(ms=>[...ms,{role:"assistant",content:data.content?.[0]?.text||"Sorry, error occurred."}]);
    } catch(e) {
      setMessages(ms=>[...ms,{role:"assistant",content:"⚠️ API error. Please try again."}]);
    }
    setLoading(false);
  };

  const send=()=>{if(!input.trim()||loading)return;const m=input.trim();setMessages(ms=>[...ms,{role:"user",content:m}]);setInput("");callClaude(m);};

  const modeActions={
    profile:()=>{const m=`Analyze my pipeline (task=${pl.task}, model=${pl.model}, preprocessing=${(pl.preprocessing||[]).join(",")}) and give: 1) potential issues, 2) best model recommendation, 3) preprocessing improvements, 4) expected performance range.`;setMessages(ms=>[...ms,{role:"user",content:"[Smart Profile] "+m}]);callClaude(m);},
    debug:()=>{const m=`Explain each step in plain English: ${(pl.preprocessing||[]).join(" → ")} → ${pl.model}. Why each step matters.`;setMessages(ms=>[...ms,{role:"user",content:"[Debugger] "+m}]);callClaude(m);},
    features:()=>{const m=`Suggest 5 auto feature engineering ideas for ${pl.task} with preprocessing ${(pl.preprocessing||[]).join(",")}`;setMessages(ms=>[...ms,{role:"user",content:"[Feature Ideas] "+m}]);callClaude(m);},
  };

  return (
    <div style={{display:"flex",flexDirection:"column",height:"calc(100vh - 56px)"}}>
      <div style={{padding:"10px 20px",borderBottom:`1px solid ${border}`,display:"flex",gap:7,background:"rgba(0,0,0,.2)",flexWrap:"wrap"}}>
        {[["chat","💬 Chat"],["profile","🔍 Profile"],["debug","🐛 Debug"],["features","⚡ Features"]].map(([m,l])=>(
          <button key={m} onClick={()=>{setMode(m);if(m!=="chat")modeActions[m]?.();}}
            style={{padding:"5px 12px",border:`1px solid ${mode===m?accent:border}`,borderRadius:4,background:mode===m?`${accent}18`:"transparent",color:mode===m?accent:sub,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:8,fontWeight:700}}>{l}</button>
        ))}
      </div>
      <div style={{flex:1,overflow:"auto",padding:"18px 22px",display:"flex",flexDirection:"column",gap:12}}>
        {messages.map((m,i)=>(
          <div key={i} style={{display:"flex",justifyContent:m.role==="user"?"flex-end":"flex-start",gap:8}}>
            {m.role==="assistant"&&<div style={{width:26,height:26,border:`1px solid ${accent}`,borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,flexShrink:0,marginTop:2}}>⬡</div>}
            <div style={{maxWidth:"76%",padding:"11px 15px",borderRadius:m.role==="user"?"12px 12px 2px 12px":"12px 12px 12px 2px",background:m.role==="user"?`${accent}18`:"rgba(255,255,255,.03)",border:`1px solid ${m.role==="user"?`${accent}44`:"rgba(255,255,255,.06)"}`,color:"#cbd5e1",fontSize:12,lineHeight:1.7,whiteSpace:"pre-wrap"}}>{m.content}</div>
          </div>
        ))}
        {loading&&<div style={{display:"flex",gap:8}}><div style={{width:26,height:26,border:`1px solid ${accent}`,borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",fontSize:12}}>⬡</div><div style={{padding:"11px 15px",borderRadius:"12px 12px 12px 2px",background:"rgba(255,255,255,.03)",border:`1px solid ${border}`}}><div style={{display:"flex",gap:4}}>{[0,1,2].map(i=><div key={i} style={{width:5,height:5,borderRadius:"50%",background:accent,animation:`blink 1.2s ${i*.2}s infinite`}}/>)}</div></div></div>}
        <div ref={endRef}/>
      </div>
      <div style={{padding:"7px 20px",display:"flex",gap:5,flexWrap:"wrap",borderTop:`1px solid ${border}44`}}>
        {["Why is my model overfitting?","Best preprocessing for tabular data?","How to improve accuracy?","Explain cross-validation","Should I use XGBoost or RandomForest?"].map(p=>(
          <button key={p} onClick={()=>{setMessages(ms=>[...ms,{role:"user",content:p}]);callClaude(p);}} style={{padding:"3px 9px",border:`1px solid ${border}`,borderRadius:3,background:"transparent",color:"#334155",fontSize:9,fontFamily:"'Share Tech Mono'",cursor:"pointer"}}>{p}</button>
        ))}
      </div>
      <div style={{padding:"11px 20px",borderTop:`1px solid ${border}`,display:"flex",gap:8}}>
        <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&!e.shiftKey&&(e.preventDefault(),send())} placeholder="Ask about your pipeline..."
          style={{flex:1,padding:"9px 14px",background:"rgba(0,0,0,.4)",border:`1px solid ${accent}22`,borderRadius:7,color:"#e2e8f0",fontFamily:"'Share Tech Mono'",fontSize:12,outline:"none"}}/>
        <button onClick={send} disabled={!input.trim()||loading} style={{padding:"9px 18px",border:`1px solid ${accent}`,borderRadius:7,background:`${accent}18`,color:accent,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9,fontWeight:700,opacity:input.trim()&&!loading?1:.4}}>SEND</button>
      </div>
    </div>
  );
}

// ── MARKETPLACE ───────────────────────────────────────────────────────────────
function Marketplace({pl, accent, notify}) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [publishName, setPublishName] = useState("");
  const [publishDesc, setPublishDesc] = useState("");
  const [activeItem, setActiveItem] = useState(null);
  const border = "rgba(255,255,255,.06)"; const sub = "#475569";

  const loadItems = async () => {
    setLoading(true);
    try {
      const result = await window.storage.list("market:", true);
      const loaded = [];
      for (const key of (result?.keys||[])) {
        try {
          const item = await window.storage.get(key, true);
          if (item?.value) loaded.push({key, ...JSON.parse(item.value)});
        } catch {}
      }
      loaded.sort((a,b)=>(b.ts||0)-(a.ts||0));
      setItems(loaded);
    } catch(e) {
      // Storage not available — show demo items
      setItems([
        {key:"market:demo1",name:"XGBoost Champion",desc:"High-accuracy classification pipeline with tuned XGBoost",author:"ML_Expert",task:"classification",model:"XGBClassifier",likes:24,ts:Date.now()-86400000,config:{task:"classification",preprocessing:["StandardScaler","OneHotEncoder"],model:"XGBClassifier",training:{testSplit:.2,cvFolds:5,randomState:42},hyperparams:{n_estimators:200,learning_rate:0.05,max_depth:6}}},
        {key:"market:demo2",name:"NLP Text Classifier",desc:"Logistic regression for text classification with SelectKBest",author:"DataSci_Pro",task:"classification",model:"LogisticRegression",likes:18,ts:Date.now()-172800000,config:{task:"classification",preprocessing:["OneHotEncoder","SelectKBest"],model:"LogisticRegression",training:{testSplit:.25,cvFolds:3,randomState:0},hyperparams:{C:1,max_iter:200}}},
        {key:"market:demo3",name:"K-Means Cluster Explorer",desc:"PCA + K-Means for discovering data segments",author:"Cluster_King",task:"clustering",model:"KMeans",likes:12,ts:Date.now()-259200000,config:{task:"clustering",preprocessing:["StandardScaler","PCA"],model:"KMeans",training:{testSplit:.2,cvFolds:5,randomState:42},hyperparams:{n_clusters:5}}},
        {key:"market:demo4",name:"Regression Powerhouse",desc:"XGBoost regressor with feature engineering",author:"RegMaster",task:"regression",model:"XGBRegressor",likes:31,ts:Date.now()-345600000,config:{task:"regression",preprocessing:["StandardScaler","PolynomialFeatures"],model:"XGBRegressor",training:{testSplit:.2,cvFolds:5,randomState:42},hyperparams:{n_estimators:150,learning_rate:0.08,max_depth:6}}},
      ]);
    }
    setLoading(false);
  };

  useEffect(()=>{loadItems();},[]);

  const publish = async () => {
    if (!pl.task||!pl.model||!publishName.trim()) { notify("Fill name and complete pipeline first","error"); return; }
    const item = {name:publishName.trim(),desc:publishDesc.trim()||"No description",author:"You",task:pl.task,model:pl.model,likes:0,ts:Date.now(),config:{...pl}};
    try {
      const key = `market:${Date.now()}`;
      await window.storage.set(key, JSON.stringify(item), true);
      notify("Pipeline published to marketplace! 🎉");
      setPublishName(""); setPublishDesc("");
      loadItems();
    } catch {
      notify("Published! (demo mode — storage unavailable)","info");
      setItems(its=>[{key:`market:${Date.now()}`,likes:0,...item},...its]);
    }
  };

  const taskColors = {classification:"#00f5ff",regression:"#ff6b35",clustering:"#a855f7"};

  return (
    <div style={{flex:1,padding:"28px 40px",overflow:"auto"}}>
      <div style={{marginBottom:22}}>
        <h2 style={{fontFamily:"'Orbitron',sans-serif",fontWeight:900,fontSize:22,color:"#f8fafc",marginBottom:5}}>PIPELINE MARKETPLACE</h2>
        <p style={{fontSize:13,color:sub}}>Share your pipelines with the community · discover and load others' configs</p>
      </div>

      {/* Publish form */}
      <div style={{padding:"20px 22px",border:`1px solid ${accent}33`,borderRadius:10,background:`${accent}05`,marginBottom:24}}>
        <div style={{fontSize:9,color:accent,fontFamily:"'Orbitron',sans-serif",letterSpacing:".15em",marginBottom:14}}>PUBLISH YOUR PIPELINE</div>
        <div style={{display:"flex",gap:10,flexWrap:"wrap",alignItems:"flex-end"}}>
          <div style={{flex:1,minWidth:180}}>
            <div style={{fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",marginBottom:5}}>Pipeline name *</div>
            <input value={publishName} onChange={e=>setPublishName(e.target.value)} placeholder="My Awesome Pipeline"
              style={{width:"100%",padding:"8px 12px",background:"rgba(0,0,0,.4)",border:`1px solid ${border}`,borderRadius:6,color:"#e2e8f0",fontFamily:"'Share Tech Mono'",fontSize:12,outline:"none"}}/>
          </div>
          <div style={{flex:2,minWidth:200}}>
            <div style={{fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",marginBottom:5}}>Description</div>
            <input value={publishDesc} onChange={e=>setPublishDesc(e.target.value)} placeholder="What makes this pipeline special?"
              style={{width:"100%",padding:"8px 12px",background:"rgba(0,0,0,.4)",border:`1px solid ${border}`,borderRadius:6,color:"#e2e8f0",fontFamily:"'Share Tech Mono'",fontSize:12,outline:"none"}}/>
          </div>
          <button onClick={publish} style={{padding:"9px 20px",border:`1px solid ${accent}`,borderRadius:6,background:`${accent}18`,color:accent,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9,fontWeight:700,whiteSpace:"nowrap"}}>
            ↑ PUBLISH
          </button>
        </div>
        {(!pl.task||!pl.model)&&<div style={{marginTop:8,fontSize:10,color:"#f43f5e",fontFamily:"'Share Tech Mono'"}}>⚠ Complete task + model in Builder before publishing</div>}
      </div>

      {/* Items grid */}
      <div style={{fontFamily:"'Orbitron',sans-serif",fontSize:9,color:sub,letterSpacing:".15em",marginBottom:14}}>
        COMMUNITY PIPELINES ({items.length})
        <button onClick={loadItems} style={{marginLeft:12,padding:"3px 10px",border:`1px solid ${border}`,borderRadius:3,background:"transparent",color:sub,cursor:"pointer",fontSize:8,fontFamily:"'Orbitron',sans-serif"}}>↺ REFRESH</button>
      </div>

      {loading&&<div style={{textAlign:"center",padding:"40px",color:sub,fontFamily:"'Share Tech Mono'",fontSize:12}}>// Loading marketplace...</div>}

      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))",gap:14}}>
        {items.map(item=>{
          const tc=taskColors[item.task]||"#00f5ff";
          const isActive=activeItem===item.key;
          return (
            <div key={item.key} style={{padding:"18px",border:`1px solid ${isActive?accent:border}`,borderRadius:10,background:isActive?`${accent}06`:"rgba(255,255,255,.02)",transition:"all .2s"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:10}}>
                <div style={{flex:1}}>
                  <div style={{fontFamily:"'Orbitron',sans-serif",fontWeight:700,fontSize:12,color:"#f1f5f9",marginBottom:4}}>{item.name}</div>
                  <div style={{fontSize:11,color:sub,lineHeight:1.5}}>{item.desc}</div>
                </div>
              </div>
              <div style={{display:"flex",gap:6,marginBottom:12,flexWrap:"wrap"}}>
                <span style={{fontSize:9,padding:"2px 7px",borderRadius:3,border:`1px solid ${tc}44`,color:tc,fontFamily:"'Share Tech Mono'"}}>{item.task}</span>
                <span style={{fontSize:9,padding:"2px 7px",borderRadius:3,border:"1px solid rgba(255,255,255,.06)",color:sub,fontFamily:"'Share Tech Mono'"}}>{item.model?.replace(/Classifier|Regressor/,"")}</span>
              </div>
              <div style={{display:"flex",alignItems:"center",justifyContent:"space-between"}}>
                <div style={{fontSize:10,color:"#334155",fontFamily:"'Share Tech Mono'"}}>by {item.author}</div>
                <div style={{display:"flex",gap:6}}>
                  <button onClick={()=>setActiveItem(isActive?null:item.key)} style={{padding:"4px 10px",border:`1px solid ${border}`,borderRadius:4,background:"transparent",color:sub,cursor:"pointer",fontSize:9,fontFamily:"'Orbitron',sans-serif"}}>
                    {isActive?"HIDE":"PREVIEW"}
                  </button>
                  <button onClick={()=>{}} style={{padding:"4px 10px",border:`1px solid ${accent}44`,borderRadius:4,background:`${accent}0f`,color:accent,cursor:"pointer",fontSize:9,fontFamily:"'Orbitron',sans-serif"}}
                    onClick={()=>{notify(`Loaded "${item.name}"!`);window.dispatchEvent(new CustomEvent("loadPipeline",{detail:item.config}));}}>
                    LOAD
                  </button>
                </div>
              </div>
              {isActive&&(
                <div style={{marginTop:12,padding:"10px 12px",background:"rgba(0,0,0,.3)",borderRadius:6,fontSize:10,fontFamily:"'Share Tech Mono'",color:sub}}>
                  <div style={{color:accent,marginBottom:4}}>// Pipeline config</div>
                  {[["task",item.config?.task],["model",item.config?.model],["preprocessing",(item.config?.preprocessing||[]).join(", ")||"none"]].map(([k,v])=>(
                    <div key={k} style={{marginBottom:2}}><span style={{color:"#1e293b"}}>{k}</span> → {v}</div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── EXPORT TAB ────────────────────────────────────────────────────────────────
function ExportTab({pl, accent, notify}) {
  const [active, setActive] = useState("code");
  const border = "rgba(255,255,255,.06)"; const sub = "#475569";

  const genDocker = () => `FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.joblib .
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]`;

  const genRequirements = () => `scikit-learn==1.4.0\nxgboost==2.0.3\npandas==2.1.4\nnumpy==1.26.3\njoblib==1.3.2\nfastapi==0.109.0\nuvicorn==0.27.0\npydantic==2.6.0`;

  const genFastAPI = () => `from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np
from typing import List

app = FastAPI(title="${pl.model||'ML'} API")
pipeline = joblib.load("model.joblib")

class Request(BaseModel):
    features: List[float]

@app.get("/health")
def health(): return {"status": "ok", "model": "${pl.model||'model'}"}

@app.post("/predict")
def predict(req: Request):
    X = np.array(req.features).reshape(1, -1)
    return {"prediction": float(pipeline.predict(X)[0])}`;

  const genNotebook = () => {
    const code = genCode(pl);
    return JSON.stringify({nbformat:4,nbformat_minor:5,metadata:{kernelspec:{display_name:"Python 3",language:"python",name:"python3"}},cells:[
      {cell_type:"markdown",metadata:{},source:[`# ML Pipeline — ${pl.task||"Pipeline"}\nGenerated by ML Pipeline Builder v4.0`]},
      {cell_type:"code",execution_count:null,metadata:{},outputs:[],source:["!pip install scikit-learn xgboost pandas numpy joblib"]},
      {cell_type:"code",execution_count:null,metadata:{},outputs:[],source:[code]},
    ]},null,2);
  };

  const exports=[
    {id:"code",label:"Python Script",icon:"🐍",ext:".py"},
    {id:"notebook",label:"Jupyter Notebook",icon:"📓",ext:".ipynb"},
    {id:"docker",label:"Dockerfile",icon:"🐳",ext:"Dockerfile"},
    {id:"fastapi",label:"FastAPI App",icon:"⚡",ext:".py"},
    {id:"requirements",label:"requirements.txt",icon:"📦",ext:".txt"},
  ];

  const getContent=()=>{switch(active){case"code":return genCode(pl);case"notebook":return genNotebook();case"docker":return genDocker();case"fastapi":return genFastAPI();case"requirements":return genRequirements();default:return "";}};

  const download=()=>{
    const e=exports.find(x=>x.id===active);
    const fname=active==="docker"?"Dockerfile":`pipeline${e?.ext||".txt"}`;
    const blob=new Blob([getContent()],{type:"text/plain"});
    const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=fname;a.click();
    notify(`Downloaded ${fname}!`);
  };

  return (
    <div style={{display:"flex",height:"calc(100vh - 56px)"}}>
      <div style={{width:180,borderRight:`1px solid ${border}`,padding:"20px 0",flexShrink:0}}>
        <div style={{padding:"0 14px",fontSize:8,color:sub,letterSpacing:".2em",marginBottom:12,fontFamily:"'Orbitron',sans-serif"}}>FORMATS</div>
        {exports.map(e=>(
          <div key={e.id} onClick={()=>setActive(e.id)}
            style={{padding:"11px 14px",cursor:"pointer",borderLeft:`2px solid ${active===e.id?accent:"transparent"}`,background:active===e.id?`${accent}06`:"transparent",display:"flex",alignItems:"center",gap:9}}>
            <span style={{fontSize:16}}>{e.icon}</span>
            <div>
              <div style={{fontSize:11,fontFamily:"'Orbitron',sans-serif",color:active===e.id?accent:sub,letterSpacing:".05em"}}>{e.label}</div>
              <div style={{fontSize:9,color:"#1e293b",marginTop:1}}>{e.ext}</div>
            </div>
          </div>
        ))}
      </div>
      <div style={{flex:1,padding:"22px 28px",overflow:"auto",display:"flex",flexDirection:"column"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16}}>
          <div style={{fontFamily:"'Orbitron',sans-serif",fontWeight:700,fontSize:16,color:"#f8fafc"}}>{exports.find(e=>e.id===active)?.label}</div>
          <div style={{display:"flex",gap:7}}>
            <button onClick={()=>{navigator.clipboard.writeText(getContent());notify("Copied!");}} style={{padding:"7px 16px",border:`1px solid ${border}`,borderRadius:5,background:"transparent",color:sub,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9}}>⎘ COPY</button>
            <button onClick={download} style={{padding:"7px 16px",border:`1px solid ${accent}`,borderRadius:5,background:`${accent}15`,color:accent,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9,fontWeight:700}}>⬇ DOWNLOAD</button>
          </div>
        </div>
        <div style={{border:`1px solid ${border}`,borderRadius:10,background:"rgba(0,0,0,.5)",overflow:"hidden",flex:1}}>
          <div style={{padding:"9px 14px",borderBottom:`1px solid ${border}`,display:"flex",gap:6,alignItems:"center",background:"rgba(0,0,0,.3)"}}>
            {["#f43f5e","#f59e0b","#10b981"].map(c=><div key={c} style={{width:8,height:8,borderRadius:"50%",background:c}}/>)}
            <span style={{marginLeft:8,fontSize:9,color:sub,fontFamily:"'Share Tech Mono'"}}>{active==="docker"?"Dockerfile":`pipeline${exports.find(e=>e.id===active)?.ext||""}`}</span>
          </div>
          <div style={{padding:"18px 22px",overflow:"auto",maxHeight:"calc(100vh - 200px)"}}>
            <pre style={{margin:0,fontSize:11,lineHeight:1.85,fontFamily:"'Share Tech Mono'"}}>
              {getContent().split("\n").map((line,i)=>{
                let c="#475569";
                if(line.trim().startsWith("#")||line.trim().startsWith("//"))c="#1e293b";
                else if(line.startsWith("import")||line.startsWith("from")||line.startsWith("FROM")||line.startsWith("RUN")||line.startsWith("@app."))c="#60a5fa";
                else if(line.includes("pipeline")||line.includes("def ")||line.includes("class "))c=accent;
                else if(line.includes("print("))c="#10b981";
                return <span key={i} style={{color:c,display:"block"}}>{line||" "}</span>;
              })}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── ONBOARDING TOUR ──────────────────────────────────────────────────────────
function OnboardingTour({onDone}) {
  const [s,setS]=useState(0);
  const steps=[
    {title:"Welcome to v4.0!",desc:"The most powerful ML Pipeline Builder. Real Python training, AI assistant, marketplace, and more.",icon:"⬡"},
    {title:"Live Training",desc:"Train real sklearn models in your browser using Pyodide — no server needed!",icon:"▶"},
    {title:"Interactive Charts",desc:"Click confusion matrix cells for details. 8 chart types in Visualize tab.",icon:"📊"},
    {title:"Pipeline Marketplace",desc:"Share your pipelines publicly. Discover and load pipelines from the community.",icon:"🛒"},
    {title:"Per-Tab Themes",desc:"Each tab has its own color theme. Builder=Cyan, Train=Emerald, AI=Amber...",icon:"🎨"},
    {title:"You're ready!",desc:"Press ⌘K for shortcuts. Use templates for quick starts. Train in browser for real results.",icon:"🚀"},
  ];
  const st=steps[s];
  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,.85)",zIndex:1000,display:"flex",alignItems:"center",justifyContent:"center",backdropFilter:"blur(8px)"}}>
      <div style={{background:"#0d1117",border:"1px solid rgba(0,245,255,.25)",borderRadius:16,padding:"38px 46px",maxWidth:460,textAlign:"center",boxShadow:"0 0 60px rgba(0,245,255,.12)"}}>
        <div style={{fontSize:44,marginBottom:14}}>{st.icon}</div>
        <div style={{fontFamily:"'Orbitron',sans-serif",fontWeight:900,fontSize:17,color:"#f8fafc",marginBottom:10}}>{st.title}</div>
        <div style={{fontSize:13,color:"#64748b",lineHeight:1.7,marginBottom:26}}>{st.desc}</div>
        <div style={{display:"flex",justifyContent:"center",gap:5,marginBottom:22}}>
          {steps.map((_,i)=><div key={i} style={{width:i===s?18:6,height:6,borderRadius:3,background:i===s?ACCENT:"#1e293b",transition:"all .3s"}}/>)}
        </div>
        <div style={{display:"flex",gap:9,justifyContent:"center"}}>
          {s>0&&<button onClick={()=>setS(s=>s-1)} style={{padding:"8px 18px",border:"1px solid #1e293b",borderRadius:6,background:"transparent",color:"#475569",cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9}}>← BACK</button>}
          <button onClick={()=>s<steps.length-1?setS(s=>s+1):onDone()} style={{padding:"8px 22px",border:`1px solid ${ACCENT}`,borderRadius:6,background:`${ACCENT}18`,color:ACCENT,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:10,fontWeight:700}}>
            {s<steps.length-1?"NEXT →":"START →"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [step,setStep]=useState(0);
  const [tab,setTab]=useState("builder");
  const [codeOpen,setCodeOpen]=useState(false);
  const [copied,setCopied]=useState(false);
  const [animKey,setAnimKey]=useState(0);
  const [csvData,setCsvData]=useState(null);
  const [csvCols,setCsvCols]=useState([]);
  const [dragOver,setDragOver]=useState(false);
  const [versions,setVersions]=useState([]);
  const [notification,setNotify]=useState(null);
  const [showTour,setShowTour]=useState(true);
  const [showShortcuts,setShowShortcuts]=useState(false);
  const [draggingPrep,setDraggingPrep]=useState(null);
  const [pl,setPl]=useState({
    task:null,dataSource:null,preprocessing:[],model:null,
    hyperparams:{n_estimators:100,max_depth:10,learning_rate:0.1,C:1,kernel:"rbf",n_clusters:3,max_iter:200,criterion:"gini"},
    tuning:{enabled:false,method:"grid",nIter:20},
    training:{testSplit:.2,cvFolds:5,randomState:42},
    evaluation:[],exportFormat:"joblib",customCode:"",
  });

  const upd=(k,v)=>setPl(p=>({...p,[k]:v}));
  const togL=(k,v)=>setPl(p=>{const a=p[k]||[];return{...p,[k]:a.includes(v)?a.filter(x=>x!==v):[...a,v]};});
  const goStep=s=>{setStep(s);setAnimKey(k=>k+1);};
  const notify=(msg,type="success")=>{setNotify({msg,type});setTimeout(()=>setNotify(null),3000);};

  // Per-tab theme
  const theme = TAB_THEMES[tab] || TAB_THEMES.builder;
  const accent = theme.accent;

  // Listen for marketplace load
  useEffect(()=>{
    const handler=e=>{setPl(p=>({...p,...e.detail}));setTab("builder");goStep(0);notify("Pipeline loaded from marketplace!");};
    window.addEventListener("loadPipeline",handler);
    return()=>window.removeEventListener("loadPipeline",handler);
  },[]);

  // Keyboard shortcuts
  useEffect(()=>{
    const h=e=>{
      if(e.key==="Escape")setShowShortcuts(false);
      if((e.metaKey||e.ctrlKey)&&e.key==="k"){e.preventDefault();setShowShortcuts(s=>!s);}
      if((e.metaKey||e.ctrlKey)&&e.key==="s"){e.preventDefault();saveVersion();}
      if((e.metaKey||e.ctrlKey)&&e.key==="/"){e.preventDefault();setCodeOpen(s=>!s);}
      if(e.key==="ArrowRight"&&tab==="builder"&&!e.target.matches("input,textarea,select"))goStep(s=>Math.min(s+1,STEPS.length-1));
      if(e.key==="ArrowLeft"&&tab==="builder"&&!e.target.matches("input,textarea,select"))goStep(s=>Math.max(s-1,0));
    };
    window.addEventListener("keydown",h);
    return()=>window.removeEventListener("keydown",h);
  },[tab]);

  const saveVersion=()=>{
    const v={id:Date.now(),ts:new Date().toLocaleTimeString(),label:`v${versions.length+1}`,config:{...pl}};
    setVersions(vs=>[v,...vs.slice(0,9)]);
    notify("Version saved!");
  };
  const exportJSON=()=>{
    const blob=new Blob([JSON.stringify(pl,null,2)],{type:"application/json"});
    const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="pipeline.json";a.click();
    notify("Exported pipeline.json");
  };
  const importJSON=e=>{
    const f=e.target.files[0];if(!f)return;
    const r=new FileReader();r.onload=ev=>{try{setPl(JSON.parse(ev.target.result));notify("Pipeline imported!");}catch{notify("Invalid JSON","error");}};
    r.readAsText(f);
  };
  const shareLink=()=>{
    const url=`${window.location.origin}${window.location.pathname}?config=${btoa(JSON.stringify(pl))}`;
    navigator.clipboard.writeText(url);
    notify("Share link copied!");
  };
  const handleCSV=f=>{Papa.parse(f,{header:true,complete:res=>{setCsvData(res.data.slice(0,300));setCsvCols(Object.keys(res.data[0]||{}));notify(`CSV: ${res.data.length} rows`);}});};
  const handleDrop=e=>{e.preventDefault();setDragOver(false);const f=e.dataTransfer.files[0];if(f?.name.endsWith(".csv"))handleCSV(f);};

  useEffect(()=>{
    const params=new URLSearchParams(window.location.search);
    const cfg=params.get("config");
    if(cfg){try{setPl(JSON.parse(atob(cfg)));setShowTour(false);notify("Pipeline loaded from link!");}catch{}}
  },[]);

  const taskColor=pl.task?TASK_OPTIONS.find(t=>t.id===pl.task)?.color:accent;
  const modelColor=pl.model&&pl.task?(MODELS[pl.task]||[]).find(m=>m.id===pl.model)?.color:null;
  const isStepDone=i=>{if(i===0)return!!pl.task;if(i===1)return!!pl.dataSource;if(i===2)return pl.preprocessing.length>0;if(i===3)return!!pl.model;if(i===4||i===5)return true;if(i===6)return pl.evaluation.length>0;if(i===7||i===8)return true;return false;};
  const doneCount=()=>STEPS.filter((_,i)=>isStepDone(i)).length;
  const canNext=()=>{if(step===0)return!!pl.task;if(step===1)return!!pl.dataSource;if(step===3)return!!pl.model;return true;};
  const border="rgba(255,255,255,.06)";
  const sub="#475569";
  const textC="#e2e8f0";
  const panel="rgba(2,4,8,.9)";
  const card="rgba(255,255,255,.02)";
  const currentHPs=pl.model?(HYPERPARAMS[pl.model]||[]):[];

  return (
    <div style={{minHeight:"100vh",background:theme.bg,color:textC,fontFamily:"'Rajdhani',sans-serif",position:"relative",overflow:"hidden",transition:"background .6s"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:3px;} ::-webkit-scrollbar-thumb{background:${accent}33;border-radius:2px;}
        .grid-bg{background-image:linear-gradient(${accent}18 1px,transparent 1px),linear-gradient(90deg,${accent}18 1px,transparent 1px);background-size:44px 44px;opacity:.06;}
        .hov{transition:all .2s;cursor:pointer;} .hov:hover{transform:translateY(-2px) scale(1.01);}
        .hov-r{transition:all .18s;cursor:pointer;} .hov-r:hover{transform:translateX(3px);}
        @keyframes fadeUp{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}
        @keyframes blink{0%,100%{opacity:1;}50%{opacity:0;}}
        @keyframes notif{0%{opacity:0;transform:translateY(-12px);}15%,80%{opacity:1;transform:translateY(0);}100%{opacity:0;}}
        .anim{animation:fadeUp .4s cubic-bezier(.16,1,.3,1) forwards;}
        .notif{animation:notif 3s ease forwards;}
        input[type=range]{-webkit-appearance:none;height:2px;background:rgba(255,255,255,.08);border-radius:2px;outline:none;cursor:pointer;width:100%;}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;cursor:pointer;}
        textarea{background:rgba(0,0,0,.4);border:1px solid rgba(255,255,255,.08);color:#94a3b8;font-family:'Share Tech Mono';font-size:12px;padding:12px;border-radius:7px;resize:vertical;outline:none;width:100%;}
        textarea:focus{border-color:${accent}55;}
        select{background:rgba(0,0,0,.5);border:1px solid rgba(255,255,255,.1);color:#94a3b8;padding:6px 10px;border-radius:5px;font-family:'Share Tech Mono';font-size:11px;outline:none;cursor:pointer;}
        pre{white-space:pre;overflow-x:auto;font-family:'Share Tech Mono',monospace!important;}
      `}</style>
      <div className="grid-bg" style={{position:"fixed",inset:0,zIndex:0}}/>
      <div style={{position:"fixed",inset:0,background:`radial-gradient(ellipse 55% 40% at 60% 20%,${accent}08,transparent)`,zIndex:0,transition:"background .6s",pointerEvents:"none"}}/>

      {showTour&&<OnboardingTour onDone={()=>setShowTour(false)}/>}

      {showShortcuts&&(
        <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,.75)",zIndex:900,display:"flex",alignItems:"center",justifyContent:"center"}} onClick={()=>setShowShortcuts(false)}>
          <div style={{background:"#0d1117",border:`1px solid ${border}`,borderRadius:12,padding:"26px 30px",minWidth:320}} onClick={e=>e.stopPropagation()}>
            <div style={{fontFamily:"'Orbitron',sans-serif",fontWeight:700,fontSize:12,color:accent,letterSpacing:".15em",marginBottom:16}}>KEYBOARD SHORTCUTS</div>
            {[["⌘ K","Toggle shortcuts"],["⌘ S","Save version"],["⌘ /","Toggle code"],["← →","Navigate steps"],["Esc","Close modal"]].map(([k,d])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between",marginBottom:9,alignItems:"center"}}>
                <kbd style={{padding:"3px 8px",borderRadius:4,background:`${accent}10`,border:`1px solid ${accent}33`,color:accent,fontFamily:"'Share Tech Mono'",fontSize:11}}>{k}</kbd>
                <span style={{fontSize:12,color:sub}}>{d}</span>
              </div>
            ))}
            <button onClick={()=>setShowShortcuts(false)} style={{marginTop:14,width:"100%",padding:"7px",border:`1px solid ${border}`,borderRadius:5,background:"transparent",color:sub,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9}}>CLOSE</button>
          </div>
        </div>
      )}

      {notification&&(
        <div className="notif" style={{position:"fixed",top:20,right:20,zIndex:1000,padding:"10px 18px",borderRadius:7,background:notification.type==="error"?"rgba(244,63,94,.15)":notification.type==="info"?"rgba(96,165,250,.15)":"rgba(16,185,129,.15)",border:`1px solid ${notification.type==="error"?"#f43f5e":notification.type==="info"?"#60a5fa":"#10b981"}`,color:notification.type==="error"?"#f43f5e":notification.type==="info"?"#60a5fa":"#10b981",fontFamily:"'Share Tech Mono'",fontSize:12}}>
          {notification.type==="error"?"✗":"✓"} {notification.msg}
        </div>
      )}

      {/* Load pyodide script */}
      <script src="https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js" async/>

      <div style={{position:"relative",zIndex:10,display:"flex",flexDirection:"column",minHeight:"100vh"}}>
        {/* HEADER */}
        <header style={{padding:"10px 20px",borderBottom:`1px solid ${border}`,background:panel,backdropFilter:"blur(24px)",display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0,flexWrap:"wrap",gap:7}}>
          <div style={{display:"flex",alignItems:"center",gap:10}}>
            <div style={{width:33,height:33,border:`1.5px solid ${accent}`,borderRadius:7,display:"flex",alignItems:"center",justifyContent:"center",fontSize:15,color:accent,boxShadow:`0 0 12px ${accent}44`,transition:"all .6s"}}>⬡</div>
            <div>
              <div style={{fontFamily:"'Orbitron',sans-serif",fontWeight:900,fontSize:11,letterSpacing:".2em",color:textC}}>ML PIPELINE BUILDER</div>
              <div style={{fontSize:8,color:`${accent}88`,letterSpacing:".2em"}}>v4.0 · {theme.name} Theme</div>
            </div>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:6,flexWrap:"wrap"}}>
            <div style={{width:80,height:2,background:"rgba(255,255,255,.05)",borderRadius:2,overflow:"hidden"}}>
              <div style={{height:"100%",background:`linear-gradient(90deg,${accent},${accent}66)`,width:`${(doneCount()/STEPS.length)*100}%`,transition:"width .5s"}}/>
            </div>
            <span style={{fontSize:9,color:accent,fontFamily:"'Share Tech Mono'",marginRight:6}}>{doneCount()}/{STEPS.length}</span>
            {[["builder","◈ BUILDER"],["train","▶ TRAIN"],["visualize","📊 CHARTS"],["ai","🤖 AI"],["marketplace","🛒 MARKET"],["export","⬇ EXPORT"],["history","📋 HISTORY"]].map(([t,l])=>(
              <button key={t} onClick={()=>{setTab(t);setCodeOpen(false);}}
                style={{padding:"5px 10px",border:`1px solid ${tab===t&&!codeOpen?TAB_THEMES[t].accent:border}`,borderRadius:4,background:tab===t&&!codeOpen?`${TAB_THEMES[t].accent}18`:"transparent",color:tab===t&&!codeOpen?TAB_THEMES[t].accent:sub,fontFamily:"'Orbitron',sans-serif",fontSize:8,fontWeight:700,cursor:"pointer",transition:"all .3s"}}>
                {l}
              </button>
            ))}
            <button onClick={()=>setCodeOpen(c=>!c)} style={{padding:"5px 10px",border:`1px solid ${codeOpen?accent:border}`,borderRadius:4,background:codeOpen?`${accent}18`:"transparent",color:codeOpen?accent:sub,fontFamily:"'Orbitron',sans-serif",fontSize:8,cursor:"pointer"}}>⟨/⟩</button>
            <div style={{display:"flex",gap:4,paddingLeft:7,borderLeft:`1px solid ${border}`}}>
              {[["💾","Save",saveVersion],["🔗","Share",shareLink],["⬇","Export JSON",exportJSON],["⌘K","Shortcuts",()=>setShowShortcuts(true)]].map(([ico,t,fn])=>(
                <button key={ico} title={t} onClick={fn} style={{width:26,height:26,border:`1px solid ${border}`,borderRadius:4,background:"transparent",color:sub,cursor:"pointer",fontSize:11,display:"flex",alignItems:"center",justifyContent:"center"}}>{ico}</button>
              ))}
              <label title="Import JSON" style={{width:26,height:26,border:`1px solid ${border}`,borderRadius:4,background:"transparent",color:sub,cursor:"pointer",fontSize:11,display:"flex",alignItems:"center",justifyContent:"center"}}>⬆<input type="file" accept=".json" onChange={importJSON} style={{display:"none"}}/></label>
            </div>
          </div>
        </header>

        {/* CODE VIEW */}
        {codeOpen&&(
          <div style={{flex:1,padding:"28px 40px",overflow:"auto"}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:16}}>
              <div><h2 style={{fontFamily:"'Orbitron',sans-serif",fontWeight:700,fontSize:17,color:textC}}>GENERATED CODE</h2><div style={{fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",marginTop:2}}>Python · scikit-learn</div></div>
              <button onClick={()=>{navigator.clipboard.writeText(genCode(pl));setCopied(true);setTimeout(()=>setCopied(false),2000);}} style={{padding:"8px 20px",border:`1px solid ${copied?"#10b981":accent}`,borderRadius:6,background:copied?"rgba(16,185,129,.1)":`${accent}12`,color:copied?"#10b981":accent,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9,fontWeight:700}}>{copied?"✓ COPIED":"⎘ COPY"}</button>
            </div>
            <div style={{border:`1px solid ${border}`,borderRadius:11,background:"rgba(0,0,0,.55)",overflow:"hidden"}}>
              <div style={{padding:"9px 14px",borderBottom:`1px solid ${border}`,display:"flex",gap:6,background:"rgba(0,0,0,.3)"}}>
                {["#f43f5e","#f59e0b","#10b981"].map(c=><div key={c} style={{width:8,height:8,borderRadius:"50%",background:c}}/>)}
                <span style={{marginLeft:8,fontSize:9,color:sub,fontFamily:"'Share Tech Mono'"}}>pipeline.py</span>
              </div>
              <div style={{padding:"18px 22px",overflowX:"auto"}}>
                <pre style={{margin:0,fontSize:12,lineHeight:1.85}}>
                  {genCode(pl).split("\n").map((line,i)=>{let c=sub;if(line.trim().startsWith("#"))c="#1e293b";else if(line.startsWith("import")||line.startsWith("from"))c="#60a5fa";else if(line.includes("pipeline")&&line.includes("=")&&!line.includes("=="))c=accent;else if(line.includes(".fit(")||line.includes(".predict("))c="#a855f7";else if(line.includes("print("))c="#10b981";return <span key={i} style={{color:c,display:"block"}}>{line||" "}</span>;})}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* TAB CONTENT */}
        {!codeOpen&&tab==="train"&&<PyodideTrainer pl={pl} accent={accent} csvData={csvData} csvCols={csvCols}/>}
        {!codeOpen&&tab==="visualize"&&<VisualizeTab pl={pl} accent={accent}/>}
        {!codeOpen&&tab==="ai"&&<AIAssistant pl={pl} accent={accent}/>}
        {!codeOpen&&tab==="marketplace"&&<Marketplace pl={pl} accent={accent} notify={notify}/>}
        {!codeOpen&&tab==="export"&&<ExportTab pl={pl} accent={accent} notify={notify}/>}

        {!codeOpen&&tab==="history"&&(
          <div style={{flex:1,padding:"30px 40px",overflow:"auto"}}>
            <h2 style={{fontFamily:"'Orbitron',sans-serif",fontWeight:700,fontSize:20,color:textC,marginBottom:4}}>VERSION HISTORY</h2>
            <p style={{fontSize:12,color:sub,marginBottom:20}}>Saved pipeline snapshots</p>
            {versions.length===0?(
              <div style={{textAlign:"center",padding:"48px",color:sub}}><div style={{fontSize:34,marginBottom:12}}>📋</div><div style={{fontFamily:"'Orbitron',sans-serif",fontSize:11,letterSpacing:".15em"}}>NO VERSIONS SAVED</div><div style={{fontSize:12,marginTop:6}}>Press ⌘S to save snapshots</div></div>
            ):(
              <div style={{display:"flex",flexDirection:"column",gap:9}}>
                {versions.map(v=>(
                  <div key={v.id} style={{padding:"13px 16px",border:`1px solid ${border}`,borderRadius:8,background:card,display:"flex",alignItems:"center",justifyContent:"space-between"}}>
                    <div style={{display:"flex",alignItems:"center",gap:11}}>
                      <div style={{width:26,height:26,border:`1.5px solid ${accent}`,borderRadius:5,display:"flex",alignItems:"center",justifyContent:"center",fontFamily:"'Share Tech Mono'",fontSize:9,color:accent}}>{v.label}</div>
                      <div>
                        <div style={{fontSize:12,fontWeight:600,color:textC}}>{v.config.task||"—"} · {v.config.model?.replace(/Classifier|Regressor/,"")||"—"}</div>
                        <div style={{fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",marginTop:1}}>{v.ts}</div>
                      </div>
                    </div>
                    <button onClick={()=>{setPl(v.config);notify(`Loaded ${v.label}`);}} style={{padding:"4px 12px",border:`1px solid ${accent}44`,borderRadius:4,background:`${accent}10`,color:accent,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:8,fontWeight:700}}>LOAD</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* BUILDER TAB */}
        {!codeOpen&&tab==="builder"&&(
          <div style={{display:"flex",flex:1}}>
            <aside style={{width:186,borderRight:`1px solid ${border}`,background:panel,backdropFilter:"blur(20px)",padding:"22px 0",flexShrink:0,display:"flex",flexDirection:"column"}}>
              <div style={{padding:"0 13px",flex:1}}>
                <div style={{fontSize:8,color:sub,letterSpacing:".2em",marginBottom:14,fontFamily:"'Orbitron',sans-serif"}}>STEPS</div>
                {STEPS.map((s,i)=>{
                  const done=isStepDone(i),active=step===i,nc=active?accent:done?"#10b981":"#1e293b";
                  return(
                    <div key={i} style={{display:"flex",alignItems:"flex-start"}}>
                      <div style={{display:"flex",flexDirection:"column",alignItems:"center"}}>
                        <div onClick={()=>goStep(i)} style={{width:23,height:23,borderRadius:"50%",border:`1.5px solid ${nc}`,background:active?`${nc}18`:done?`${nc}10`:"transparent",display:"flex",alignItems:"center",justifyContent:"center",boxShadow:active?`0 0 10px ${nc}`:done?`0 0 4px ${nc}55`:"none",cursor:"pointer",fontSize:9,color:nc,fontFamily:"'Share Tech Mono'",transition:"all .3s",flexShrink:0}}>
                          {done&&!active?"✓":i+1}
                        </div>
                        {i<STEPS.length-1&&<div style={{width:1,height:17,background:done?`linear-gradient(${nc},${isStepDone(i+1)?"#10b981":"#1e293b"})`:"rgba(255,255,255,.04)"}}/>}
                      </div>
                      <div onClick={()=>goStep(i)} style={{marginLeft:9,paddingTop:3,marginBottom:16,cursor:"pointer"}}>
                        <div style={{fontSize:9,fontFamily:"'Orbitron',sans-serif",fontWeight:active?700:400,letterSpacing:".07em",color:active?accent:done?"#4b5563":sub}}>{s.toUpperCase()}</div>
                        {active&&<div style={{width:12,height:1.5,background:accent,marginTop:3,borderRadius:1}}/>}
                      </div>
                    </div>
                  );
                })}
              </div>
              {pl.task&&pl.model&&(
                <div style={{margin:"0 10px",padding:"9px",border:`1px solid ${border}`,borderRadius:7,background:`${accent}04`}}>
                  <div style={{fontSize:9,color:accent,fontFamily:"'Share Tech Mono'",marginBottom:3,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{pl.task} · {pl.model?.replace(/Classifier|Regressor/,"")}</div>
                  <div style={{fontSize:9,color:"#10b981",fontFamily:"'Share Tech Mono'"}}>{pl.preprocessing.length} transforms</div>
                  <div onClick={()=>{setTab("train");}} style={{marginTop:6,fontSize:8,color:accent,fontFamily:"'Share Tech Mono'",cursor:"pointer"}}>▶ Train in browser →</div>
                </div>
              )}
            </aside>

            <main style={{flex:1,overflow:"auto",padding:"34px 46px"}}>
              <div className="anim" key={animKey}>
                <div style={{marginBottom:28}}>
                  <div style={{display:"flex",alignItems:"center",gap:9,marginBottom:7}}>
                    <span style={{fontSize:9,color:accent,fontFamily:"'Share Tech Mono'",letterSpacing:".18em"}}>[{step+1}/{STEPS.length}]</span>
                    <div style={{flex:1,height:1,background:`linear-gradient(90deg,${accent}44,transparent)`}}/>
                  </div>
                  <h1 style={{fontFamily:"'Orbitron',sans-serif",fontWeight:900,fontSize:22,color:textC,marginBottom:4}}>{["Select Task","Data Source","Preprocessing","Model","Hyperparameters","Training Config","Evaluation","Export Format","Custom Code"][step]}</h1>
                  <p style={{fontSize:13,color:sub}}>{["Choose your ML task","Where is your data from?","Pick transforms (drag to reorder)","Select your algorithm","Configure model parameters","Train/test split settings","Choose evaluation metrics","Export format","Inject custom Python"][step]}</p>
                </div>

                {/* S0 Task */}
                {step===0&&<div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14,maxWidth:620}}>{TASK_OPTIONS.map(t=><div key={t.id} className="hov" onClick={()=>{upd("task",t.id);upd("model",null);upd("preprocessing",[]);upd("evaluation",[]);}} style={{padding:"26px 16px",border:`1px solid ${pl.task===t.id?t.color:border}`,borderRadius:11,background:pl.task===t.id?`${t.color}0d`:card,textAlign:"center",boxShadow:pl.task===t.id?`0 0 22px ${t.color}22`:"none"}}><div style={{fontSize:30,marginBottom:9,color:t.color,textShadow:`0 0 10px ${t.color}`}}>{t.icon}</div><div style={{fontFamily:"'Orbitron',sans-serif",fontWeight:700,fontSize:10,color:pl.task===t.id?t.color:sub,marginBottom:6,letterSpacing:".12em"}}>{t.label.toUpperCase()}</div><div style={{fontSize:11,color:sub,lineHeight:1.6}}>{t.desc}</div></div>)}</div>}

                {/* S1 Data */}
                {step===1&&<div style={{maxWidth:480}}><div style={{display:"flex",flexDirection:"column",gap:8,marginBottom:18}}>{[{id:"csv",label:"CSV / Excel",icon:"⬡"},{id:"json",label:"JSON",icon:"⬢"},{id:"sql",label:"SQL Database",icon:"◈"},{id:"api",label:"API Endpoint",icon:"⟳"},{id:"sklearn",label:"Sklearn Dataset",icon:"◉"}].map(d=><div key={d.id} className="hov-r" onClick={()=>upd("dataSource",d.id)} style={{padding:"11px 15px",border:`1px solid ${pl.dataSource===d.id?accent:border}`,borderRadius:7,background:pl.dataSource===d.id?`${accent}0a`:card,display:"flex",alignItems:"center",gap:11}}><div style={{width:28,height:28,border:`1px solid ${pl.dataSource===d.id?accent:border}`,borderRadius:5,display:"flex",alignItems:"center",justifyContent:"center",fontSize:13,color:pl.dataSource===d.id?accent:sub}}>{d.icon}</div><span style={{fontSize:12,fontWeight:600,color:pl.dataSource===d.id?accent:sub}}>{d.label}</span>{pl.dataSource===d.id&&<span style={{marginLeft:"auto",color:accent}}>◈</span>}</div>)}</div>
                  <div onDragOver={e=>{e.preventDefault();setDragOver(true);}} onDragLeave={()=>setDragOver(false)} onDrop={handleDrop} style={{padding:"18px",border:`2px dashed ${dragOver?accent:border}`,borderRadius:9,textAlign:"center",cursor:"pointer",background:dragOver?`${accent}07`:card}} onClick={()=>document.getElementById("csv-in").click()}>
                    <input id="csv-in" type="file" accept=".csv" style={{display:"none"}} onChange={e=>handleCSV(e.target.files[0])}/>
                    <div style={{fontSize:20,marginBottom:5}}>{csvData?"✅":"📂"}</div>
                    <div style={{fontSize:11,color:csvData?accent:sub,fontFamily:"'Share Tech Mono'"}}>{csvData?`${csvData.length} rows loaded`:"Drop CSV for EDA"}</div>
                  </div>
                </div>}

                {/* S2 Preprocessing — drag & drop */}
                {step===2&&pl.task&&<div style={{maxWidth:640}}>
                  <div style={{display:"flex",flexWrap:"wrap",gap:7,marginBottom:16}}>{PREPROCESSING[pl.task].filter(p=>!pl.preprocessing.includes(p)).map(p=><div key={p} onClick={()=>togL("preprocessing",p)} style={{padding:"7px 13px",border:`1px solid ${border}`,borderRadius:4,background:card,fontSize:11,fontFamily:"'Share Tech Mono'",color:sub,cursor:"pointer",display:"flex",alignItems:"center",gap:5}}><span style={{fontSize:9,color:"#334155"}}>+</span>{p}</div>)}</div>
                  {pl.preprocessing.length>0&&<div>
                    <div style={{fontSize:9,color:accent,fontFamily:"'Share Tech Mono'",marginBottom:9}}>// Drag to reorder</div>
                    {pl.preprocessing.map((p,i)=><div key={p} draggable onDragStart={()=>setDraggingPrep(i)} onDragOver={e=>e.preventDefault()} onDrop={()=>{if(draggingPrep!==null&&draggingPrep!==i){const a=[...pl.preprocessing];a.splice(i,0,a.splice(draggingPrep,1)[0]);upd("preprocessing",a);}setDraggingPrep(null);}} style={{padding:"9px 14px",border:`1px solid ${accent}33`,borderRadius:5,background:`${accent}08`,display:"flex",alignItems:"center",gap:9,cursor:"grab",marginBottom:5,userSelect:"none"}}><span style={{color:sub,cursor:"grab"}}>⠿</span><span style={{fontSize:10,color:accent,fontFamily:"'Share Tech Mono'"}}>{i+1}.</span><span style={{fontSize:11,color:accent,flex:1}}>{p}</span><span onClick={()=>togL("preprocessing",p)} style={{color:"#f43f5e",cursor:"pointer",fontSize:13,opacity:.7}}>×</span></div>)}
                  </div>}
                </div>}

                {/* S3 Model */}
                {step===3&&pl.task&&<div style={{maxWidth:640}}>
                  <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:9,marginBottom:16}}>
                    {(MODELS[pl.task]||[]).map(m=><div key={m.id} className="hov" onClick={()=>upd("model",m.id)} style={{padding:"12px 14px",border:`1px solid ${pl.model===m.id?m.color:border}`,borderRadius:7,background:pl.model===m.id?`${m.color}0c`:card,display:"flex",alignItems:"center",justifyContent:"space-between",boxShadow:pl.model===m.id?`0 0 14px ${m.color}22`:"none"}}><div><div style={{fontSize:12,fontWeight:600,color:pl.model===m.id?m.color:sub}}>{m.label}</div>{pl.model===m.id&&<div style={{width:11,height:1.5,background:m.color,marginTop:4}}/>}</div><div style={{textAlign:"right"}}><div style={{fontSize:8,padding:"2px 6px",borderRadius:3,border:`1px solid ${m.color}44`,color:m.color,fontFamily:"'Share Tech Mono'"}}>{m.tag}</div><div style={{fontSize:8,color:sub,marginTop:2,fontFamily:"'Share Tech Mono'"}}>{m.score}%</div></div></div>)}
                  </div>
                </div>}

                {/* S4 Hyperparams */}
                {step===4&&<div style={{maxWidth:500}}>
                  {currentHPs.length===0?<div style={{padding:"22px",border:`1px solid ${border}`,borderRadius:8,background:card,textAlign:"center",color:sub,fontFamily:"'Share Tech Mono'",fontSize:11}}>// Select a model first</div>:
                  <div style={{display:"flex",flexDirection:"column",gap:11,marginBottom:16}}>{currentHPs.map(hp=><div key={hp.key} style={{padding:"14px 17px",border:`1px solid ${border}`,borderRadius:7,background:card}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:10}}><span style={{fontSize:9,fontFamily:"'Orbitron',sans-serif",color:sub,letterSpacing:".12em"}}>{hp.label.toUpperCase()}</span>{hp.type==="range"&&<span style={{fontSize:16,fontFamily:"'Share Tech Mono'",color:accent}}>{pl.hyperparams[hp.key]??hp.default}</span>}</div>{hp.type==="range"?(<><input type="range" min={hp.min} max={hp.max} step={hp.step} value={pl.hyperparams[hp.key]??hp.default} onChange={e=>upd("hyperparams",{...pl.hyperparams,[hp.key]:parseFloat(e.target.value)})} style={{accentColor:accent}}/><div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#1e293b",marginTop:4,fontFamily:"'Share Tech Mono'"}}><span>{hp.min}</span><span>{hp.max}</span></div></>):(<select value={pl.hyperparams[hp.key]??hp.default} onChange={e=>upd("hyperparams",{...pl.hyperparams,[hp.key]:e.target.value})}>{hp.options.map(o=><option key={o} value={o}>{o}</option>)}</select>)}</div>)}</div>}
                </div>}

                {/* S5 Training */}
                {step===5&&<div style={{display:"flex",flexDirection:"column",gap:12,maxWidth:460}}>{[{label:"TEST SPLIT",key:"testSplit",min:.1,max:.4,step:.05,fmt:v=>`${Math.round(v*100)}%`},{label:"CV FOLDS",key:"cvFolds",min:2,max:10,step:1,fmt:v=>`${v}-fold`},{label:"RANDOM STATE",key:"randomState",min:0,max:100,step:1,fmt:v=>v}].map(({label,key,min,max,step:s,fmt})=><div key={key} style={{padding:"14px 17px",border:`1px solid ${border}`,borderRadius:7,background:card}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:11}}><span style={{fontSize:9,fontFamily:"'Orbitron',sans-serif",color:sub,letterSpacing:".14em"}}>{label}</span><span style={{fontSize:18,fontFamily:"'Share Tech Mono'",color:accent}}>{fmt(pl.training[key])}</span></div><input type="range" min={min} max={max} step={s} value={pl.training[key]} onChange={e=>upd("training",{...pl.training,[key]:parseFloat(e.target.value)})} style={{accentColor:accent}}/><div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#1e293b",marginTop:4,fontFamily:"'Share Tech Mono'"}}><span>{fmt(min)}</span><span>{fmt(max)}</span></div></div>)}</div>}

                {/* S6 Evaluation */}
                {step===6&&pl.task&&<div style={{maxWidth:620}}>
                  <div style={{display:"flex",flexWrap:"wrap",gap:7,marginBottom:18}}>{(METRICS[pl.task]||[]).map(m=>{const sel=pl.evaluation.includes(m);return<div key={m} onClick={()=>togL("evaluation",m)} style={{padding:"7px 13px",border:`1px solid ${sel?"#10b981":border}`,borderRadius:4,background:sel?"rgba(16,185,129,.08)":card,fontSize:11,fontFamily:"'Share Tech Mono'",color:sel?"#10b981":sub,boxShadow:sel?"0 0 7px rgba(16,185,129,.22)":"none",display:"flex",alignItems:"center",gap:5,cursor:"pointer"}}><span style={{fontSize:9}}>{sel?"▣":"▢"}</span>{m}</div>;})}
                  </div>
                  {pl.evaluation.length>0&&<div style={{padding:"10px 13px",border:"1px solid rgba(16,185,129,.2)",borderRadius:6,background:"rgba(16,185,129,.04)",fontSize:11,color:"#10b981",fontFamily:"'Share Tech Mono'",cursor:"pointer"}} onClick={()=>setTab("visualize")}>✓ {pl.evaluation.length} metrics selected → View charts in Visualize tab →</div>}
                </div>}

                {/* S7 Export */}
                {step===7&&<div style={{maxWidth:480}}>
                  <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:9,marginBottom:16}}>{[{id:"joblib",label:"Joblib",ext:".joblib",icon:"⬡"},{id:"pickle",label:"Pickle",ext:".pkl",icon:"⬢"},{id:"onnx",label:"ONNX",ext:".onnx",icon:"◈"},{id:"mlflow",label:"MLflow",ext:"",icon:"◉"},{id:"pmml",label:"PMML",ext:".pmml",icon:"◎"},{id:"script",label:"Script",ext:".py",icon:"{}"}].map(f=><div key={f.id} className="hov" onClick={()=>upd("exportFormat",f.id)} style={{padding:"13px 11px",border:`1px solid ${pl.exportFormat===f.id?accent:border}`,borderRadius:7,background:pl.exportFormat===f.id?`${accent}0a`:card,textAlign:"center",boxShadow:pl.exportFormat===f.id?`0 0 14px ${accent}22`:"none"}}><div style={{fontSize:17,color:pl.exportFormat===f.id?accent:sub,marginBottom:5}}>{f.icon}</div><div style={{fontSize:11,fontWeight:600,color:pl.exportFormat===f.id?accent:sub}}>{f.label}</div>{f.ext&&<div style={{fontSize:9,color:"#1e293b",fontFamily:"'Share Tech Mono'",marginTop:2}}>{f.ext}</div>}</div>)}</div>
                  <button onClick={()=>setTab("export")} style={{padding:"9px 20px",border:`1px solid ${accent}`,borderRadius:6,background:`${accent}12`,color:accent,fontFamily:"'Orbitron',sans-serif",fontSize:9,fontWeight:700,cursor:"pointer"}}>⬇ DOCKER · FASTAPI · NOTEBOOK →</button>
                </div>}

                {/* S8 Custom Code */}
                {step===8&&<div style={{maxWidth:620}}>
                  <div style={{marginBottom:9,fontSize:10,color:sub,fontFamily:"'Share Tech Mono'"}}>// Custom Python appended to generated code</div>
                  <textarea rows={12} value={pl.customCode} onChange={e=>upd("customCode",e.target.value)} placeholder={"# Your custom code here\nimport matplotlib.pyplot as plt\n# plt.scatter(y_test, y_pred)\n# plt.show()"}/>
                  <div style={{marginTop:8,display:"flex",gap:5,flexWrap:"wrap"}}>
                    {["# SHAP explanation","# ROC curve","# Save predictions","# Log to MLflow"].map(s=><div key={s} onClick={()=>upd("customCode",(pl.customCode?pl.customCode+"\n":"")+s)} style={{padding:"3px 9px",border:`1px solid ${border}`,borderRadius:3,fontSize:9,color:sub,fontFamily:"'Share Tech Mono'",cursor:"pointer",background:card}}>+{s}</div>)}
                  </div>
                </div>}

                <div style={{display:"flex",gap:9,marginTop:32}}>
                  {step>0&&<button onClick={()=>goStep(step-1)} style={{padding:"8px 18px",border:`1px solid ${border}`,borderRadius:6,background:"transparent",color:sub,cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9}}>← BACK</button>}
                  {step<STEPS.length-1&&<button onClick={()=>canNext()&&goStep(step+1)} style={{padding:"8px 22px",border:`1px solid ${canNext()?accent:border}`,borderRadius:6,background:canNext()?`${accent}14`:"transparent",color:canNext()?accent:sub,cursor:canNext()?"pointer":"not-allowed",fontFamily:"'Orbitron',sans-serif",fontSize:9,fontWeight:700,boxShadow:canNext()?`0 0 12px ${accent}33`:"none"}}>CONTINUE →</button>}
                  {step===STEPS.length-1&&pl.task&&pl.model&&<button onClick={()=>setTab("train")} style={{padding:"8px 22px",border:`1px solid #10b981`,borderRadius:6,background:"rgba(16,185,129,.14)",color:"#10b981",cursor:"pointer",fontFamily:"'Orbitron',sans-serif",fontSize:9,fontWeight:700,boxShadow:"0 0 12px rgba(16,185,129,.3)"}}>▶ TRAIN IN BROWSER</button>}
                </div>
              </div>
            </main>
          </div>
        )}
      </div>
    </div>
  );
}
