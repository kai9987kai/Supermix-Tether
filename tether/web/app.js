"use strict";
const byId = (id) => document.getElementById(id);
const arms = ["passive", "dropout", "iit", "effect_only", "invariance_only", "unpaired", "tether"];
const armNames = {passive:"Passive",dropout:"Donor dropout",iit:"IIT",effect_only:"Effect only",invariance_only:"Invariance only",unpaired:"Unpaired control",tether:"Tether"};
const modes = {
  factual:"Unmodified donor outputs and context from a frozen test task.",
  relevant:"Replace the queried source. The correct answer changes; the old context stays.",
  irrelevant:"Replace a source outside the query. The correct answer stays the same.",
  damaged:"Corrupt one expert posterior on the queried source. The task and its correct answer stay the same.",
  damage_two:"Zero-shot 2-of-3 specialist corruption on the queried source. Majority consensus fails; dark-knowledge forensics isolates the lone intact specialist.",
  damage_common:"Total 3-of-3 specialist corruption. Dark-knowledge tail entropy approaches log 9; Epistemic Trust collapses to 0.0, safely triggering calibrated abstention.",
  erased:"Remove donor information to measure the receiver's dependence on specialists.",
  context_shift:"Change the answer-proxy context while retaining the source images and task."
};
let study = null, mode = "factual", exampleRequest = 0;
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function finite(value) { return typeof value === "number" && Number.isFinite(value); }
function mean(values) { const kept = values.filter(finite); return kept.length ? kept.reduce((a,b)=>a+b,0)/kept.length : null; }
function percentage(value) { return finite(value) ? (100 * value).toFixed(1) + "%" : "—"; }
function points(value) { return finite(value) ? (value >= 0 ? "+" : "") + (100*value).toFixed(1) + " pp" : "—"; }
function human(value) { return String(value ?? "").replaceAll("_", " ").replaceAll("-", " "); }
function errorNotice(id, message) { const node=byId(id); if(node){node.textContent=message || ""; node.classList.toggle("hidden", !message);} }
async function api(path) {
  const response = await fetch(path, {cache:"no-store"});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Saved study could not be loaded.");
  return data;
}
function renderConformalChips(container, setArr, abstain, topPred) {
  if (!container) return;
  container.replaceChildren();
  if (abstain || !Array.isArray(setArr) || setArr.length === 0) {
    container.append(element("span", "∅ Empty Set (Epistemic Abstention)", "conformal-chip empty"));
    return;
  }
  setArr.forEach((d, idx) => {
    const chip = element("span", "Digit " + d, "conformal-chip" + (d === topPred || idx === 0 ? " primary" : ""));
    container.append(chip);
  });
}
function metricsFor(arm, condition) {
  return (study?.rows || []).filter(row => row.arm === arm).map(row => row.metrics?.[condition]).filter(Boolean);
}
function renderCondition() {
  const condition=byId("condition-select").value;
  const metric=byId("metric-select")?.value || "accuracy";
  const isError=metric.endsWith("_error");
  byId("condition-name").textContent=human(condition) || "—";
  const records=metricsFor("tether",condition);
  byId("hero-accuracy").textContent=percentage(mean(records.map(x=>x.accuracy)));
  byId("hero-gap").textContent=points(mean(records.map(x=>x.donor_erasure_gap)));
  byId("condition-caption").textContent=records.length ? records.length+" recorded seed"+(records.length===1?"":"s")+" · frozen evaluation" : "No Tether metrics for this condition";
  if(byId("arm-chart-title")) byId("arm-chart-title").textContent=human(metric).replace(/^./,c=>c.toUpperCase())+" by receiver";
  if(byId("arm-chart-subtitle")) byId("arm-chart-subtitle").textContent=isError?"Seed means · L2 error (lower is better)":"Seed means · 0–100%";
  const bars=byId("arm-bars"); bars.replaceChildren();
  arms.forEach(arm=>{
    const values=metricsFor(arm,condition);
    if (!values.length) return;
    const val=mean(values.map(x=>x[metric]));
    const row=element("div",undefined,"arm-row "+arm);
    row.append(element("span",armNames[arm],"arm-label"));
    const track=element("div",undefined,"bar-track"), fill=element("div",undefined,"bar-fill");
    const pct=finite(val)?(isError?Math.min(100,val*50):Math.max(0,Math.min(100,val*100))):0;
    fill.style.width=pct+"%";
    const formatted=finite(val)?(isError?val.toFixed(3):percentage(val)):"—";
    track.append(fill); row.append(track,element("span",formatted,"arm-value")); bars.append(row);
  });
  if (!bars.children.length) bars.append(element("p","No metrics recorded for this condition.","empty-state"));
  const paired=byId("paired-results"); paired.replaceChildren();
  const wanted=new Set([condition+"."+metric, condition+".accuracy", condition+".correct_pair_rate"]);
  const comparisons=(study?.comparisons || []).filter(x=>wanted.has(x.metric));
  const sorted=[...comparisons].sort((a,b)=>(a.metric===condition+"."+metric?-1:b.metric===condition+"."+metric?1:a.metric.localeCompare(b.metric))||a.baseline.localeCompare(b.baseline));
  sorted.slice(0,4).forEach(item=>{
    const itemError=item.metric.endsWith("_error");
    const fmt=v=>finite(v)?(itemError?(v>=0?"+":"")+v.toFixed(3):points(v)):"—";
    const row=element("div",undefined,"paired-item"), title=element("div",undefined,"paired-title");
    title.append(element("span",(armNames[item.arm]||human(item.arm))+" − "+(armNames[item.baseline]||human(item.baseline))),element("strong",fmt(item.mean_difference)));
    row.append(title);
    const interval=Array.isArray(item.interval) && item.interval.length===2 ? "["+fmt(item.interval[0])+", "+fmt(item.interval[1])+"]" : "interval not recorded";
    row.append(element("small",human(item.metric)+" · "+interval)); paired.append(row);
  });
  if (!paired.children.length) paired.append(element("p","No paired comparisons recorded for this condition.","empty-state"));
}
function renderProvenance() {
  const dl=byId("provenance"); dl.replaceChildren();
  const entries = Object.entries(study.provenance || {});
  const config=study.config || {};
  const rows=[["Study schema",study.schema],["Recorded seeds",[...new Set((study.rows||[]).map(x=>x.seed))].join(", ")],["Phase",config.phase || config.stage || "Saved study"],...entries];
  rows.slice(0,10).forEach(([key,value])=>{
    dl.append(element("dt",human(key)),element("dd",typeof value==="object"?JSON.stringify(value).slice(0,600):String(value??"—")));
  });
  byId("decision-detail").textContent=JSON.stringify(study.decision || {},null,2);
}
async function loadStudy() {
  try {
    study=await api("/api/results");
    if (!Array.isArray(study.rows)) throw new Error("Saved summary has no study rows.");
    const count=new Set(study.rows.map(x=>x.seed)).size;
    const complete=study.status==="complete";
    byId("study-status").textContent=(complete?"Completed study":"Incomplete study")+" · "+count+" seed"+(count===1?"":"s");
    byId("study-status").classList.toggle("unavailable",!complete);
    const config=study.config || {};
    const proto=config.protocol || "v1";
    if(byId("protocol-version")) byId("protocol-version").textContent="v0.7 ("+proto+")";
    byId("study-config").textContent=study.rows.length+" arm records · "+human(config.phase||config.stage||"saved evaluation")+" ("+proto+")";
    const seeds=[...new Set(study.rows.map(x=>x.seed))].sort((a,b)=>a-b);
    const seedSelect=byId("seed-select");
    if(seedSelect){
      seedSelect.replaceChildren();
      seeds.forEach(s=>{const opt=element("option","Seed "+s);opt.value=String(s);seedSelect.append(opt);});
    }
    const decision=study.decision||{};
    byId("hero-decision").textContent=decision.candidate_supported_in_this_study===true ? "Supported" : (decision.promoted===true ? "Promoted" : "Unpromoted");
    byId("decision-caption").textContent=decision.reason || decision.status || "See recorded decision below";
    const conditions=[...new Set(study.rows.flatMap(row=>Object.keys(row.metrics||{})))];
    const select=byId("condition-select"); select.replaceChildren();
    conditions.forEach(key=>{const option=element("option",human(key));option.value=key;select.append(option);});
    select.disabled=!conditions.length;
    const preferred=conditions.find(x=>/wrong|shift|independent/.test(x));
    if(preferred) select.value=preferred;
    renderCondition(); renderProvenance(); errorNotice("study-error",complete?"":"This study has not completed. Metrics may be partial; checkpoint inspection is unavailable.");
    if(complete) await loadExample();
  } catch(error) {
    byId("study-status").textContent="Saved study unavailable"; byId("study-status").classList.add("unavailable");
    byId("study-config").textContent="The viewer is ready; no completed results were loaded.";
    errorNotice("study-error",error.message+" Start this viewer with a completed study directory.");
    byId("example-status").textContent="No completed saved study";
  }
}
function drawDigit(canvas,pixels) {
  const ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,128,128);ctx.fillStyle="#202633";ctx.fillRect(0,0,128,128);
  if (!Array.isArray(pixels)||pixels.length!==64) return;
  for(let i=0;i<64;i++){
    const value=Math.max(0,Math.min(1,Number(pixels[i])||0));
    ctx.fillStyle="rgb("+Math.round(32+value*209)+","+Math.round(38+value*198)+","+Math.round(51+value*185)+")";
    ctx.fillRect((i%8)*16,Math.floor(i/8)*16,16,16);
  }
}
function renderExample(data) {
  const source=Number(data.query?.source), operation=Number(data.query?.operation);
  const operationNames=["identity","next digit (modulo 10)","mirror digit (9 − digit)"];
  byId("task-query").textContent="Source "+(source+1)+" → "+(operationNames[operation]||human(data.query?.operation));
  byId("mode-tag").textContent=human(mode).toUpperCase();
  byId("prediction").textContent=String(data.prediction??"—");
  byId("target").textContent=String(data.target??"—");
  const valid=data.prediction !== undefined && data.target !== undefined;
  const correct=valid && data.prediction===data.target;
  byId("correctness").textContent=valid?(correct?"CORRECT":"INCORRECT"):"NO PREDICTION";
  byId("correctness").className="tag "+(valid?(correct?"correct":"wrong"):"");
  const probabilities=Array.isArray(data.probabilities)?data.probabilities:[];
  byId("confidence").textContent=finite(probabilities[data.prediction])?percentage(probabilities[data.prediction]):"—";
  const chart=byId("probabilities");chart.replaceChildren();
  for(let i=0;i<10;i++){
    const value=probabilities[i], col=element("div",undefined,"prob-column"+(i===data.prediction?" winner":"")), fill=element("i",undefined,"prob-fill");
    fill.style.height=(finite(value)?Math.max(2,Math.min(75,value*75)):2)+"px";
    col.title="Digit "+i+": "+percentage(value);col.append(fill,element("span",String(i)));chart.append(col);
  }
  document.querySelectorAll(".source-tile").forEach((tile,i)=>{
    drawDigit(tile.querySelector("canvas"),data.images?.[i]);
    tile.classList.toggle("selected",i===source);
    tile.querySelector(".source-title span").textContent=i===source?"QUERIED":"";
    const weight=data.source_weights?.[i];
    tile.querySelector(".weight-track i").style.width=(finite(weight)?Math.max(0,Math.min(100,weight*100)):0)+"%";
    tile.querySelector("small b").textContent=percentage(weight);
  });
  const experts=byId("expert-weights");experts.replaceChildren();
  for(let i=0;i<3;i++){
    const row=element("div",undefined,"expert-row"), stack=element("div",undefined,"expert-stack");
    row.append(element("span","Source "+(i+1)));
    for(let j=0;j<3;j++){const value=data.expert_weights?.[i]?.[j], part=element("i");part.style.width=(finite(value)?Math.max(0,Math.min(100,value*100)):0)+"%";part.title="Expert "+["A","B","C"][j]+": "+percentage(value);stack.append(part);}
    row.append(stack);experts.append(row);
  }
  const trustBox=byId("trust-scores");
  if(trustBox){
    trustBox.replaceChildren();
    const damagedSet=new Set(Array.isArray(data.damaged_experts)?data.damaged_experts:[]);
    if(byId("damaged-badge")) byId("damaged-badge").textContent=damagedSet.size?("Corrupted: "+[...damagedSet].map(x=>["A","B","C"][x]).join(", ")):"1 − tail entropy";
    for(let j=0;j<3;j++){
      const tVal=data.trust_score?.[source]?.[j], wVal=data.expert_weights?.[source]?.[j];
      const isDmg=damagedSet.has(j);
      const pill=element("div",undefined,"trust-pill"+(isDmg?" damaged":""));
      pill.append(element("strong","Expert "+["A","B","C"][j]+(isDmg?" (damaged)":"")),element("span","Trust "+percentage(tVal)+" · Wt "+percentage(wVal)));
      trustBox.append(pill);
    }
  }
  const isAbstain=Boolean(data.abstain);
  byId("inspector-abstain-banner")?.classList.toggle("hidden",!isAbstain);
  if(isAbstain){
    byId("correctness").textContent="SAFE ABSTAIN";
    byId("correctness").className="tag wrong";
  }
  renderConformalChips(byId("inspector-conformal-set"), data.conformal_set, isAbstain, data.prediction);
  if(byId("inspector-conformal-meta") && finite(data.temperature) && finite(data.conformal_q_hat)){
    byId("inspector-conformal-meta").textContent="T* = "+data.temperature.toFixed(2)+" · q̂ = "+data.conformal_q_hat.toFixed(3);
  }
  const trustVal=Number(data.epistemic_trust??0);
  if(byId("inspector-trust-val")) byId("inspector-trust-val").textContent=percentage(trustVal);
  if(byId("inspector-meter-fill")) byId("inspector-meter-fill").style.width=Math.max(0,Math.min(100,trustVal*100))+"%";
  if(byId("inspector-meter-val")) byId("inspector-meter-val").textContent="τ = "+trustVal.toFixed(3);
  if(byId("inspector-ig-err") && finite(data.ig_completeness_error)){
    byId("inspector-ig-err").textContent="Completeness Δ = "+data.ig_completeness_error.toFixed(4);
  }
  const insSal=byId("inspector-saliency");
  if(insSal){
    insSal.replaceChildren();
    const expSal=(data.ig_expert_saliency||data.expert_saliency)?.[source]||[0,0,0];
    const maxSal=Math.max(0.0001,...expSal);
    ["Expert A","Expert B","Expert C"].forEach((name,j)=>{
      const sVal=expSal[j]||0;
      const row=element("div",undefined,"saliency-row");
      row.append(element("span",name));
      const track=element("div",undefined,"saliency-track"), fill=element("div",undefined,"saliency-fill");
      fill.style.width=Math.min(100,(sVal/maxSal)*100)+"%";
      track.append(fill);row.append(track,element("span",sVal.toFixed(3),"saliency-val"));
      insSal.append(row);
    });
  }
  const insShapley=byId("inspector-shapley");
  if(insShapley && Array.isArray(data.shapley_values)){
    insShapley.replaceChildren();
    const phis=data.shapley_values;
    const maxAbs=Math.max(0.0001,...phis.map(x=>Math.abs(Number(x)||0)));
    if(byId("inspector-shapley-err") && finite(data.shapley_efficiency_error)){
      byId("inspector-shapley-err").textContent="Σφ = "+(phis.reduce((a,b)=>a+b,0)).toFixed(2);
    }
    ["Expert A (φ₀)","Expert B (φ₁)","Expert C (φ₂)"].forEach((name,j)=>{
      const val=Number(phis[j]||0);
      const row=element("div",undefined,"saliency-row");
      row.style.gridTemplateColumns="72px minmax(20px,1fr) 52px";
      row.append(element("span",name));
      const track=element("div",undefined,"saliency-track"), fill=element("div",undefined,"saliency-fill");
      fill.style.width=Math.min(100,(Math.abs(val)/maxAbs)*100)+"%";
      fill.style.background=val>=0?"#578776":"#c0392b";
      track.append(fill);row.append(track,element("span",(val>=0?"+":"")+val.toFixed(2),"saliency-val"));
      insShapley.append(row);
    });
  }
  renderPidBars(byId("inspector-pid"), byId("inspector-pid-meta"), data.pid);
  const patchBox=byId("inspector-patching");
  if(patchBox && data.interchange_patching){
    patchBox.replaceChildren();
    const nodeLabels={
      expert_weights:"Reliability Gate",
      source_posteriors:"Source Posteriors",
      source_weights:"Source Router",
      pooled_posterior:"Pooled State",
      task_logits:"Task Logits",
      shortcut_logits:"Shortcut Logits"
    };
    Object.entries(nodeLabels).forEach(([key,label])=>{
      const info=data.interchange_patching[key];
      if(!info)return;
      const rest=Math.max(0,Math.min(1,Number(info.restoration)||0));
      const row=element("div",undefined,"saliency-row");
      row.style.gridTemplateColumns="105px minmax(20px,1fr) 70px";
      row.append(element("span",label));
      const track=element("div",undefined,"saliency-track"), fill=element("div",undefined,"saliency-fill");
      fill.style.width=(rest*100)+"%";
      fill.style.background=rest>0.5?"#578776":"#b5c9c0";
      track.append(fill);
      row.append(track,element("span","ŷ="+info.prediction+" ("+Math.round(rest*100)+"%)","saliency-val"));
      patchBox.append(row);
    });
  }
  byId("mode-description").textContent=modes[mode];
  byId("example-status").textContent="Saved #"+data.index+" · Seed "+(data.seed??"—")+" · "+armNames[data.arm || byId("arm-select").value];
}
async function loadExample() {
  if(study?.status!=="complete") return;
  const token=++exampleRequest;
  const index=Number(byId("example-index").value);
  if(!Number.isInteger(index)||index<0||index>1000000){errorNotice("example-error","Choose a non-negative whole example index.");return;}
  byId("example-status").textContent="Running saved inference…";
  try {
    const params={index:String(index),arm:byId("arm-select").value,mode};
    const seedVal=byId("seed-select")?.value;
    if(seedVal) params.seed=seedVal;
    const query=new URLSearchParams(params);
    const data=await api("/api/example?"+query);
    if(token!==exampleRequest)return;
    renderExample(data);errorNotice("example-error","");
  } catch(error) {
    if(token!==exampleRequest)return;
    byId("example-status").textContent="Inspection unavailable";
    byId("prediction").textContent="—";byId("target").textContent="—";
    byId("correctness").textContent="UNAVAILABLE";byId("correctness").className="tag";
    errorNotice("example-error",error.message);
  }
}
byId("condition-select").addEventListener("change",renderCondition);
if(byId("metric-select")) byId("metric-select").addEventListener("change",renderCondition);
byId("arm-select").addEventListener("change",loadExample);
if(byId("seed-select")) byId("seed-select").addEventListener("change",()=>{loadExample();scheduleLivePredict();});
byId("example-index").addEventListener("change",loadExample);
byId("previous-example").addEventListener("click",()=>{byId("example-index").value=Math.max(0,Number(byId("example-index").value)-1);loadExample();});
byId("next-example").addEventListener("click",()=>{byId("example-index").value=Math.min(1000000,Number(byId("example-index").value)+1);loadExample();});
document.querySelectorAll("[data-mode]").forEach(button=>button.addEventListener("click",()=>{
  mode=button.dataset.mode;
  document.querySelectorAll("[data-mode]").forEach(item=>{item.classList.toggle("selected",item===button);item.setAttribute("aria-pressed",String(item===button));});
  byId("mode-description").textContent=modes[mode];loadExample();
}));
document.querySelectorAll(".nav-item").forEach(link=>link.addEventListener("click",()=>{document.querySelectorAll(".nav-item").forEach(item=>item.classList.toggle("active",item===link));}));
document.querySelectorAll(".source-tile canvas").forEach(canvas=>drawDigit(canvas,null));

let pgPixels=new Float32Array(64), pgDigit=3, pgTamper="clean", pgIsDrawing=false, pgDebounceTimer=null;
function pgGetAlpha(){
  const cov=Number(byId("pg-coverage")?.value||0.95);
  return Math.max(0.01,Math.min(0.20,1.0-cov));
}
function pgInitCanvas(){
  const canvas=byId("pg-canvas");if(!canvas)return;
  const ctx=canvas.getContext("2d");
  function redraw(){
    ctx.clearRect(0,0,192,192);ctx.fillStyle="#202633";ctx.fillRect(0,0,192,192);
    for(let i=0;i<64;i++){
      const col=i%8, row=Math.floor(i/8), val=pgPixels[i];
      if(val>0){
        ctx.fillStyle="rgb("+Math.round(32+val*209)+","+Math.round(38+val*198)+","+Math.round(51+val*185)+")";
        ctx.fillRect(col*24,row*24,24,24);
      }
      ctx.strokeStyle="#283142";ctx.lineWidth=1;ctx.strokeRect(col*24,row*24,24,24);
    }
  }
  function paint(evt){
    const rect=canvas.getBoundingClientRect();
    const clientX=evt.touches?evt.touches[0].clientX:evt.clientX;
    const clientY=evt.touches?evt.touches[0].clientY:evt.clientY;
    const x=clientX-rect.left, y=clientY-rect.top;
    const col=Math.floor(x/(rect.width/8)), row=Math.floor(y/(rect.height/8));
    if(col>=0&&col<8&&row>=0&&row<8){
      pgPixels[row*8+col]=1.0;redraw();scheduleLivePredict();
    }
  }
  canvas.addEventListener("mousedown",(e)=>{pgIsDrawing=true;paint(e);});
  window.addEventListener("mouseup",()=>{pgIsDrawing=false;});
  canvas.addEventListener("mousemove",(e)=>{if(pgIsDrawing)paint(e);});
  canvas.addEventListener("touchstart",(e)=>{pgIsDrawing=true;paint(e);e.preventDefault();},{passive:false});
  window.addEventListener("touchend",()=>{pgIsDrawing=false;});
  canvas.addEventListener("touchmove",(e)=>{if(pgIsDrawing){paint(e);e.preventDefault();}},{passive:false});
  byId("pg-clear")?.addEventListener("click",()=>{pgPixels.fill(0);redraw();scheduleLivePredict();});
  byId("pg-invert")?.addEventListener("click",()=>{for(let i=0;i<64;i++)pgPixels[i]=Math.max(0,1.0-pgPixels[i]);redraw();scheduleLivePredict();});
  byId("pg-noise")?.addEventListener("click",()=>{
    for(let i=0;i<64;i++){
      const delta=(Math.sin(i*12.9898+pgPixels[i]*78.233)*43758.5453)%1;
      pgPixels[i]=Math.max(0,Math.min(1,pgPixels[i]*0.82+Math.abs(delta)*0.28));
    }
    redraw();scheduleLivePredict();
  });
  document.querySelectorAll(".digit-btn").forEach(btn=>{
    btn.addEventListener("click",()=>{
      document.querySelectorAll(".digit-btn").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");
      pgDigit=Number(btn.dataset.digit);
      loadPresetDigit(pgDigit);
    });
  });
  document.querySelectorAll(".tamper-btn").forEach(btn=>{
    btn.addEventListener("click",()=>{
      document.querySelectorAll(".tamper-btn").forEach(b=>b.classList.remove("selected"));
      btn.classList.add("selected");
      pgTamper=btn.dataset.tamper;
      byId("pg-tamper-tag").textContent=pgTamper==="clean"?"CLEAN":pgTamper.toUpperCase().replace("_"," ");
      scheduleLivePredict();
    });
  });
  byId("pg-confidence")?.addEventListener("input",(e)=>{
    byId("pg-conf-val").textContent=(Number(e.target.value)*100).toFixed(1)+"%";
    scheduleLivePredict();
  });
  byId("pg-coverage")?.addEventListener("input",(e)=>{
    const cov=Number(e.target.value);
    byId("pg-cov-val").textContent=Math.round(cov*100)+"%";
    if(byId("pg-conformal-label")) byId("pg-conformal-label").innerHTML="SPLIT CONFORMAL SET C<sub>"+cov.toFixed(2)+"</sub>(x)";
    scheduleLivePredict();
  });
  ["pg-arm-select","pg-source-select","pg-op-select","pg-context-mode","pg-corrupt-class","pg-defense"].forEach(id=>{
    byId(id)?.addEventListener("change",scheduleLivePredict);
  });
  initReceiptDialog();
  window.pgRedrawCanvas=redraw;
  loadPresetDigit(pgDigit);
}
let lastLiveData=null;
function renderPidBars(container, metaNode, pid){
  if(!container||!pid)return;
  container.replaceChildren();
  const shares=pid.normalized_shares||{};
  if(metaNode && finite(pid.total_information)){
    metaNode.textContent="I = "+pid.total_information.toFixed(2)+" nats (Δ="+Number(pid.conservation_error||0).toFixed(4)+")";
  }
  const items=[
    ["Redundancy R", Number(shares.redundancy||0), "#578776"],
    ["Unique U₀ (A)", Number(shares.unique?.[0]||0), "#7358cf"],
    ["Unique U₁ (B)", Number(shares.unique?.[1]||0), "#8a72cf"],
    ["Unique U₂ (C)", Number(shares.unique?.[2]||0), "#9b83d9"],
    ["Synergy S", Number(shares.synergy||0), "#ce874a"]
  ];
  items.forEach(([label,val,color])=>{
    const row=element("div",undefined,"saliency-row");
    row.style.gridTemplateColumns="84px minmax(20px,1fr) 48px";
    row.append(element("span",label));
    const track=element("div",undefined,"saliency-track"), fill=element("div",undefined,"saliency-fill");
    fill.style.width=Math.max(0,Math.min(100,val*100))+"%";
    fill.style.background=color;
    track.append(fill);
    row.append(track,element("span",percentage(val),"saliency-val"));
    container.append(row);
  });
}
let supportsDisplayTransition;
function canTransitionDisplay(){
  if(supportsDisplayTransition!==undefined)return supportsDisplayTransition;
  const probe=document.createElement("div");
  probe.style.cssText="transition: display 1s allow-discrete !important; display: block;";
  document.body.appendChild(probe);
  getComputedStyle(probe).display;
  probe.style.display="none";
  supportsDisplayTransition=getComputedStyle(probe).display==="block";
  probe.remove();
  return supportsDisplayTransition;
}
function supportsTopLayerExit(){
  return window.CSS?.supports?.("overlay","auto") && canTransitionDisplay();
}
async function closeTopLayer(el){
  if(!el)return;
  if(!supportsTopLayerExit()){
    el.setAttribute("data-closing","");
    const animations=el.getAnimations({subtree:true});
    if(animations.length>0){
      await Promise.race([
        Promise.allSettled(animations.map(a=>a.finished)),
        new Promise(r=>setTimeout(r,2000)),
      ]);
    }
    if(!el.hasAttribute("data-closing"))return;
    el.removeAttribute("data-closing");
  }
  if(typeof el.close==="function")el.close();
  else if(typeof el.hidePopover==="function")el.hidePopover();
}
function initReceiptDialog(){
  const dialog=byId("circuit-receipt-dialog");
  const openBtn=byId("pg-open-receipt");
  const closeBtn=byId("close-receipt-dialog");
  if(!dialog)return;
  if(openBtn){
    openBtn.addEventListener("click",()=>{
      populateReceiptDialog(lastLiveData);
      dialog.showModal();
    });
  }
  if(closeBtn){
    closeBtn.addEventListener("click",()=>closeTopLayer(dialog));
  }
  dialog.addEventListener("cancel",(event)=>{
    event.preventDefault();
    closeTopLayer(dialog);
  });
  if(!("closedBy" in HTMLDialogElement.prototype)){
    dialog.addEventListener("click",(event)=>{
      const rect=dialog.getBoundingClientRect();
      const clickedInDialog=(
        rect.top<=event.clientY &&
        event.clientY<=rect.top+rect.height &&
        rect.left<=event.clientX &&
        event.clientX<=rect.left+rect.width
      );
      if(!clickedInDialog){
        closeTopLayer(dialog);
      }
    });
  }
}
function populateReceiptDialog(data){
  const body=byId("circuit-receipt-body");
  if(!body||!data)return;
  body.replaceChildren();
  const pid=data.pid||{};
  const med=data.causal_mediation||{};
  const ig=data.information_geometry||{};
  const tr=data.tangent_robustness||{};
  const grid=element("div",undefined,"receipt-grid");
  const b1=element("div",undefined,"receipt-box");
  b1.append(
    element("strong","1. Williams-Beer PID (arXiv:1004.2515)"),
    element("div","Total Directed Info I(012; y*) = "+Number(pid.total_information||0).toFixed(3)+" nats"),
    element("div","Shared Redundancy R = "+Number(pid.redundancy||0).toFixed(3)+" ("+percentage(pid.normalized_shares?.redundancy)+")"),
    element("div","Unique [U₀, U₁, U₂] = ["+(pid.unique||[0,0,0]).map(x=>Number(x).toFixed(2)).join(", ")+"]"),
    element("div","Coalitional Synergy S = "+Number(pid.synergy||0).toFixed(3)+" · Conservation Δ = "+Number(pid.conservation_error||0).toFixed(6))
  );
  const b2=element("div",undefined,"receipt-box");
  b2.append(
    element("strong","2. Causal Mediation & Crit Breakdown (Vig et al. 2020)"),
    element("div","Visual NDE = "+Number(med.nde_visual||0).toFixed(3)+" · Shortcut NIE = "+Number(med.nie_shortcut||0).toFixed(3)),
    element("div","Visual Dominance Ratio ρ_vis = "+percentage(med.visual_dominance_ratio)),
    element("div","Critical Breakdown α*_crit = "+(finite(data.critical_corruption_threshold)?percentage(data.critical_corruption_threshold):"—")),
    element("div","Context Mode = "+String(data.context_mode||"neutral").toUpperCase())
  );
  const b3=element("div",undefined,"receipt-box");
  b3.append(
    element("strong","3. Weighted Conformal & CRC (Tibshirani 2019; Angelopoulos 2024)"),
    element("div","Standard RAPS C(x) = {"+(data.conformal_set||[]).join(", ")+"} · WCP C_w(x) = {"+(data.wcp_conformal_set||data.conformal_set||[]).join(", ")+"}"),
    element("div","Conformal Risk Control C_CRC(x) = {"+(data.crc_conformal_set||data.conformal_set||[]).join(", ")+"} (λ_CRC = "+Number(data.crc_lambda||0).toFixed(3)+")")
  );
  const b4=element("div",undefined,"receipt-box");
  const eEnergies=Array.isArray(data.tail_free_energy)?data.tail_free_energy:[0,0,0];
  const nEff=Array.isArray(data.spectral_participation_ratio)?data.spectral_participation_ratio:[1,1,1];
  b4.append(
    element("strong","4. Helmholtz Tail Free Energy (Liu et al. 2020)"),
    element("div","Queried Source E_tail = ["+eEnergies.map(x=>Number(x).toFixed(2)).join(", ")+"]"),
    element("div","Effective Runner-up Support N_eff = ["+nEff.map(x=>Number(x).toFixed(2)).join(", ")+"] / 9.0"),
    element("div","Mahalanobis D_M = ["+(data.mahalanobis_distance||[0,0,0]).map(x=>Number(x).toFixed(1)).join(", ")+"]")
  );
  const b5=element("div",undefined,"receipt-box");
  const frMat=ig.fisher_rao_matrix||[[0,0,0],[0,0,0],[0,0,0]];
  b5.append(
    element("strong","5. Fisher–Rao Simplex Geometry (Rao 1945; Amari 2016)"),
    element("div","Geodesic d_FR(A,B)="+Number(frMat[0]?.[1]||0).toFixed(2)+" · d_FR(A,C)="+Number(frMat[0]?.[2]||0).toFixed(2)+" · d_FR(B,C)="+Number(frMat[1]?.[2]||0).toFixed(2)+" rad"),
    element("div","Fréchet Var Unweighted = "+Number(ig.unweighted_frechet_variance||0).toFixed(3)+" → Routed = "+Number(ig.routed_frechet_variance||0).toFixed(4)),
    element("div","Geodesic Conflict Suppression = "+percentage(ig.geodesic_suppression_ratio))
  );
  const b6=element("div",undefined,"receipt-box");
  b6.append(
    element("strong","6. Tangent-Space Certified TV Robustness (v0.7)"),
    element("div","Logit Margin γ(P) = "+Number(tr.logit_margin||0).toFixed(2)+" logits"),
    element("div","Tangent Span ||P_tan ∇γ|| = ["+(tr.tangent_span_sensitivity||[0,0,0]).map(x=>Number(x).toFixed(2)).join(", ")+"]"),
    element("div","Certified TV Radius r^TV = ["+(tr.certified_tv_radius||[1,1,1]).map(x=>Number(x).toFixed(2)).join(", ")+"]")
  );
  const vb=data.vertex_breakdown||{};
  const b7=element("div",undefined,"receipt-box");
  b7.append(
    element("strong","7. Simplex Vertex Breakdown TV Profile (v0.8)"),
    element("div","Min Singleton Breakdown TV = "+(vb.min_singleton_breakdown_tv!=null?Number(vb.min_singleton_breakdown_tv).toFixed(3):"Immune (No vertex flip)")),
    element("div","Min Pair Coalition Breakdown TV = "+(vb.min_pair_breakdown_tv!=null?Number(vb.min_pair_breakdown_tv).toFixed(3):"—")),
    element("div","Grand Coalition Breakdown TV = "+(vb.grand_coalition_breakdown_tv!=null?Number(vb.grand_coalition_breakdown_tv).toFixed(3):"—")+" (Adv Class: "+(vb.grand_coalition?.adversarial_prediction??'—')+")")
  );
  grid.append(b1,b2,b3,b4,b5,b6,b7);
  const coal=element("div",undefined,"receipt-mono");
  coal.textContent="2³ Coalition Payoff Lattice (logits): "+JSON.stringify(data.coalition_payoffs||{});
  body.append(grid,coal);
}
function renderGeodesicWidget(data){
  const svg=byId("pg-geodesic-svg");
  const meta=byId("pg-geodesic-meta");
  const popBody=byId("geodesic-popover-body");
  if(!svg||!data)return;
  const ig=data.information_geometry||{};
  const tr=data.tangent_robustness||{};
  const vb=data.vertex_breakdown||{};
  const src=Number(data.source||0);
  const wts=data.expert_weights?.[src]||[0.33,0.33,0.34];
  const dmg=new Set(Array.isArray(data.damaged_experts)?data.damaged_experts:[]);
  const fr=ig.fisher_rao_matrix||[[0,0,0],[0,0,0],[0,0,0]];
  const tvRad=tr.certified_tv_radius||[1,1,1];
  const vA=[105,14], vB=[28,78], vC=[182,78];
  const bx=wts[0]*vA[0]+wts[1]*vB[0]+wts[2]*vC[0];
  const by=wts[0]*vA[1]+wts[1]*vB[1]+wts[2]*vC[1];
  svg.innerHTML=`
    <polygon points="${vA[0]},${vA[1]} ${vB[0]},${vB[1]} ${vC[0]},${vC[1]}" fill="#f4f2fb" stroke="#d5ccf0" stroke-width="1.2"/>
    <text x="${(vA[0]+vB[0])/2-14}" y="${(vA[1]+vB[1])/2}" font-size="7" fill="#7c828b">${Number(fr[0]?.[1]||0).toFixed(2)}</text>
    <text x="${(vA[0]+vC[0])/2+4}" y="${(vA[1]+vC[1])/2}" font-size="7" fill="#7c828b">${Number(fr[0]?.[2]||0).toFixed(2)}</text>
    <text x="105" y="88" text-anchor="middle" font-size="7" fill="#7c828b">d_FR(B,C)=${Number(fr[1]?.[2]||0).toFixed(2)} rad</text>
    <circle cx="${vA[0]}" cy="${vA[1]}" r="5" fill="${dmg.has(0)?'#c0392b':'#578776'}"/>
    <circle cx="${vB[0]}" cy="${vB[1]}" r="5" fill="${dmg.has(1)?'#c0392b':'#578776'}"/>
    <circle cx="${vC[0]}" cy="${vC[1]}" r="5" fill="${dmg.has(2)?'#c0392b':'#578776'}"/>
    <circle cx="${bx.toFixed(1)}" cy="${by.toFixed(1)}" r="5.5" fill="#7358cf" stroke="#fff" stroke-width="1.5"/>
    <text x="${vA[0]}" y="${vA[1]-6}" text-anchor="middle" font-size="7.5" font-weight="600" fill="#222b36">A</text>
    <text x="${vB[0]-9}" y="${vB[1]+3}" text-anchor="middle" font-size="7.5" font-weight="600" fill="#222b36">B</text>
    <text x="${vC[0]+9}" y="${vC[1]+3}" text-anchor="middle" font-size="7.5" font-weight="600" fill="#222b36">C</text>
  `;
  if(meta){
    meta.replaceChildren(
      element("div","Fréchet σ²: "+Number(ig.unweighted_frechet_variance||0).toFixed(2)+" → "+Number(ig.routed_frechet_variance||0).toFixed(3)+" rad²"),
      element("div","Geodesic Suppression: "+percentage(ig.geodesic_suppression_ratio)),
      element("div","Cert r^TV: ["+tvRad.map(x=>Number(x).toFixed(2)).join(", ")+"] · Break TV: "+(vb.grand_coalition_breakdown_tv!=null?Number(vb.grand_coalition_breakdown_tv).toFixed(2):"—")),
      element("div","CRC Set: {"+(data.crc_conformal_set||data.conformal_set||[]).join(", ")+"} (λ="+Number(data.crc_lambda||0).toFixed(2)+")")
    );
  }
  if(popBody){
    popBody.replaceChildren();
    const g=element("div",undefined,"receipt-grid");
    const p1=element("div",undefined,"receipt-box");
    p1.append(
      element("strong","Fisher–Rao Geodesic Arc Matrix d_FR (rad)"),
      element("div","d_FR(A, B) = "+Number(fr[0]?.[1]||0).toFixed(3)+" · d_FR(A, C) = "+Number(fr[0]?.[2]||0).toFixed(3)),
      element("div","d_FR(B, C) = "+Number(fr[1]?.[2]||0).toFixed(3)+" (max π ≈ 3.142)"),
      element("div","Fréchet Suppression = "+percentage(ig.geodesic_suppression_ratio))
    );
    const p2=element("div",undefined,"receipt-box");
    p2.append(
      element("strong","Tangent-Space Certified TV Radius r^TV"),
      element("div","Margin γ(P) = "+Number(tr.logit_margin||0).toFixed(2)+" logits"),
      element("div","Span ||P_tan ∇γ|| = ["+(tr.tangent_span_sensitivity||[0,0,0]).map(x=>Number(x).toFixed(2)).join(", ")+"]"),
      element("div","Certified r^TV = ["+tvRad.map(x=>Number(x).toFixed(2)).join(", ")+"]")
    );
    const p3=element("div",undefined,"receipt-box");
    p3.append(
      element("strong","Simplex Vertex Breakdown TV Profile"),
      element("div","Singleton TV = "+(vb.min_singleton_breakdown_tv!=null?Number(vb.min_singleton_breakdown_tv).toFixed(3):"Immune (No vertex flip)")),
      element("div","Pair Coalition TV = "+(vb.min_pair_breakdown_tv!=null?Number(vb.min_pair_breakdown_tv).toFixed(3):"—")),
      element("div","Grand Coalition TV = "+(vb.grand_coalition_breakdown_tv!=null?Number(vb.grand_coalition_breakdown_tv).toFixed(3):"—")+" → Class "+(vb.grand_coalition?.adversarial_prediction??'—'))
    );
    g.append(p1,p2,p3);
    popBody.append(g);
  }
}
function scheduleLivePredict(){clearTimeout(pgDebounceTimer);pgDebounceTimer=setTimeout(runLivePredict,80);}
async function loadPresetDigit(digit){
  try{
    const seedVal=byId("seed-select")?.value;
    const params=new URLSearchParams({
      digit:String(digit),
      source:byId("pg-source-select")?.value||"0",
      operation:byId("pg-op-select")?.value||"0",
      tamper_mode:pgTamper,
      arm:byId("pg-arm-select")?.value||"tether",
      alpha:String(pgGetAlpha().toFixed(3)),
      defense:byId("pg-defense")?.checked?"1":"0",
      context_mode:byId("pg-context-mode")?.value||"neutral"
    });
    if(seedVal)params.append("seed",seedVal);
    const res=await api("/api/live_predict?"+params);
    const img=res.images?.[res.source];
    if(Array.isArray(img)&&img.length===64){
      for(let i=0;i<64;i++)pgPixels[i]=Number(img[i])||0;
      if(window.pgRedrawCanvas)window.pgRedrawCanvas();
    }
    renderLiveOutput(res);
    await runLiveArena();
  }catch(e){errorNotice("pg-error",e.message);}
}
async function runLivePredict(){
  if(!study||study.status!=="complete")return;
  const body={
    pixels:Array.from(pgPixels),
    digit:pgDigit,
    source:Number(byId("pg-source-select")?.value||0),
    operation:Number(byId("pg-op-select")?.value||0),
    tamper_mode:pgTamper,
    corrupted_class:Number(byId("pg-corrupt-class")?.value||4),
    confidence:Number(byId("pg-confidence")?.value||0.99),
    arm:byId("pg-arm-select")?.value||"tether",
    alpha:Number(pgGetAlpha().toFixed(3)),
    defense:Boolean(byId("pg-defense")?.checked),
    context_mode:byId("pg-context-mode")?.value||"neutral"
  };
  const seedVal=byId("seed-select")?.value;
  if(seedVal)body.seed=Number(seedVal);
  byId("pg-status").textContent="Running live inference…";
  try{
    const [predRes, arenaRes] = await Promise.all([
      fetch("/api/live_predict",{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(body),cache:"no-store"
      }),
      fetch("/api/live_arena",{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(body),cache:"no-store"
      })
    ]);
    const data=await predRes.json();
    if(!predRes.ok)throw new Error(data.error||"Inference error");
    renderLiveOutput(data);
    if(arenaRes.ok){
      const arenaData=await arenaRes.json();
      renderLiveArena(arenaData);
    }
    byId("pg-status").textContent="Ready · "+(armNames[body.arm]||body.arm);
    errorNotice("pg-error","");
  }catch(err){
    byId("pg-status").textContent="Inference error";
    errorNotice("pg-error",err.message);
  }
}
async function runLiveArena(){
  if(!study||study.status!=="complete")return;
  const body={
    pixels:Array.from(pgPixels),
    digit:pgDigit,
    source:Number(byId("pg-source-select")?.value||0),
    operation:Number(byId("pg-op-select")?.value||0),
    tamper_mode:pgTamper,
    corrupted_class:Number(byId("pg-corrupt-class")?.value||4),
    confidence:Number(byId("pg-confidence")?.value||0.99),
    alpha:Number(pgGetAlpha().toFixed(3)),
    defense:Boolean(byId("pg-defense")?.checked),
    context_mode:byId("pg-context-mode")?.value||"neutral"
  };
  const seedVal=byId("seed-select")?.value;
  if(seedVal)body.seed=Number(seedVal);
  try{
    const res=await fetch("/api/live_arena",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body),cache:"no-store"
    });
    if(res.ok){
      renderLiveArena(await res.json());
    }
  }catch(_){}
}
function renderLiveOutput(data){
  lastLiveData=data;
  const isAbstain=Boolean(data.abstain);
  byId("pg-pred-digit").textContent=isAbstain?"∅":String(data.prediction??"—");
  byId("pg-raw-digit").textContent=String(data.prediction??"—");
  byId("pg-decision-badge").textContent=isAbstain?"SAFE ABSTAIN":"TRUSTED";
  byId("pg-decision-badge").className="tag "+(isAbstain?"wrong":"safe");
  byId("pg-abstain-banner")?.classList.toggle("hidden",!isAbstain);
  renderConformalChips(byId("pg-conformal-set"), data.wcp_conformal_set||data.conformal_set, isAbstain, data.prediction);
  if(byId("pg-conformal-meta") && finite(data.temperature) && finite(data.conformal_q_hat)){
    const qw=finite(data.wcp_q_hat)?data.wcp_q_hat:data.conformal_q_hat;
    byId("pg-conformal-meta").textContent="T* = "+data.temperature.toFixed(2)+" · q̂_w = "+qw.toFixed(3);
  }
  const trust=Number(data.epistemic_trust??0);
  byId("pg-trust-percent").textContent=percentage(trust);
  byId("pg-meter-fill").style.width=Math.max(0,Math.min(100,trust*100))+"%";
  byId("pg-meter-score").textContent="τ = "+trust.toFixed(3);
  const probs=Array.isArray(data.probabilities)?data.probabilities:[];
  byId("pg-top-conf").textContent=finite(probs[data.prediction])?percentage(probs[data.prediction]):"—";
  const probChart=byId("pg-probabilities");probChart.replaceChildren();
  for(let i=0;i<10;i++){
    const val=probs[i], col=element("div",undefined,"prob-column"+(i===data.prediction?" winner":"")), fill=element("i",undefined,"prob-fill");
    fill.style.height=(finite(val)?Math.max(2,Math.min(75,val*75)):2)+"px";
    col.title="Digit "+i+": "+percentage(val);col.append(fill,element("span",String(i)));probChart.append(col);
  }
  const uncBars=byId("pg-uncertainty-bars");
  if(uncBars){
    uncBars.replaceChildren();
    const items=[
      ["Aleatoric Entropy", Number(data.aleatoric_uncertainty||0), "#578776"],
      ["Conflict (JSD)", Number(data.epistemic_conflict_jsd||0), "#ce874a"],
      ["Forensic Spoof (1−τ)", Number(data.forensic_spoof_risk||0), "#c0392b"]
    ];
    items.forEach(([label,val,color])=>{
      const row=element("div",undefined,"saliency-row");
      row.style.gridTemplateColumns="110px minmax(20px,1fr) 48px";
      row.append(element("span",label));
      const track=element("div",undefined,"saliency-track"), fill=element("div",undefined,"saliency-fill");
      fill.style.width=Math.max(0,Math.min(100,val*100))+"%";
      fill.style.background=color;
      track.append(fill);
      row.append(track,element("span",percentage(val),"saliency-val"));
      uncBars.append(row);
    });
  }
  const src=Number(data.source||0);
  const expSal=(data.ig_expert_saliency||data.expert_saliency)?.[src]||[0,0,0];
  const maxSal=Math.max(0.0001,...expSal);
  if(byId("pg-ig-err") && finite(data.ig_completeness_error)){
    byId("pg-ig-err").textContent="Δ = "+data.ig_completeness_error.toFixed(4);
  }
  const salContainer=byId("pg-saliency-breakdown");salContainer.replaceChildren();
  ["Expert A","Expert B","Expert C"].forEach((name,j)=>{
    const sVal=expSal[j]||0, row=element("div",undefined,"saliency-row");
    row.append(element("span",name));
    const track=element("div",undefined,"saliency-track"), fill=element("div",undefined,"saliency-fill");
    fill.style.width=Math.min(100,(sVal/maxSal)*100)+"%";
    track.append(fill);row.append(track,element("span",sVal.toFixed(3),"saliency-val"));
    salContainer.append(row);
  });
  const pgShapley=byId("pg-shapley-breakdown");
  if(pgShapley && Array.isArray(data.shapley_values)){
    pgShapley.replaceChildren();
    const phis=data.shapley_values;
    const maxPhi=Math.max(0.0001,...phis.map(x=>Math.abs(Number(x)||0)));
    if(byId("pg-shapley-meta")){
      byId("pg-shapley-meta").textContent="Σφ = "+(phis.reduce((a,b)=>a+b,0)).toFixed(2);
    }
    ["Expert A (φ₀)","Expert B (φ₁)","Expert C (φ₂)"].forEach((name,j)=>{
      const val=Number(phis[j]||0), row=element("div",undefined,"saliency-row");
      row.style.gridTemplateColumns="72px minmax(20px,1fr) 52px";
      row.append(element("span",name));
      const track=element("div",undefined,"saliency-track"), fill=element("div",undefined,"saliency-fill");
      fill.style.width=Math.min(100,(Math.abs(val)/maxPhi)*100)+"%";
      fill.style.background=val>=0?"#578776":"#c0392b";
      track.append(fill);row.append(track,element("span",(val>=0?"+":"")+val.toFixed(2),"saliency-val"));
      pgShapley.append(row);
    });
  }
  renderPidBars(byId("pg-pid-breakdown"), byId("pg-pid-meta"), data.pid);
  const medPills=byId("pg-mediation-pills");
  if(medPills && data.causal_mediation){
    medPills.replaceChildren();
    const med=data.causal_mediation;
    const critAlpha=data.critical_corruption_threshold;
    if(byId("pg-mediation-meta")){
      byId("pg-mediation-meta").textContent="ρ_vis = "+percentage(med.visual_dominance_ratio)+" · α* = "+(finite(critAlpha)?percentage(critAlpha):"—");
    }
    const p1=element("div",undefined,"trust-pill");
    p1.append(element("strong","Visual NDE"),element("span",(med.nde_visual>=0?"+":"")+Number(med.nde_visual||0).toFixed(2)+" logits"));
    const p2=element("div",undefined,"trust-pill"+(Math.abs(med.nie_shortcut||0)>1.0?" damaged":""));
    p2.append(element("strong","Shortcut NIE"),element("span",(med.nie_shortcut>=0?"+":"")+Number(med.nie_shortcut||0).toFixed(2)+" logits"));
    const p3=element("div",undefined,"trust-pill");
    p3.append(element("strong","Crit Breakdown α*"),element("span",finite(critAlpha)?("Flip at "+percentage(critAlpha)):"Unbreakable (1.0)"));
    medPills.append(p1,p2,p3);
  }
  const damagedSet=new Set(Array.isArray(data.damaged_experts)?data.damaged_experts:[]);
  byId("pg-damaged-notice").textContent=damagedSet.size?(damagedSet.size+" Corrupted"+(data.defense?" · Sanitized":"")):"3 Intact";
  const pills=byId("pg-trust-pills");pills.replaceChildren();
  for(let j=0;j<3;j++){
    const tVal=data.trust_score?.[src]?.[j], wVal=data.expert_weights?.[src]?.[j], mDist=data.mahalanobis_distance?.[j], isDmg=damagedSet.has(j);
    const pill=element("div",undefined,"trust-pill"+(isDmg?" damaged":""));
    const mStr=finite(mDist)?" · Dₘ "+mDist.toFixed(1):"";
    pill.append(element("strong","Expert "+["A","B","C"][j]+(isDmg?" (corrupted)":"")),element("span","τ "+percentage(tVal)+mStr+" · Wt "+percentage(wVal)));
    pills.append(pill);
  }
  renderGeodesicWidget(data);
}
function renderLiveArena(arena){
  if(!arena||!Array.isArray(arena.arms))return;
  if(byId("arena-target-badge")){
    byId("arena-target-badge").textContent="CLEAN TARGET: "+arena.expected_target+" (DIGIT "+arena.clean_digit+")";
  }
  const tbody=byId("arena-tbody");
  if(tbody){
    tbody.replaceChildren();
    const order=["tether","invariance_only","iit","effect_only","dropout","passive","unpaired"];
    const sorted=[...arena.arms].sort((a,b)=>order.indexOf(a.arm)-order.indexOf(b.arm));
    sorted.forEach(item=>{
      const tr=element("tr",undefined,item.arm==="tether"?"row-tether":"");
      tr.append(element("th",armNames[item.arm]||item.arm));
      const predText=item.abstain?"∅ (Abstain)":"ŷ = "+item.prediction;
      tr.append(element("td",predText));
      const cSet=item.wcp_conformal_set||item.conformal_set;
      const setStr=item.abstain||!cSet?.length?"∅":"{"+cSet.join(", ")+"}";
      tr.append(element("td",setStr));
      tr.append(element("td",percentage(item.target_probability)));
      const visDom=finite(item.visual_dominance_ratio)?percentage(item.visual_dominance_ratio):percentage(item.intact_weight_share);
      const critA=finite(item.critical_corruption_threshold)?percentage(item.critical_corruption_threshold):"—";
      tr.append(element("td",visDom+" / "+critA));
      const tdStatus=element("td");
      const badge=element("span",item.abstain?"SAFE ABSTAIN":(item.survived?"SURVIVED":"HIJACKED"),"tag "+(item.survived?"safe":"danger"));
      tdStatus.append(badge);
      tr.append(tdStatus);
      tbody.append(tr);
    });
  }
  const svg=byId("arena-sweep-svg");
  if(svg && arena.sweep && arena.sweep.curves){
    const confs=arena.sweep.confidences||[];
    const curves=arena.sweep.curves;
    if(byId("arena-sweep-tag")){
      byId("arena-sweep-tag").textContent=human(arena.sweep.mode).toUpperCase()+" SWEEP";
    }
    const w=380, h=210, padL=38, padR=16, padT=16, padB=28;
    const plotW=w-padL-padR, plotH=h-padT-padB;
    const xPos=(idx)=>padL+(confs.length>1?(idx/(confs.length-1))*plotW:plotW/2);
    const yPos=(val)=>padT+(1-Math.max(0,Math.min(1,val)))*plotH;
    let markup=`<rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#fff" stroke="#e6e7e3" rx="4"/>`;
    [0,0.5,1.0].forEach(tick=>{
      const y=yPos(tick);
      markup+=`<line x1="${padL}" y1="${y}" x2="${w-padR}" y2="${y}" stroke="#f0f0ec" stroke-dasharray="3,3"/>`;
      markup+=`<text x="${padL-6}" y="${y+3}" text-anchor="end" font-size="8" fill="#8c9097">${Math.round(tick*100)}%</text>`;
    });
    confs.forEach((c,idx)=>{
      const x=xPos(idx);
      markup+=`<text x="${x}" y="${h-8}" text-anchor="middle" font-size="8" fill="#8c9097">${(c*100).toFixed(0)}%</text>`;
    });
    const colors={tether:"#7358cf",iit:"#578776",unpaired:"#ce874a",passive:"#95a5a6"};
    ["passive","unpaired","iit","tether"].forEach(armKey=>{
      const series=curves[armKey]?.target_probabilities;
      if(!Array.isArray(series)||!series.length)return;
      const pts=series.map((v,i)=>`${xPos(i).toFixed(1)},${yPos(v).toFixed(1)}`).join(" ");
      const strokeW=armKey==="tether"?2.6:1.7;
      markup+=`<polyline fill="none" stroke="${colors[armKey]}" stroke-width="${strokeW}" points="${pts}"/>`;
      series.forEach((v,i)=>{
        markup+=`<circle cx="${xPos(i).toFixed(1)}" cy="${yPos(v).toFixed(1)}" r="${armKey==="tether"?3.2:2.2}" fill="${colors[armKey]}"/>`;
      });
    });
    svg.innerHTML=markup;
  }
}

loadStudy().then(()=>pgInitCanvas());
