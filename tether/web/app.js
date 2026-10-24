"use strict";
const byId = (id) => document.getElementById(id);
const arms = ["passive", "dropout", "iit", "effect_only", "invariance_only", "tether"];
const armNames = {passive:"Passive",dropout:"Donor dropout",iit:"IIT",effect_only:"Effect only",invariance_only:"Invariance only",tether:"Tether"};
const modes = {
  factual:"Unmodified donor outputs and context from a frozen test task.",
  relevant:"Replace the queried source. The correct answer changes; the old context stays.",
  irrelevant:"Replace a source outside the query. The correct answer stays the same.",
  damaged:"Corrupt one expert posterior. The task and its correct answer stay the same.",
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
function errorNotice(id, message) { const node=byId(id); node.textContent=message || ""; node.classList.toggle("hidden", !message); }
async function api(path) {
  const response = await fetch(path, {cache:"no-store"});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Saved study could not be loaded.");
  return data;
}
function metricsFor(arm, condition) {
  return (study?.rows || []).filter(row => row.arm === arm).map(row => row.metrics?.[condition]).filter(Boolean);
}
function renderCondition() {
  const condition=byId("condition-select").value;
  byId("condition-name").textContent=human(condition) || "—";
  const records=metricsFor("tether",condition);
  byId("hero-accuracy").textContent=percentage(mean(records.map(x=>x.accuracy)));
  byId("hero-gap").textContent=points(mean(records.map(x=>x.donor_erasure_gap)));
  byId("condition-caption").textContent=records.length ? records.length+" recorded seed"+(records.length===1?"":"s")+" · frozen evaluation" : "No Tether metrics for this condition";
  const bars=byId("arm-bars"); bars.replaceChildren();
  arms.forEach(arm=>{
    const values=metricsFor(arm,condition);
    if (!values.length) return;
    const accuracy=mean(values.map(x=>x.accuracy));
    const row=element("div",undefined,"arm-row "+arm);
    row.append(element("span",armNames[arm],"arm-label"));
    const track=element("div",undefined,"bar-track"), fill=element("div",undefined,"bar-fill");
    fill.style.width=(finite(accuracy)?Math.max(0,Math.min(100,accuracy*100)):0)+"%";
    track.append(fill); row.append(track,element("span",percentage(accuracy),"arm-value")); bars.append(row);
  });
  if (!bars.children.length) bars.append(element("p","No metrics recorded for this condition.","empty-state"));
  const paired=byId("paired-results"); paired.replaceChildren();
  const comparisons=(study?.comparisons || []).filter(x=>String(x.metric || "").includes(condition));
  const sorted=[...comparisons].sort((a,b)=>(a.arm==="tether"&&a.baseline==="iit"?-1:0)-(b.arm==="tether"&&b.baseline==="iit"?-1:0));
  sorted.slice(0,4).forEach(item=>{
    const row=element("div",undefined,"paired-item"), title=element("div",undefined,"paired-title");
    title.append(element("span",(armNames[item.arm]||human(item.arm))+" − "+(armNames[item.baseline]||human(item.baseline))),element("strong",points(item.mean_difference)));
    row.append(title);
    const interval=Array.isArray(item.interval) && item.interval.length===2 ? "["+points(item.interval[0])+", "+points(item.interval[1])+"]" : "interval not recorded";
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
    byId("study-config").textContent=study.rows.length+" arm records · "+human(config.phase||config.stage||"saved evaluation");
    const decision=study.decision||{};
    byId("hero-decision").textContent=decision.promoted===true ? "Promoted" : "Unpromoted";
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
    errorNotice("study-error",error.message+" Start this viewer with --study pointing to a completed study directory.");
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
  byId("mode-description").textContent=modes[mode];
  byId("example-status").textContent="Test #"+data.index+" · "+armNames[data.arm || byId("arm-select").value];
}
async function loadExample() {
  if(study?.status!=="complete") return;
  const token=++exampleRequest;
  const index=Number(byId("example-index").value);
  if(!Number.isInteger(index)||index<0||index>1000000){errorNotice("example-error","Choose a non-negative whole example index.");return;}
  byId("example-status").textContent="Running saved inference…";
  try {
    const query=new URLSearchParams({index:String(index),arm:byId("arm-select").value,mode});
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
byId("arm-select").addEventListener("change",loadExample);
byId("example-index").addEventListener("change",loadExample);
byId("previous-example").addEventListener("click",()=>{byId("example-index").value=Math.max(0,Number(byId("example-index").value)-1);loadExample();});
byId("next-example").addEventListener("click",()=>{byId("example-index").value=Math.min(1000000,Number(byId("example-index").value)+1);loadExample();});
document.querySelectorAll("[data-mode]").forEach(button=>button.addEventListener("click",()=>{
  mode=button.dataset.mode;
  document.querySelectorAll("[data-mode]").forEach(item=>{item.classList.toggle("selected",item===button);item.setAttribute("aria-pressed",String(item===button));});
  byId("mode-description").textContent=modes[mode];loadExample();
}));
document.querySelectorAll(".nav-item").forEach(link=>link.addEventListener("click",()=>{document.querySelectorAll(".nav-item").forEach(item=>item.classList.toggle("active",item===link));}));
document.querySelectorAll("canvas").forEach(canvas=>drawDigit(canvas,null));
loadStudy();

