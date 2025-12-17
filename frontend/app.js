let watchlist = [
  "AAPL",
  "MSFT",
  "AMZN",
  "GOOGL",
  "TSLA",
  "NVDA",
  "JPM",
  "META",
  "INTC",
  "KO",
];
const cardsContainer = document.getElementById("cardsContainer");
const selectedTickerEl = document.getElementById("selectedTicker");
const selectedPriceEl = document.getElementById("selectedPrice");
const selectedChangeEl = document.getElementById("selectedChange");
const chartNotice = document.getElementById("chartNotice");
const predictMsg = document.getElementById("predictMsg");
const baseUrlInput = document.getElementById("baseUrl");

let mainChart = null;
let currentSymbol = "AAPL";
let latestDaily = { labels: [], closes: [] };

function createCard(symbol) {
  const el = document.createElement("button");
  el.className =
    "glass rounded-xl p-4 text-left transform hover:-translate-y-1 transition-shadow shadow-sm";
  el.setAttribute("data-symbol", symbol);
  el.innerHTML = `
    <div class="flex justify-between items-start">
      <div>
        <div class="text-sm text-slate-300">${symbol}</div>
        <div class="text-xl font-bold mt-1" data-price>--</div>
        <div class="text-xs text-slate-400 mt-1" data-pct>--</div>
      </div>
      <div class="text-right">
        <div class="text-xs text-slate-400">Live</div>
        <div class="text-sm text-slate-300" data-last>--</div>
      </div>
    </div>`;
  el.addEventListener("click", () => selectSymbol(symbol, el));
  return el;
}

function renderWatchlist() {
  cardsContainer.innerHTML = "";
  watchlist.forEach((s) => cardsContainer.appendChild(createCard(s)));
}

async function fetchJSON(path) {
  const base = baseUrlInput.value.replace(/\/+$/, "");
  const res = await fetch(`${base}${path}`);
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`${res.status} ${t}`);
  }
  return res.json();
}

async function updateQuote(symbol) {
  try {
    const data = await fetchJSON(
      `/api/quote?symbol=${encodeURIComponent(symbol)}`
    );
    if (!data || data.error) return;
    const card = cardsContainer.querySelector(`[data-symbol="${symbol}"]`);
    if (!card) return;
    const priceEl = card.querySelector("[data-price]");
    const pctEl = card.querySelector("[data-pct]");
    const lastEl = card.querySelector("[data-last]");

    priceEl.textContent = `$${data.price.toFixed(2)}`;
    pctEl.textContent = `${data.change >= 0 ? "+" : ""}${data.change.toFixed(
      2
    )} (${data.change_percent >= 0 ? "+" : ""}${data.change_percent.toFixed(
      2
    )}%)`;
    pctEl.classList.remove("positive", "negative", "neutral");
    if (data.change_percent > 0.1) pctEl.classList.add("positive");
    else if (data.change_percent < -0.1) pctEl.classList.add("negative");
    else pctEl.classList.add("neutral");
    lastEl.textContent = new Date(data.ts * 1000).toLocaleTimeString();

    if (symbol === currentSymbol) {
      selectedPriceEl.textContent = `$${data.price.toFixed(2)}`;
      selectedChangeEl.textContent = `${
        data.change >= 0 ? "▲" : "▼"
      } ${data.change.toFixed(2)} (${
        data.change_percent >= 0 ? "+" : ""
      }${data.change_percent.toFixed(2)}%)`;
      selectedChangeEl.className =
        data.change_percent > 0.1
          ? "positive"
          : data.change_percent < -0.1
          ? "negative"
          : "neutral";
    }
  } catch (e) {
    console.error(e);
  }
}

async function loadDaily(symbol) {
  console.log("symbol", symbol)
  chartNotice.textContent = "Loading daily data...";
  try {
    const daily = await fetchJSON(
      `/api/daily?symbol=${encodeURIComponent(symbol)}&outputsize=compact`
    );
    console.log("daily", daily)
    if (
      !daily ||
      daily.error ||
      !Array.isArray(daily.labels) ||
      !Array.isArray(daily.closes)
    ) {
      chartNotice.textContent =
        daily && daily.error ? daily.error : "No daily data available.";
      return;
    }
    latestDaily = { labels: daily.labels, closes: daily.closes };
    renderChart(symbol, latestDaily.labels, latestDaily.closes);
    chartNotice.textContent = "Daily data loaded.";
  } catch (e) {
    console.error("Daily data error:", e);
    chartNotice.textContent = "Failed to load daily data.";
  }
}

function renderChart(symbol, labels, closes, forecast = null) {
  const canvas = document.getElementById("mainChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (mainChart) mainChart.destroy();

  // Create Gradients
  const gradientHist = ctx.createLinearGradient(0, 0, 0, 400);
  gradientHist.addColorStop(0, "rgba(99, 102, 241, 0.5)"); // Indigo
  gradientHist.addColorStop(1, "rgba(99, 102, 241, 0.0)");

  const gradientForecast = ctx.createLinearGradient(0, 0, 0, 400);
  gradientForecast.addColorStop(0, "rgba(34, 211, 238, 0.5)"); // Cyan
  gradientForecast.addColorStop(1, "rgba(34, 211, 238, 0.0)");

  const datasets = [
    {
      label: `${symbol} (History)`,
      data: closes,
      tension: 0.3,
      borderWidth: 3,
      pointRadius: 0,
      pointHoverRadius: 6,
      fill: true,
      backgroundColor: gradientHist,
      borderColor: "#6366f1", // Indigo 500
    },
  ];

  let fullLabels = labels;

  if (forecast && forecast.points.length) {
    const fLabels = forecast.points.map((p) => p.ds);
    fullLabels = [...labels, ...fLabels];

    // Gap handling: Join last historical point to first forecast point
    // We add nulls to align the forecast line
    const emptyHistory = new Array(closes.length - 1).fill(null);
    // The "transition point" is the last close
    const lastClose = closes[closes.length - 1]; 
    
    // Forecast Line (Premium Cyan/Teal)
    datasets.push({
      label: "AI Forecast",
      data: [...emptyHistory, lastClose, ...forecast.points.map((p) => p.yhat)],
      tension: 0.4,
      borderWidth: 3,
      pointRadius: 0,
      pointHoverRadius: 6,
      borderDash: [0, 0], // Solid line for impact
      fill: true,
      backgroundColor: gradientForecast,
      borderColor: "#22d3ee", // Cyan 400
      shadowBlur: 10,
      shadowColor: "#22d3ee"
    });

    // Confidence Band (Subtle)
    datasets.push({
      label: "Confidence Band",
      data: [...emptyHistory, lastClose, ...forecast.points.map((p) => p.yhat_upper)],
      tension: 0.4,
      pointRadius: 0,
      borderWidth: 0,
      fill: "+1", // Fill to next dataset
      backgroundColor: "rgba(34, 211, 238, 0.1)",
    });
    datasets.push({
      label: "Lower Band",
      data: [...emptyHistory, lastClose, ...forecast.points.map((p) => p.yhat_lower)],
      tension: 0.4,
      pointRadius: 0,
      borderWidth: 0,
      fill: false,
    });
  }

  mainChart = new Chart(ctx, {
    type: "line",
    data: { labels: fullLabels, datasets },
    options: {
      responsive: true,
      plugins: { 
        legend: { labels: { color: "#cbd5e1" } },
        tooltip: { 
            mode: "index", 
            intersect: false,
            backgroundColor: "rgba(15, 23, 42, 0.9)",
            titleColor: "#f8fafc",
            bodyColor: "#f8fafc",
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1
        }
      },
      interaction: { mode: "nearest", axis: 'x', intersect: false },
      scales: {
        x: { 
            ticks: { color: "#94a3b8", maxTicksLimit: 8 }, 
            grid: { color: "rgba(255,255,255,0.05)" } 
        },
        y: { 
            ticks: { color: "#94a3b8" }, 
            grid: { color: "rgba(255,255,255,0.05)" } 
        },
      },
    },
  });
}

async function selectSymbol(symbol, cardEl) {
  currentSymbol = symbol;
  selectedTickerEl.textContent = symbol;
  cardsContainer
    .querySelectorAll("button")
    .forEach((b) => b.classList.remove("ring-2", "ring-indigo-400"));
  cardEl.classList.add("ring-2", "ring-indigo-400");

  await Promise.all([updateQuote(symbol), loadDaily(symbol)]);
}

async function predict(symbol, days) {
  predictMsg.textContent = "Training Prophet model...";
  try {
    const data = await fetchJSON(
      `/api/predict?symbol=${encodeURIComponent(symbol)}&days=${days}`
    );

    if (!data || !data.forecast || !Array.isArray(data.forecast)) {
      throw new Error(data.error || "No forecast data returned from backend");
    }

    const points = data.forecast.map((p) => ({
      ds: p.ds,
      yhat: p.yhat,
      yhat_lower: p.yhat_lower,
      yhat_upper: p.yhat_upper,
    }));

    renderChart(symbol, latestDaily.labels, latestDaily.closes, { points });
    predictMsg.textContent = `Forecast ready for ${days} day(s).`;
  } catch (e) {
    console.error("Prediction error:", e);
    predictMsg.textContent = "Prediction error: " + e.message;
  }
}

document.getElementById("applyWatchlistBtn").addEventListener("click", () => {
  const v = document.getElementById("watchlistInput").value.trim();
  const arr = v
    .split(",")
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
  if (!arr.length) return;
  watchlist = arr;
  renderWatchlist();
  const first = cardsContainer.querySelector(`[data-symbol="${watchlist[0]}"]`);
  if (first) first.click();
});

document.getElementById("refreshNowBtn").addEventListener("click", () => {
  watchlist.forEach((s) => updateQuote(s));
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const days = Math.max(
    1,
    Math.min(30, parseInt(document.getElementById("days").value || "7", 10))
  );
  await predict(currentSymbol, days);
});

async function searchGlossary() {
  const term = document.getElementById("glossaryInput").value.trim();
  const resEl = document.getElementById("glossaryResult");
  if (!term) return;
  
  resEl.textContent = "Searching...";
  resEl.className = "mt-2 text-xs text-slate-400 italic";
  
  try {
    const base = baseUrlInput.value.replace(/\/+$/, "");
    const res = await fetch(`${base}/api/glossary?term=${encodeURIComponent(term)}`);
    const data = await res.json();
    
    if (data.error) {
       resEl.textContent = "Definition not found.";
       resEl.className = "mt-2 text-xs text-red-400";
    } else {
       resEl.textContent = data.definition;
       resEl.className = "mt-2 text-xs text-emerald-400 font-medium";
    }
  } catch (e) {
    console.error(e);
    resEl.textContent = "Error fetching definition";
  }
}
document.getElementById("searchGlossaryBtn").addEventListener("click", searchGlossary);

async function init() {
  renderWatchlist();
  const first = cardsContainer.querySelector(`[data-symbol="${watchlist[0]}"]`);
  if (first) first.click();
  watchlist.forEach((s) => updateQuote(s));
}
init();


// ----------------------------------------------------
// Backtesting Logic
// ----------------------------------------------------
let backtestChartVar = null;

async function runBacktest() {
  const mode = document.getElementById("btMode").value;
  const short = parseInt(document.getElementById("btShort").value) || 10;
  const long = parseInt(document.getElementById("btLong").value) || 50;
  const capital = parseFloat(document.getElementById("btCapital").value) || 10000;
  const symbol = mode === "REAL" ? currentSymbol : "SYNTHETIC";
  
  const msgEl = document.getElementById("btMsg");
  msgEl.textContent = "Running simulation...";
  
  try {
     const base = baseUrlInput.value.replace(/\/+$/, "");
     const res = await fetch(`${base}/api/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
           symbol,
           sma_short: short,
           sma_long: long,
           capital,
           days: 365
        })
     });
     
     const data = await res.json();
     if (data.error) throw new Error(data.error);
     
     // Update metrics
     document.getElementById("btReturn").textContent = `${data.metrics.total_return}%`;
     document.getElementById("btReturn").className = `text-2xl font-bold ${data.metrics.total_return >= 0 ? "text-green-400" : "text-red-400"}`;
     document.getElementById("btDrawdown").textContent = `${data.metrics.max_drawdown}%`;
     document.getElementById("btTrades").textContent = data.metrics.trades_count;
     
     // Render Chart
     renderBacktestChart(data.equity_curve);
     msgEl.textContent = "Simulation complete.";
     
  } catch(e) {
     console.error(e);
     msgEl.textContent = "Error: " + e.message;
  }
}

function renderBacktestChart(equityCurve) {
  const ctx = document.getElementById("backtestChart").getContext("2d");
  if (backtestChartVar) backtestChartVar.destroy();
  
  const labels = equityCurve.map(p => p.time);
  const data = equityCurve.map(p => p.value);
  
  backtestChartVar = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Portfolio Value ($)',
        data: data,
        borderColor: '#ec4899', // pink-500
        backgroundColor: 'rgba(236, 72, 153, 0.1)',
        borderWidth: 2,
        fill: true,
        pointRadius: 0,
        tension: 0.1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
         x: { ticks: { color: "#94a3b8", maxTicksLimit: 10 }, grid: { display: false } },
         y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

document.getElementById("runBacktestBtn").addEventListener("click", runBacktest);


// Chatbot Logic
const chatbot = document.getElementById("chatbot");
const chatbotTriggerBtn = document.getElementById("chatbotTriggerBtn");
const closeChatBtn = document.getElementById("closeChatBtn");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");

function toggleChat() {
  chatbot.classList.toggle("hidden");
  if (!chatbot.classList.contains("hidden")) {
    chatMessages.scrollTop = chatMessages.scrollHeight;
    chatInput.focus();
  }
}

if (chatbotTriggerBtn) chatbotTriggerBtn.addEventListener("click", toggleChat);
if (closeChatBtn) closeChatBtn.addEventListener("click", toggleChat);

function appendMessage(role, text, details = null) {
  const div = document.createElement("div");
  div.className = `max-w-[85%] p-3 rounded-lg text-sm ${
    role === "user"
      ? "self-end bg-blue-500 text-white"
      : "self-start bg-slate-700 text-white"
  }`;
  
  const contentDiv = document.createElement("div");
  contentDiv.textContent = text;
  div.appendChild(contentDiv);

  if (details) {
    const detailsDiv = document.createElement("div");
    detailsDiv.className = "mt-1 text-xs text-slate-300 whitespace-pre-wrap";
    detailsDiv.textContent = details;
    div.appendChild(detailsDiv);
  }

  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

if (chatForm) {
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    // Add user message
    appendMessage("user", text);
    chatInput.value = "";

    // Show thinking
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "self-start text-xs text-slate-400 ml-2 animate-pulse";
    loadingDiv.textContent = "Thinking...";
    loadingDiv.id = "chatLoading";
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
      const base = baseUrlInput.value.replace(/\/+$/, "");
      const res = await fetch(`${base}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      const data = await res.json();
      
      // Remove loading
      const loadingEl = document.getElementById("chatLoading");
      if (loadingEl) loadingEl.remove();

      if (data.error) {
        appendMessage("assistant", `Error: ${data.error}`);
      } else {
        const sentiment = data.sentiment;
        const s = sentiment["FinBERT Scores"] || {};
        const breakdown = Object.entries(s)
           .map(([k, v]) => `${k}: ${(v*100).toFixed(1)}%`)
           .join(", ");
           
        const details = `\nFinBERT: ${sentiment["FinBERT Label"]} (Conf: ${sentiment["FinBERT Confidence"]})\n[${breakdown}]\nVADER: ${sentiment["VADER Compound"]}`;
        appendMessage("assistant", data.response, details);
      }
    } catch (err) {
      const loadingEl = document.getElementById("chatLoading");
      if (loadingEl) loadingEl.remove();
      appendMessage("assistant", "Failed to reach the brain. Is the backend running?");
      console.error(err);
    }
  });
}
