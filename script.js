// Real-time data storage
let realTimeData = {
    'AAPL': { 
        price: 175.43, 
        change: 2.15, 
        changePercent: 1.24,
        company: 'Apple Inc.',
        sector: 'Technology',
        marketCap: '2.8T',
        volume: '45.2M',
        pe: 28.5
    },
    'GOOGL': { 
        price: 2847.50, 
        change: -15.30, 
        changePercent: -0.53,
        company: 'Alphabet Inc.',
        sector: 'Technology',
        marketCap: '1.9T',
        volume: '1.2M',
        pe: 24.1
    },
    'TSLA': { 
        price: 248.90, 
        change: 8.75, 
        changePercent: 3.64,
        company: 'Tesla Inc.',
        sector: 'Automotive',
        marketCap: '790B',
        volume: '85.3M',
        pe: 65.2
    },
    'MSFT': { 
        price: 342.18, 
        change: 4.22, 
        changePercent: 1.25,
        company: 'Microsoft Corp.',
        sector: 'Technology',
        marketCap: '2.5T',
        volume: '28.7M',
        pe: 32.8
    },
    'AMZN': { 
        price: 3156.78, 
        change: -22.45, 
        changePercent: -0.71,
        company: 'Amazon.com Inc.',
        sector: 'E-commerce',
        marketCap: '1.6T',
        volume: '3.1M',
        pe: 58.3
    },
    'NVDA': { 
        price: 875.32, 
        change: 18.90, 
        changePercent: 2.21,
        company: 'NVIDIA Corp.',
        sector: 'Technology',
        marketCap: '2.1T',
        volume: '42.8M',
        pe: 45.7
    }
};

let marketIndices = {
    'SP500': { value: 4200.50, change: 1.2 },
    'NASDAQ': { value: 13150.25, change: -0.8 },
    'DOW': { value: 33850.75, change: 0.5 },
    'VIX': { value: 21.5, change: -2.1 }
};

// Initialize market chart with real-time capability
const ctx = document.getElementById('marketChart').getContext('2d');
let chartData = [];
let chartLabels = [];

// Generate initial chart data
for (let i = 0; i < 20; i++) {
    const time = new Date();
    time.setMinutes(time.getMinutes() - (20 - i) * 5);
    chartLabels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
    chartData.push(4200 + Math.random() * 50 - 25);
}

const marketChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: chartLabels,
        datasets: [{
            label: 'S&P 500',
            data: chartData,
            borderColor: '#667eea',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            tension: 0.4,
            fill: true,
            pointRadius: 0,
            pointHoverRadius: 5
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 750,
            easing: 'easeInOutQuart'
        },
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                grid: {
                    color: 'rgba(0,0,0,0.1)'
                }
            },
            x: {
                grid: {
                    color: 'rgba(0,0,0,0.1)'
                }
            }
        }
    }
});

// Real-time data update functions
function updateRealTimeData() {
    // Simulate real-time price updates
    Object.keys(realTimeData).forEach(symbol => {
        const stock = realTimeData[symbol];
        const priceChange = (Math.random() - 0.5) * 2; // Random change between -1 and 1
        stock.price += priceChange;
        stock.change += priceChange;
        stock.changePercent = (stock.change / (stock.price - stock.change)) * 100;
        
        // Keep prices realistic
        if (stock.price < 10) stock.price = 10;
        if (stock.price > 5000) stock.price = 5000;
    });

    // Update market indices
    Object.keys(marketIndices).forEach(index => {
        const idx = marketIndices[index];
        const change = (Math.random() - 0.5) * 0.5;
        idx.change += change;
        idx.value += (idx.value * change / 100);
    });

    // Update market overview display
    updateMarketOverview();
}

function updateMarketOverview() {
    const stats = document.querySelectorAll('.stat-item');
    const indices = ['SP500', 'NASDAQ', 'DOW', 'VIX'];
    
    stats.forEach((stat, index) => {
        if (indices[index]) {
            const data = marketIndices[indices[index]];
            const valueEl = stat.querySelector('.stat-value');
            const change = data.change.toFixed(1);
            valueEl.textContent = `${change >= 0 ? '+' : ''}${change}%`;
            valueEl.style.color = change >= 0 ? '#38a169' : '#e53e3e';
        }
    });
}

function predictStock() {
    const symbol = document.getElementById('stockSymbol').value.toUpperCase();
    const resultDiv = document.getElementById('predictionResult');
    
    // Use real-time data if available, otherwise simulate
    let currentPrice, stockName;
    if (realTimeData[symbol]) {
        currentPrice = realTimeData[symbol].price.toFixed(2);
        stockName = getStockName(symbol);
    } else {
        currentPrice = (Math.random() * 200 + 50).toFixed(2);
        stockName = `${symbol} - Stock Analysis`;
    }
    
    const predictedPrice = (parseFloat(currentPrice) * (1 + (Math.random() * 0.2 - 0.1))).toFixed(2);
    const change = ((predictedPrice - currentPrice) / currentPrice * 100).toFixed(2);
    const confidence = (Math.random() * 30 + 60).toFixed(0);
    
    resultDiv.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #2d3748;">${stockName}</div>
                <div style="color: #718096;">Current: $${currentPrice} <span id="live-${symbol}" style="color: #38a169;">●</span></div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.5rem; font-weight: bold; color: ${change >= 0 ? '#38a169' : '#e53e3e'};">$${predictedPrice}</div>
                <div style="color: #718096;">7-day prediction</div>
            </div>
        </div>
        <div style="margin-top: 15px;">
            <div style="font-size: 0.9rem; color: #4a5568;">Confidence: ${confidence}% | Expected ${change >= 0 ? 'gain' : 'loss'}: ${change >= 0 ? '+' : ''}${change}%</div>
            <div style="font-size: 0.8rem; color: #718096; margin-top: 5px;">Last updated: ${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    resultDiv.style.display = 'block';
}

function getStockName(symbol) {
    const names = {
        'AAPL': 'AAPL - Apple Inc.',
        'GOOGL': 'GOOGL - Alphabet Inc.',
        'TSLA': 'TSLA - Tesla Inc.',
        'MSFT': 'MSFT - Microsoft Corp.',
        'AMZN': 'AMZN - Amazon.com Inc.',
        'NVDA': 'NVDA - NVIDIA Corp.'
    };
    return names[symbol] || `${symbol} - Stock Analysis`;
}

function analyzeRisk() {
    const symbol = document.getElementById('riskSymbol').value.toUpperCase();
    
    // Simulate risk analysis with random data
    const volatilityRisk = Math.random() * 100;
    const marketRisk = Math.random() * 100;
    const liquidityRisk = Math.random() * 100;
    
    const getRiskLevel = (value) => {
        if (value < 33) return { level: 'Low', class: 'risk-low' };
        if (value < 66) return { level: 'Medium', class: 'risk-medium' };
        return { level: 'High', class: 'risk-high' };
    };
    
    const volRisk = getRiskLevel(volatilityRisk);
    const marRisk = getRiskLevel(marketRisk);
    const liqRisk = getRiskLevel(liquidityRisk);
    
    const overallRisk = (volatilityRisk + marketRisk + liquidityRisk) / 3;
    const overallLevel = getRiskLevel(overallRisk);
    
    document.getElementById('riskResult').innerHTML = `
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Volatility Risk</span>
                <span style="font-weight: bold;">${volRisk.level}</span>
            </div>
            <div class="risk-meter">
                <div class="risk-bar">
                    <div class="risk-fill ${volRisk.class}" style="width: ${volatilityRisk}%;"></div>
                </div>
            </div>
        </div>
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Market Risk</span>
                <span style="font-weight: bold;">${marRisk.level}</span>
            </div>
            <div class="risk-meter">
                <div class="risk-bar">
                    <div class="risk-fill ${marRisk.class}" style="width: ${marketRisk}%;"></div>
                </div>
            </div>
        </div>
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Liquidity Risk</span>
                <span style="font-weight: bold;">${liqRisk.level}</span>
            </div>
            <div class="risk-meter">
                <div class="risk-bar">
                    <div class="risk-fill ${liqRisk.class}" style="width: ${liquidityRisk}%;"></div>
                </div>
            </div>
        </div>
        <div style="background: ${overallLevel.level === 'High' ? '#fff5f5' : overallLevel.level === 'Medium' ? '#fffaf0' : '#f0fff4'}; border: 1px solid ${overallLevel.level === 'High' ? '#feb2b2' : overallLevel.level === 'Medium' ? '#fbd38d' : '#9ae6b4'}; border-radius: 8px; padding: 12px; margin-top: 15px;">
            <div style="font-weight: bold; color: ${overallLevel.level === 'High' ? '#c53030' : overallLevel.level === 'Medium' ? '#d69e2e' : '#22543d'};">${overallLevel.level === 'High' ? '⚠️' : overallLevel.level === 'Medium' ? '⚡' : '✅'} ${overallLevel.level} Risk Investment</div>
            <div style="font-size: 0.9rem; color: ${overallLevel.level === 'High' ? '#742a2a' : overallLevel.level === 'Medium' ? '#975a16' : '#276749'}; margin-top: 5px;">${symbol} shows ${overallLevel.level.toLowerCase()} overall risk. ${overallLevel.level === 'High' ? 'Consider diversification and risk management strategies.' : overallLevel.level === 'Medium' ? 'Monitor closely and consider position sizing.' : 'Suitable for conservative portfolios.'}</div>
        </div>
    `;
}

function addToPortfolio() {
    const symbol = document.getElementById('portfolioSymbol').value.toUpperCase();
    if (!symbol) return;
    
    const portfolioList = document.getElementById('portfolioList');
    const gain = (Math.random() * 200 - 100).toFixed(2);
    const shares = Math.floor(Math.random() * 20 + 1);
    
    const newItem = document.createElement('div');
    newItem.className = 'portfolio-item';
    newItem.innerHTML = `
        <div>
            <div style="font-weight: bold;">${symbol}</div>
            <div style="font-size: 0.9rem; color: #718096;">${shares} shares</div>
        </div>
        <div class="portfolio-gain ${gain >= 0 ? 'gain-positive' : 'gain-negative'}">${gain >= 0 ? '+' : ''}$${gain}</div>
    `;
    
    portfolioList.appendChild(newItem);
    document.getElementById('portfolioSymbol').value = '';
}

function analyzeTechnical() {
    const symbol = document.getElementById('technicalSymbol').value.toUpperCase();
    
    // Simulate technical analysis
    const rsi = Math.floor(Math.random() * 100);
    const macd = (Math.random() * 10 - 5).toFixed(1);
    const ma50 = (Math.random() * 500 + 100).toFixed(0);
    const volume = Math.random() > 0.5 ? 'High' : 'Low';
    const volumeChange = (Math.random() * 50 - 25).toFixed(0);
    
    const getRSIStatus = (rsi) => {
        if (rsi < 30) return { status: 'Oversold', color: '#e53e3e', bg: '#fff5f5' };
        if (rsi > 70) return { status: 'Overbought', color: '#e53e3e', bg: '#fff5f5' };
        return { status: 'Neutral', color: '#38a169', bg: '#f0fff4' };
    };
    
    const rsiStatus = getRSIStatus(rsi);
    const macdStatus = macd > 0 ? { status: 'Bullish', color: '#38a169', bg: '#f0fff4' } : { status: 'Bearish', color: '#e53e3e', bg: '#fff5f5' };
    const overallSignal = (rsi > 30 && rsi < 70 && macd > 0) ? 'BUY' : (rsi > 70 || macd < -2) ? 'SELL' : 'HOLD';
    
    document.getElementById('technicalResult').innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
            <div style="text-align: center; padding: 12px; background: ${rsiStatus.bg}; border-radius: 8px;">
                <div style="font-weight: bold; color: ${rsiStatus.color};">RSI: ${rsi}</div>
                <div style="font-size: 0.8rem; color: #718096;">${rsiStatus.status}</div>
            </div>
            <div style="text-align: center; padding: 12px; background: ${macdStatus.bg}; border-radius: 8px;">
                <div style="font-weight: bold; color: ${macdStatus.color};">MACD: ${macd >= 0 ? '+' : ''}${macd}</div>
                <div style="font-size: 0.8rem; color: #718096;">${macdStatus.status}</div>
            </div>
            <div style="text-align: center; padding: 12px; background: #fffaf0; border-radius: 8px;">
                <div style="font-weight: bold; color: #ed8936;">MA(50): $${ma50}</div>
                <div style="font-size: 0.8rem; color: #718096;">Support</div>
            </div>
            <div style="text-align: center; padding: 12px; background: ${volume === 'High' ? '#f0fff4' : '#fffaf0'}; border-radius: 8px;">
                <div style="font-weight: bold; color: ${volume === 'High' ? '#38a169' : '#ed8936'};">Volume: ${volume}</div>
                <div style="font-size: 0.8rem; color: #718096;">${volumeChange >= 0 ? '+' : ''}${volumeChange}%</div>
            </div>
        </div>
        <div style="background: ${overallSignal === 'BUY' ? '#f0fff4' : overallSignal === 'SELL' ? '#fff5f5' : '#fffaf0'}; border: 1px solid ${overallSignal === 'BUY' ? '#9ae6b4' : overallSignal === 'SELL' ? '#feb2b2' : '#fbd38d'}; border-radius: 8px; padding: 12px;">
            <div style="font-weight: bold; color: ${overallSignal === 'BUY' ? '#22543d' : overallSignal === 'SELL' ? '#c53030' : '#d69e2e'};">${overallSignal === 'BUY' ? '📈' : overallSignal === 'SELL' ? '📉' : '⏸️'} Technical Signal: ${overallSignal}</div>
            <div style="font-size: 0.9rem; color: ${overallSignal === 'BUY' ? '#276749' : overallSignal === 'SELL' ? '#742a2a' : '#975a16'}; margin-top: 5px;">${overallSignal === 'BUY' ? 'Multiple indicators suggest upward momentum. Consider entry point.' : overallSignal === 'SELL' ? 'Technical indicators suggest downward pressure. Consider exit strategy.' : 'Mixed signals detected. Wait for clearer trend confirmation.'}</div>
        </div>
    `;
}

// Real-time chart updates
function updateChart() {
    const data = marketChart.data.datasets[0].data;
    const labels = marketChart.data.labels;
    const lastValue = data[data.length - 1];
    const newValue = lastValue + (Math.random() * 10 - 5);
    
    // Add new data point
    data.shift();
    data.push(newValue);
    
    // Update time labels
    labels.shift();
    const now = new Date();
    labels.push(now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
    
    marketChart.update('none');
}

// Add real-time status indicator
function addLiveIndicator() {
    const header = document.querySelector('.header p');
    header.innerHTML = 'Advanced Stock Market Analysis & Prediction Platform <span style="color: #48bb78;">● LIVE</span>';
}

// Update portfolio with real-time data
function updatePortfolioRealTime() {
    const portfolioItems = document.querySelectorAll('.portfolio-item');
    portfolioItems.forEach(item => {
        const symbolEl = item.querySelector('div div:first-child');
        if (symbolEl) {
            const symbol = symbolEl.textContent;
            if (realTimeData[symbol]) {
                const gainEl = item.querySelector('.portfolio-gain');
                const shares = parseInt(item.querySelector('div div:last-child').textContent);
                const currentGain = realTimeData[symbol].change * shares;
                gainEl.textContent = `${currentGain >= 0 ? '+' : ''}$${currentGain.toFixed(2)}`;
                gainEl.className = `portfolio-gain ${currentGain >= 0 ? 'gain-positive' : 'gain-negative'}`;
            }
        }
    });
}

// Initialize real-time features
function initializeRealTime() {
    addLiveIndicator();
    predictStock();
    
    // Update data every 3 seconds for demo purposes
    setInterval(() => {
        updateRealTimeData();
        updateChart();
        updatePortfolioRealTime();
        updateStockCards();
        
        // Flash live indicators
        const liveIndicators = document.querySelectorAll('[id^="live-"]');
        liveIndicators.forEach(indicator => {
            indicator.style.color = '#ed8936';
            setTimeout(() => {
                indicator.style.color = '#38a169';
            }, 200);
        });
    }, 3000);

    // Update news every 30 seconds
    setInterval(updateNews, 30000);
}

// Dynamic news updates
function updateNews() {
    const newsItems = [
        { title: "Fed Signals Potential Rate Cut", impact: "positive", desc: "Federal Reserve hints at monetary policy changes affecting tech stocks..." },
        { title: "Tech Earnings Season Begins", impact: "neutral", desc: "Major tech companies report Q4 earnings with mixed results..." },
        { title: "Oil Prices Surge on Supply Concerns", impact: "negative", desc: "Energy sector sees significant gains amid geopolitical tensions..." },
        { title: "AI Stocks Rally on New Breakthrough", impact: "positive", desc: "Artificial intelligence companies surge following major announcement..." },
        { title: "Banking Sector Under Pressure", impact: "negative", desc: "Regional banks face headwinds from regulatory changes..." },
        { title: "Crypto Market Shows Volatility", impact: "neutral", desc: "Digital assets experience mixed trading amid regulatory uncertainty..." }
    ];

    const newsContainer = document.querySelector('.card:nth-child(4)');
    const currentNews = newsItems.slice(0, 3).map(news => `
        <div class="news-item">
            <div class="news-title">${news.title}</div>
            <div style="font-size: 0.9rem; color: #718096;">${news.desc}</div>
            <span class="news-impact impact-${news.impact}">${news.impact.charAt(0).toUpperCase() + news.impact.slice(1)} Impact</span>
        </div>
    `).join('');

    const newsContent = newsContainer.innerHTML.replace(
        /<div class="news-item">[\s\S]*?<\/div>\s*<\/div>/g, 
        ''
    );
    newsContainer.innerHTML = newsContent + currentNews;
}

// Generate simplified stock cards
function generateStockCards() {
    const stocksGrid = document.getElementById('stocksGrid');
    stocksGrid.innerHTML = '';
    
    Object.keys(realTimeData).forEach(symbol => {
        const stock = realTimeData[symbol];
        const isPositive = stock.change >= 0;
        
        const stockCard = document.createElement('div');
        stockCard.className = 'stock-card';
        stockCard.onclick = () => showStockAnalysis(symbol);
        stockCard.innerHTML = `
            <div class="stock-symbol">${symbol}</div>
            <div class="stock-price-large">$${stock.price.toFixed(2)}</div>
            <div class="stock-change-simple ${isPositive ? 'positive' : 'negative'}">
                ${isPositive ? '+' : ''}${stock.change.toFixed(2)} (${isPositive ? '+' : ''}${stock.changePercent.toFixed(2)}%)
            </div>
            <div class="click-hint">Click for analysis</div>
        `;
        
        stocksGrid.appendChild(stockCard);
    });
}

// Analyze individual stock
function analyzeStock(symbol) {
    const stock = realTimeData[symbol];
    if (!stock) return;
    
    // Update the search input with the selected symbol
    document.getElementById('stockSymbol').value = symbol;
    
    // Trigger prediction
    predictStock();
    
    // Scroll to analysis section
    document.querySelector('.analysis-section').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

// Add to watchlist
function addToWatchlist(symbol) {
    const stock = realTimeData[symbol];
    if (!stock) return;
    
    // Add to portfolio (reusing existing functionality)
    addToPortfolio();
    
    // Show success message
    const button = event.target;
    const originalText = button.textContent;
    button.textContent = 'Added!';
    button.style.background = '#48bb78';
    
    setTimeout(() => {
        button.textContent = originalText;
        button.style.background = '';
    }, 2000);
}


function showStockAnalysis(symbol) {
    const stock = realTimeData[symbol];
    if (!stock) return;
    
    const modal = document.getElementById('analysisModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalAnalysis = document.getElementById('modalAnalysis');
    
    
    modalTitle.textContent = `${symbol} - ${stock.company}`;
    
   
    const isPositive = stock.change >= 0;
    const predictedPrice = (stock.price * (1 + (Math.random() * 0.2 - 0.1))).toFixed(2);
    const confidence = (Math.random() * 30 + 60).toFixed(0);
    const rsi = Math.floor(Math.random() * 100);
    const macd = (Math.random() * 10 - 5).toFixed(1);
    
   
    modalAnalysis.innerHTML = `
        <div class="analysis-item">
            <h4>Current Price</h4>
            <div class="value">$${stock.price.toFixed(2)}</div>
            <div class="change ${isPositive ? 'positive' : 'negative'}">
                ${isPositive ? '+' : ''}${stock.change.toFixed(2)} (${isPositive ? '+' : ''}${stock.changePercent.toFixed(2)}%)
            </div>
        </div>
        <div class="analysis-item">
            <h4>Predicted Price</h4>
            <div class="value">$${predictedPrice}</div>
            <div class="change ${predictedPrice > stock.price ? 'positive' : 'negative'}">
                ${predictedPrice > stock.price ? '+' : ''}${((predictedPrice - stock.price) / stock.price * 100).toFixed(2)}%
            </div>
        </div>
        <div class="analysis-item">
            <h4>Confidence</h4>
            <div class="value">${confidence}%</div>
            <div class="change positive">High</div>
        </div>
        <div class="analysis-item">
            <h4>RSI</h4>
            <div class="value">${rsi}</div>
            <div class="change ${rsi > 70 ? 'negative' : rsi < 30 ? 'positive' : 'neutral'}">
                ${rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral'}
            </div>
        </div>
        <div class="analysis-item">
            <h4>MACD</h4>
            <div class="value">${macd >= 0 ? '+' : ''}${macd}</div>
            <div class="change ${macd > 0 ? 'positive' : 'negative'}">
                ${macd > 0 ? 'Bullish' : 'Bearish'}
            </div>
        </div>
        <div class="analysis-item">
            <h4>Market Cap</h4>
            <div class="value">$${stock.marketCap}</div>
            <div class="change neutral">${stock.sector}</div>
        </div>
    `;
    
    // Create stock chart
    createStockChart(symbol);
    
    // Show modal
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

// Create stock chart for modal
function createStockChart(symbol) {
    const ctx = document.getElementById('stockChart').getContext('2d');
    
    // Generate sample data for the stock
    const data = [];
    const labels = [];
    const basePrice = realTimeData[symbol].price;
    
    for (let i = 0; i < 30; i++) {
        const time = new Date();
        time.setDate(time.getDate() - (30 - i));
        labels.push(time.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        data.push(basePrice + (Math.random() - 0.5) * basePrice * 0.1);
    }
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `${symbol} Price`,
                data: data,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: '#667eea',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                }
            }
        }
    });
}

// Close analysis modal
function closeAnalysisModal() {
    const modal = document.getElementById('analysisModal');
    modal.classList.remove('active');
    document.body.style.overflow = 'auto';
}

// Close modal when clicking outside
document.addEventListener('click', function(e) {
    const modal = document.getElementById('analysisModal');
    if (e.target === modal) {
        closeAnalysisModal();
    }
});

// Update stock cards with real-time data
function updateStockCards() {
    const stockCards = document.querySelectorAll('.stock-card');
    stockCards.forEach(card => {
        const symbol = card.querySelector('.stock-symbol').textContent;
        const stock = realTimeData[symbol];
        if (stock) {
            const isPositive = stock.change >= 0;
            
            // Update price
            const priceEl = card.querySelector('.stock-price-large');
            priceEl.textContent = `$${stock.price.toFixed(2)}`;
            
            // Update change
            const changeEl = card.querySelector('.stock-change-simple');
            changeEl.textContent = `${isPositive ? '+' : ''}${stock.change.toFixed(2)} (${isPositive ? '+' : ''}${stock.changePercent.toFixed(2)}%)`;
            changeEl.className = `stock-change-simple ${isPositive ? 'positive' : 'negative'}`;
        }
    });
}

// Initialize everything when page loads
document.addEventListener('DOMContentLoaded', function() {
    generateStockCards();
    initializeRealTime();
});
