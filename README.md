# Hand&Brain - AI Stock Assistant

<div align="center">
  <img src="https://github.com/user-attachments/assets/a42a1162-0ab8-41a5-95b0-a1df348c6353" alt="Hand&Brain Logo" width="200"/>
  <h3>Intelligent Stock Analysis with AI-Powered Prediction</h3>
  
  *Advanced technical analysis, machine learning models, and interactive visualization*
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
  [![GitHub stars](https://img.shields.io/github/stars/DivyanshuSaini2112/stock-analysis-tool?style=social)](https://github.com/DivyanshuSaini2112/stock-analysis-tool/stargazers)
</div>

## 📋 Overview

Hand&Brain is a comprehensive Python application that combines traditional technical analysis with cutting-edge AI models to provide in-depth stock market analysis and predictions. Built on advanced machine learning algorithms including Conditional GANs, LSTMs, and Transformer models, this tool delivers sophisticated market insights with intuitive visualizations to help investors make informed decisions. The system integrates technical indicators, fundamental metrics, and sentiment analysis to provide a holistic view of market opportunities tailored to individual risk profiles.

## ✨ Key Features

<table>
  <tr>
    <td>
      <ul>
        <li>📊 <b>Comprehensive Technical Analysis</b>: RSI, MACD, Bollinger Bands, and moving averages</li>
        <li>🔮 <b>AI-Powered Predictions</b>: Price forecasting using advanced ML models (cGANs, LSTMs, Transformers)</li>
        <li>🤖 <b>Reinforcement Learning</b>: Optimized buy/sell decision strategies through simulated environments</li>
        <li>🌍 <b>Multi-Exchange Support</b>: Compatible with US stocks, NSE (India), and BSE (India)</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>📈 <b>Pattern Recognition</b>: Automatic identification of chart patterns and trading signals</li>
        <li>📰 <b>Sentiment Analysis</b>: News and social media sentiment integration</li>
        <li>📱 <b>Interactive Dashboards</b>: Beautiful, information-rich visualizations with multiple panels</li>
        <li>⚖️ <b>Risk-Adjusted Recommendations</b>: Personalized insights based on user risk tolerance</li>
      </ul>
    </td>
  </tr>
</table>

## 🧠 AI Prediction Models

Hand&Brain employs multiple advanced AI approaches for stock prediction:

- **Conditional GANs (cGANs)**: Generate realistic market scenarios for robust model training
- **LSTM Networks**: Capture temporal patterns in time-series stock data
- **Transformer Models**: Identify long-term dependencies using attention mechanisms
- **Reinforcement Learning**: Optimize trading strategies in simulated environments
- **Ensemble Methods**: Combine predictions from multiple models for improved accuracy

Models are evaluated using rigorous metrics:
- Regression: RMSE, MAE, R-squared
- Classification: Accuracy, Precision, Recall, F1-score

## 📊 Analysis Dashboard

The system includes a powerful analytics dashboard providing:

- 📈 Price action with multiple timeframes
- 📊 Technical indicators visualization
- 🔄 AI-generated price predictions
- 💹 Performance metrics comparison
- 🎯 Buy/Sell signal generation
- 📑 Fundamental data integration
- 📰 News sentiment analysis

## 🔍 Data Sources & Processing

Hand&Brain integrates data from multiple sources:

- **Market Data**: Historical prices from Yahoo Finance, Alpha Vantage, and Quandl
- **Sentiment Data**: News articles (Financial Times, NewsAPI) and social media
- **Fundamental Metrics**: Company financials, earnings reports, and sector data

Data processing pipeline includes:
- Automated data cleaning and normalization
- Feature engineering for technical indicators
- NLP processing for sentiment extraction
- Time-series data preparation for AI models

## 🖥️ System Requirements

- Python 3.8+
- Required Python packages:

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
yfinance>=0.1.70
ta>=0.7.0
scikit-learn>=0.24.0
tensorflow>=2.6.0
pytorch>=1.9.0
transformers>=4.5.0
plotly>=5.3.0
nltk>=3.6.0
newsapi-python>=0.2.6
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DivyanshuSaini2112/stock-analysis-tool.git
   cd stock-analysis-tool
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install optional packages for enhanced functionality:**
   ```bash
   pip install -r requirements-advanced.txt
   ```

4. **Download pre-trained models:**
   ```bash
   python download_models.py
   ```

## 🎮 Usage

### Basic Analysis

Run the main script:
```bash
python stock_analysis.py
```

Enter any stock ticker symbol when prompted. The tool will:
1. Automatically detect the appropriate exchange (NSE/BSE/US)
2. Fetch historical data for the selected timeframe
3. Calculate technical indicators
4. Generate AI predictions for future price movements
5. Create a comprehensive visualization dashboard
6. Save the analysis as a PNG file

### Advanced Options

For advanced options:
```bash
python stock_analysis.py --ticker AAPL --days 180 --indicators all --model ensemble --risk-level moderate --output report.png
```

Available parameters:
- `--ticker`: Stock symbol (e.g., AAPL, RELIANCE.NS)
- `--days`: Number of days for historical data (default: 180)
- `--indicators`: Indicators to calculate (options: all, basic, custom)
- `--model`: AI model to use (options: lstm, transformer, cgan, ensemble)
- `--risk-level`: User risk tolerance (options: conservative, moderate, aggressive)
- `--output`: Output filename for the generated report

## 🏗️ Technical Architecture

<div align="center">
  <img src="https://github.com/user-attachments/assets/a6ac1389-15b9-432b-8423-b29e92fb396d" alt="System Architecture" width="700">
</div>

The Hand&Brain system consists of several key components:

### Data Retrieval & Processing
- `MarketDataFetcher`: Retrieves historical market data with error handling
- `SymbolResolver`: Handles ticker resolution across multiple exchanges
- `SentimentCollector`: Gathers and processes news and social media data
- `DataPipeline`: Cleans and preprocesses raw market data

### Technical Analysis
- `IndicatorFactory`: Calculates various technical indicators
- `SignalGenerator`: Identifies buy/sell signals and patterns
- `SectorAnalyzer`: Performs sector-specific analysis and comparisons

### Machine Learning
- `PredictionEngine`: Implements multiple time series forecasting models
- `SentimentAnalyzer`: Processes news and social media sentiment using NLP
- `PatternRecognizer`: Identifies chart patterns using computer vision
- `RiskProfiler`: Adjusts predictions based on user risk tolerance

### Visualization
- `DashboardBuilder`: Creates multi-panel visualization dashboards
- `ChartRenderer`: Renders individual chart components
- `ReportGenerator`: Produces comprehensive analysis reports

## 📈 Sample Output

The tool generates comprehensive dashboards like this:

![Sample Dashboard](https://github.com/user-attachments/assets/a6ac1389-15b9-432b-8423-b29e92fb396d)

The dashboard includes:
- Price chart with moving averages and AI prediction bands
- Volume analysis with color-coded bars
- RSI and MACD indicators
- Price, volume, valuation, and performance metrics
- Technical signals summary
- Sentiment analysis results
- Risk-adjusted recommendations

## 🧪 Backtesting Framework

Hand&Brain includes a robust backtesting system to validate investment strategies:

- Test strategies against historical data
- Measure performance metrics (returns, drawdowns, Sharpe ratio)
- Compare against benchmark indices
- Simulate different market conditions
- Optimize parameters for maximum returns

## 📚 Documentation

For more detailed information, please refer to the following documentation:

- [User Guide](docs/UserGuide.md)
- [Technical Indicators Reference](docs/TechnicalIndicators.md)
- [AI Prediction Models](docs/AIPredictionModels.md)
- [Sentiment Analysis Framework](docs/SentimentAnalysis.md)
- [Backtesting System](docs/BacktestingFramework.md)
- [API Documentation](docs/APIDocumentation.md)

## 📂 Project Structure

```
stock-analysis-tool/
├── stock_analysis.py            # Main application script
├── data/
│   ├── fetcher.py               # Data retrieval module
│   ├── processor.py             # Data preprocessing module
│   ├── sentiment_collector.py   # News and social media data
│   └── symbol_resolver.py       # Ticker symbol resolution
├── analysis/
│   ├── indicators.py            # Technical indicators
│   ├── signals.py               # Trading signals
│   ├── patterns.py              # Chart pattern recognition
│   └── fundamentals.py          # Fundamental analysis
├── ml/
│   ├── prediction.py            # Price prediction models
│   ├── sentiment.py             # Sentiment analysis
│   ├── reinforcement.py         # Reinforcement learning agents
│   ├── transformer.py           # Transformer models
│   ├── gan.py                   # Conditional GAN implementation
│   └── models/                  # Pre-trained ML models
├── visualization/
│   ├── dashboard.py             # Dashboard generation
│   ├── charts.py                # Chart components
│   └── styling.py               # Visual styling
├── backtesting/
│   ├── engine.py                # Backtesting framework
│   ├── metrics.py               # Performance metrics
│   └── strategies.py            # Trading strategies
├── utils/
│   ├── config.py                # Configuration
│   └── helpers.py               # Helper functions
├── tests/                       # Test cases
├── docs/                        # Documentation
└── README.md                    # This file
```

## 🔬 Development Roadmap

- [x] Core technical analysis implementation
- [x] Interactive visualization dashboard
- [x] Multi-exchange support
- [x] LSTM and Transformer model implementation
- [x] Sentiment analysis integration
- [ ] Conditional GAN model implementation
- [ ] Reinforcement learning trading agent
- [ ] Portfolio optimization module
- [ ] Real-time data feed integration
- [ ] Mobile application development
- [ ] Cloud-based deployment
- [ ] API for third-party integration

## 🤝 Contributing

Contributions to the Hand&Brain project are welcome! Please feel free to submit a Pull Request.

Guidelines for contributing:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔍 Testing

Run the test suite:
```bash
python -m unittest discover tests
```

For specific test categories:
```bash
python -m unittest tests/test_indicators.py
python -m unittest tests/test_ml_models.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for informational purposes only. The technical analysis, AI predictions, and visualizations provided should not be considered financial advice. Always conduct your own research before making investment decisions. Past performance is not indicative of future results.

## 🙏 Acknowledgments

- Yahoo Finance, Alpha Vantage, and Quandl for market data
- TA-Lib community for technical analysis algorithms
- TensorFlow, PyTorch, and Hugging Face teams for ML frameworks
- NewsAPI and Financial Times for news data access
- Matplotlib and Plotly for visualization capabilities

---

<div align="center">
  <p>Developed by <a href="https://github.com/DivyanshuSaini2112">Divyanshu Saini</a> and contributors</p>
  <p>© 2023-2025 Hand&Brain. All rights reserved.</p>
</div>
