# Hand&Brain - AI Stock Assistant

<div align="center">
  <img src="https://github.com/user-attachments/assets/a42a1162-0ab8-41a5-95b0-a1df348c6353" alt="Hand&Brain Logo" width="200"/>
  <h3>Intelligent Stock Analysis with Interactive Visualization</h3>
  
  *Advanced technical analysis and AI-powered stock predictions*
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
  [![GitHub stars](https://img.shields.io/github/stars/DivyanshuSaini2112/stock-analysis-tool?style=social)](https://github.com/DivyanshuSaini2112/stock-analysis-tool/stargazers)
</div>

## 📋 Overview

Hand&Brain is a comprehensive Python application that provides in-depth technical analysis and visualization of stock market data. Built on advanced AI algorithms, this tool combines sophisticated data processing with intuitive visualizations to help investors make informed decisions. The system integrates traditional technical indicators with modern machine learning techniques to provide a holistic view of market opportunities.

## ✨ Key Features

<table>
  <tr>
    <td>
      <ul>
        <li>📊 <b>Comprehensive Technical Analysis</b>: RSI, MACD, Bollinger Bands, and moving averages</li>
        <li>🔮 <b>AI-Powered Predictions</b>: Price forecasting using advanced machine learning models</li>
        <li>🌍 <b>Multi-Exchange Support</b>: Compatible with US stocks, NSE (India), and BSE (India)</li>
        <li>📈 <b>Pattern Recognition</b>: Automatic identification of chart patterns and trading signals</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>📱 <b>Interactive Dashboards</b>: Beautiful, information-rich visualizations with multiple panels</li>
        <li>⚖️ <b>Valuation Metrics</b>: Compare company P/E ratios with industry averages</li>
        <li>🔍 <b>Sentiment Analysis</b>: News and social media sentiment integration</li>
        <li>📂 <b>Portfolio Management</b>: Track and analyze multiple stocks simultaneously</li>
      </ul>
    </td>
  </tr>
</table>

## 📊 Analysis Dashboard

The system includes a powerful analytics dashboard providing:

- 📈 Price action with multiple timeframes
- 📊 Technical indicators visualization
- 💹 Performance metrics comparison
- 🎯 Buy/Sell signal generation
- 📑 Fundamental data integration

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
tensorflow>=2.6.0 (optional, for AI predictions)
plotly>=5.3.0 (optional, for interactive charts)
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
4. Generate a comprehensive visualization dashboard
5. Save the analysis as a PNG file

### Advanced Options

For advanced options:
```bash
python stock_analysis.py --ticker AAPL --days 180 --indicators all --output report.png
```

Available parameters:
- `--ticker`: Stock symbol (e.g., AAPL, RELIANCE.NS)
- `--days`: Number of days for historical data (default: 180)
- `--indicators`: Indicators to calculate (options: all, basic, custom)
- `--output`: Output filename for the generated report

## 🏗️ Technical Architecture

<div align="center">
  <img src="https://github.com/user-attachments/assets/a6ac1389-15b9-432b-8423-b29e92fb396d" alt="System Architecture" width="700">
</div>

The Hand&Brain system consists of several key components:

### Data Retrieval
- `MarketDataFetcher`: Retrieves historical market data with error handling
- `SymbolResolver`: Handles ticker resolution across multiple exchanges
- `DataPipeline`: Cleans and preprocesses raw market data

### Technical Analysis
- `IndicatorFactory`: Calculates various technical indicators
- `SignalGenerator`: Identifies buy/sell signals and patterns
- `SectorAnalyzer`: Performs sector-specific analysis and comparisons

### Machine Learning
- `PredictionEngine`: Implements time series forecasting models
- `SentimentAnalyzer`: Processes news and social media sentiment
- `PatternRecognizer`: Identifies chart patterns using computer vision

### Visualization
- `DashboardBuilder`: Creates multi-panel visualization dashboards
- `ChartRenderer`: Renders individual chart components
- `ReportGenerator`: Produces comprehensive analysis reports

## 📈 Sample Output

The tool generates comprehensive dashboards like this:

![DLF Limited Analysis](https://github.com/user-attachments/assets/a6ac1389-15b9-432b-8423-b29e92fb396d)

The dashboard includes:
- Price chart with moving averages and Bollinger Bands
- Volume analysis with color-coded bars
- RSI and MACD indicators
- Price, volume, valuation, and performance metrics
- Technical signals summary

## 📚 Documentation

For more detailed information, please refer to the following documentation:

- [User Guide](docs/UserGuide.md)
- [Technical Indicators Reference](docs/TechnicalIndicators.md)
- [AI Prediction Models](docs/AIPredictionModels.md)
- [API Documentation](docs/APIDocumentation.md)
- [Backtesting Framework](docs/BacktestingFramework.md)

## 📂 Project Structure

```
stock-analysis-tool/
├── stock_analysis.py            # Main application script
├── data/
│   ├── fetcher.py               # Data retrieval module
│   ├── processor.py             # Data preprocessing module
│   └── symbol_resolver.py       # Ticker symbol resolution
├── analysis/
│   ├── indicators.py            # Technical indicators
│   ├── signals.py               # Trading signals
│   └── patterns.py              # Chart pattern recognition
├── ml/
│   ├── prediction.py            # Price prediction models
│   ├── sentiment.py             # Sentiment analysis
│   └── models/                  # Pre-trained ML models
├── visualization/
│   ├── dashboard.py             # Dashboard generation
│   ├── charts.py                # Chart components
│   └── styling.py               # Visual styling
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
- [ ] Backtesting framework for strategy validation
- [ ] Portfolio analysis and optimization
- [ ] Integration with real-time data feeds
- [ ] Machine learning-based pattern recognition
- [ ] Mobile application development
- [ ] API for third-party integration
- [ ] Cloud-based deployment option

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
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for informational purposes only. The technical analysis and visualizations provided should not be considered financial advice. Always conduct your own research before making investment decisions.

## 🙏 Acknowledgments

- Yahoo Finance for providing market data
- TA-Lib community for technical analysis algorithms
- TensorFlow and scikit-learn teams for machine learning frameworks
- Matplotlib and Plotly for visualization capabilities

---

<div align="center">
  <p>Developed by <a href="https://github.com/DivyanshuSaini2112">Divyanshu Saini</a> and contributors</p>
  <p>© 2023-2025 Hand&Brain. All rights reserved.</p>
</div>
