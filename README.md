# Hand&Brain - AI Stock Assistant

<div align="center">
  <img src="https://github.com/user-attachments/assets/a42a1162-0ab8-41a5-95b0-a1df348c6353" alt="Hand&Brain Logo" width="200"/>
  <h3>Intelligent Stock Analysis with Interactive Visualization</h3>
</div>

## Overview

Stock Analysis Tool is a comprehensive Python application that provides in-depth technical analysis and visualization of stock market data. Built on the Hand&Brain AI framework, this tool combines advanced data processing capabilities with intuitive visualizations to help investors make informed decisions.



## Key Features

- **Comprehensive Technical Analysis**: Calculate RSI, MACD, Bollinger Bands, and moving averages
- **Interactive Visualization**: Beautiful, information-rich dashboards with multiple panels
- **Support for Multiple Markets**: Compatible with US stocks, NSE (India), and BSE (India)
- **Technical Signal Generation**: Automatic identification of bullish and bearish patterns
- **Performance Metrics**: Track daily, weekly, and monthly price changes
- **Valuation Metrics**: Compare company P/E ratios with industry averages

## Technical Architecture

The tool is built using a modular Python architecture:

### Data Retrieval
- Utilizes `yfinance` to fetch historical market data
- Automatic market detection (NSE/BSE/US)
- Fallback mechanisms for ticker symbol variations

### Technical Indicators
- Moving averages (5, 20, 50, 200 days)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Average True Range (ATR)
- Support and resistance levels

### Visualization
- Built on Matplotlib with custom styling
- Modern, clean design with responsive layout
- Color-coded signals and metrics
- Multi-panel dashboard design

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: pandas, numpy, matplotlib, yfinance, ta

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DivyanshuSaini2112/stock-analysis-tool.git
   cd stock-analysis-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the main script:
```bash
python stock_analysis.py
```

Enter any stock ticker symbol when prompted. The tool will:
1. Automatically detect the appropriate exchange (NSE/BSE/US)
2. Fetch 6 months of historical data
3. Calculate technical indicators
4. Generate a comprehensive visualization dashboard
5. Save the analysis as a PNG file

## Core Functions

### Data Processing

- `fetch_stock_data()`: Retrieves historical price data with error handling
- `create_technical_features()`: Calculates all technical indicators
- `fetch_sector_pe()`: Estimates industry P/E ratios for comparison

### Visualization

- `visualize_stock_data()`: Creates the main dashboard with multiple panels
- `create_fancy_table()`: Formats metrics in clean, readable tables
- `create_signal_panel()`: Generates technical signal summaries
- `add_panel_styling()`: Applies consistent styling across chart components

## Integration with Hand&Brain AI

This tool is designed to integrate with the Hand&Brain AI Stock Assistant, which provides:

- Conditional GAN (cGAN) models for market simulation
- Reinforcement Learning for trading strategy optimization
- Personalized stock recommendations based on user preferences

## Sample Output

The tool generates comprehensive dashboards like this:

![DLF Limited Analysis](![Screenshot 2025-03-19 225648](https://github.com/user-attachments/assets/a6ac1389-15b9-432b-8423-b29e92fb396d)
)

The dashboard includes:
- Price chart with moving averages and Bollinger Bands
- Volume analysis with color-coded bars
- RSI and MACD indicators
- Price, volume, valuation, and performance metrics
- Technical signals summary

## Development Roadmap

- [x] Core technical analysis implementation
- [x] Interactive visualization dashboard
- [x] Multi-exchange support
- [ ] Backtesting framework for strategy validation
- [ ] Portfolio analysis and optimization
- [ ] Integration with real-time data feeds
- [ ] Machine learning-based pattern recognition
- [ ] Mobile application development

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for informational purposes only. The technical analysis and visualizations provided should not be considered financial advice. Always conduct your own research before making investment decisions.

---

<div align="center">
  <p>© 2025 Hand&Brain. All rights reserved.</p>
</div>
