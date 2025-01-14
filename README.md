# 🤖 Vision Transformer Web App

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful web application that leverages Vision Transformer (ViT) for state-of-the-art image classification, featuring a modern UI and comprehensive monitoring dashboard.

<p align="center">
  <img src="static/assets/demo.gif" alt="Demo" width="600">
</p>

## ✨ Features

### 🖼️ Image Processing
- 📤 Drag-and-drop or click-to-upload interface
- ⚡ Real-time image classification using Vision Transformer
- 📊 Top classification results with confidence scores
- ℹ️ Detailed model and image metadata display

### 🎯 Batch Processing
- 📁 Process multiple images simultaneously
- 📈 Batch progress tracking
- 📋 Detailed batch results summary
- 💾 Export results to various formats

### 📊 Monitoring Dashboard
- 💻 System Resources:
  - CPU usage and frequency
  - Memory utilization
  - Disk metrics
  - GPU monitoring
- 📈 Performance Metrics:
  - Real-time inference tracking
  - Throughput analysis
  - Error rate monitoring
- 📊 Interactive Charts:
  - Inference time distribution
  - Error analysis
  - Historical trends

### 🛠️ Additional Features
- 📱 Responsive design with Tailwind CSS
- 📄 Automated report generation
- ⏰ Task scheduling
- 🔄 Real-time updates
- 🐛 Comprehensive error handling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fahad0samara/Vision-Transformer-Web-App.git
cd Vision-Transformer-Web-App
```

2. Set up virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
```

5. Create directories:
```bash
mkdir uploads models reports
```

## 🎮 Usage

1. Start the server:
```bash
python app.py
```

2. Access the application:
- 🌐 Main App: `http://localhost:5000`
- 📦 Batch Processing: `http://localhost:5000/batch`
- 📊 Dashboard: `http://localhost:5000/dashboard`

## 🏗️ Project Structure

```
📦 Vision-Transformer-Web-App
 ┣ 📂 static                  # Frontend assets
 ┃ ┣ 📂 assets               # Images and icons
 ┃ ┣ 📜 index.html          # Main page
 ┃ ┣ 📜 batch.html          # Batch processing
 ┃ ┣ 📜 dashboard.html      # Monitoring dashboard
 ┃ ┣ 📜 styles.css          # Custom styles
 ┃ ┗ 📜 script.js           # Frontend logic
 ┣ 📂 utils                  # Backend utilities
 ┃ ┣ 📜 batch_processor.py  # Batch processing
 ┃ ┣ 📜 monitoring.py       # System monitoring
 ┃ ┗ 📜 image_utils.py      # Image processing
 ┣ 📂 uploads               # Uploaded files
 ┣ 📂 reports               # Generated reports
 ┣ 📜 app.py               # Main application
 ┣ 📜 config.py            # Configuration
 ┣ 📜 requirements.txt     # Dependencies
 ┗ 📜 README.md            # Documentation
```

## 🔧 Tech Stack

- **Backend**:
  - 🐍 Python 3.8+
  - 🌶️ Flask
  - 🔥 PyTorch
  - 🤗 Transformers
  - 📊 Prometheus Client
  - 💻 psutil

- **Frontend**:
  - 🎨 Tailwind CSS
  - 📊 Chart.js
  - 🔄 Fetch API
  - 📱 Responsive Design

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add some amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🎁 Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Vision Transformer implementation based on [Google's ViT paper](https://arxiv.org/abs/2010.11929)
- UI design inspired by modern web applications
- Thanks to all contributors who helped improve this project

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/fahad0samara">Fahad</a>
</p>
