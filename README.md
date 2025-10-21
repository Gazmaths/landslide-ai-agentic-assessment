# Landslide Susceptibility Agent

## Overview
An AI-powered application that provides landslide susceptibility analysis based on location data. The system uses machine learning-derived probability data combined with OpenAI's GPT-4 to provide comprehensive risk assessments and recommendations.

## Features
- **Address-based Analysis**: Input any address to get landslide risk assessment
- **Coordinate-based Analysis**: Direct latitude/longitude input
- **Interactive Mapping**: Visual representation of landslide probabilities
- **AI-powered Insights**: GPT-4 generated analysis and recommendations
- **Risk Categorization**: Clear risk levels from Very Low to Very High
- **Data Visualization**: Statistical overview of the dataset

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run landslide_agent.py
```

## Usage

1. **Configure API Key**: Enter your OpenAI API key in the sidebar
2. **Choose Input Method**: Select between address search or coordinate input
3. **Enter Location**: Provide the location you want to analyze
4. **Analyze Risk**: Click the analysis button to get results
5. **Review Results**: View probability, risk level, and AI-generated insights

## Data
The application uses landslide probability data covering approximately 280 km² with over 580,000 data points. The data appears to cover a region in North Carolina with coordinates ranging from:
- Longitude: -82.41° to -82.12°
- Latitude: 35.20° to 35.48°

## Risk Categories
- **Very Low**: 0.0 - 0.2 probability
- **Low**: 0.2 - 0.4 probability  
- **Moderate**: 0.4 - 0.6 probability
- **High**: 0.6 - 0.8 probability
- **Very High**: 0.8 - 1.0 probability

## API Requirements
- OpenAI API key for GPT-4 access
- Internet connection for geocoding services

## Future Enhancements
- Integration with real-time weather data
- Historical landslide event correlation
- Temporal risk modeling
- Mobile-friendly interface
- PDF report generation
- Multi-language support
# landslide-ai-agentic-assessment
