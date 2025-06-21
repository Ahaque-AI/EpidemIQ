# Kermit

Kermit is an AI-powered project that integrates advanced data analysis, simulation, and web application components to provide insightful reports and interactive features. It leverages machine learning and graph-based retrieval augmented generation (RAG) techniques to analyze datasets and generate detailed simulation reports on various health-related scenarios.

## Table of Contents

- [About](#about)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Reports](#reports)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

## About

Kermit is designed to process complex datasets related to health conditions such as COVID-19, food poisoning, gastroenteritis, and pneumonia. It performs drift analysis, generates simulation reports, and supports interactive querying through a web interface. The project combines Jupyter notebooks, Python scripts, and a web app to deliver a comprehensive AI agent capable of data-driven decision support.

## Features

- AI Agent for data analysis and simulation  
- Graph-based Retrieval Augmented Generation (graphRAG) for enhanced data querying  
- Interactive web application for user engagement  
- Detailed simulation reports for multiple health scenarios  
- Dataset management and preprocessing tools  
- Drift analysis to monitor data changes over time  

## Installation

To set up Kermit locally, follow these steps:

1. Clone the repository:

```

git clone https://github.com/Ahaque-AI/kermit.git
cd kermit

```

2. (Optional) Create and activate a virtual environment:

```

python -m venv venv
source venv/bin/activate  \# On Windows: venv\Scripts\activate

```

3. Explore datasets in the `dataset/` folder and simulation reports in the root directory.

## Usage

- Run AI agent scripts located in the `AI_Agent/` folder for data processing and analysis.  
- Use the `graphRAG/` module to perform advanced graph-based retrieval queries.  
- Launch the web application found in the `web_app/` directory to interact with the system via a browser interface.  
- Review generated reports such as `covid19_simulation_report_*.md` for insights on specific health simulations.

## Project Structure

- `AI_Agent/` - Core AI and machine learning modules  
- `graphRAG/` - Graph-based retrieval augmented generation implementation  
- `dataset/` - Data files and preprocessing scripts  
- `web_app/` - Web application source code  
- `output/` - Generated outputs and reports  
- Various simulation report markdown files for health scenarios  
- Utility scripts like `temp.py` and `try.py` for experimentation  

## Reports

The repository includes detailed simulation reports analyzing different health conditions:

- COVID-19 Simulation Reports  
- Food Poisoning Simulation Report  
- Gastroenteritis Simulation Report  
- Pneumonia Simulation Report  
- Drift Analysis Report  

These reports provide data-driven insights and are useful for researchers and practitioners interested in epidemiological modeling.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements or bug fixes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project currently does not specify a license. Please contact the maintainer for licensing information.

## Contact

For questions or support, please reach out to the repository owner.

