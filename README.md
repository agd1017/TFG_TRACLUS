# TFG TRACLUS: Web Application for Trajectory Clustering

This repository contains a web application developed using Python, Dash, and related technologies for trajectory clustering and visualization. The project focuses on the analysis and clustering of spatial trajectory data, supporting various algorithms and datasets.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Structure

```plaintext
TFG_TRACLUS  
├── code/  
│   ├── app/  
│   │   ├── controllers/  
│   │   ├── models/  
│   │   ├── utils/  
│   │   ├── views/  
│   │   │   ├── layout/  
│   │   │   ├── assets/  
│   │   ├── test/  
│   │   ├── source/  
│   │   ├── config/  
│   │   ├── main.py  
│   │   └── requirements.txt  
│   └── Research and experiments/  
├── docs/  
└── README.md 
``` 

- `app/`: Contains the core application structure following MVC principles.
- `controllers/`, `models/`, `views/`: Separate logic for handling data, business logic, and user interface components.
- `assets/`: Contains CSS files and static resources.
- `source/`: Contains the sphinx code to see the code documentation in html.
- `Research and experiments/`: Contains the code of the experiments realizados a lo largo del proyecto.
- `docs/`: Contains the documentation of the project.

## Features

| Feature                           | Description                                                                               |
|-----------------------------------|-------------------------------------------------------------------------------------------|
| Trajectory Clustering Algorithms  | Implements various clustering algorithms to analyze trajectory data.                      |
| Visualization of Clustered Data   | Provides visualizations of clustered trajectories using geographic coordinates.           |
| Configurable Parameters           | Offers flexibility to customize parameters per dataset for better clustering performance. |
| Modular and Scalable Architecture | Designed with a modular structure, facilitating future extensions and scalability.        |
| Cloud Deployment Ready            | Compatible with cloud platforms for deployment.                                           |

## Algorithms

- **OPTICS**
- **DBSCAN**
- **HDBSCAN**
- **Spectral Clustering**
- **Agglomerative Clustering**

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/TFG_TRACLUS.git
    cd TFG_TRACLUS/code

2. Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Usage

1. Run the application:
    ```bash
    python app/main.py

    Access the web interface at http://127.0.0.1:8080 in your browser.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
