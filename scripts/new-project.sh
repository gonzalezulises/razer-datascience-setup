#!/bin/bash
# Create a new data science project
# Usage: ./new-project.sh project_name [template]
# Templates: minimal, standard, full

PROJECT_NAME=${1:-"my-project"}
TEMPLATE=${2:-"standard"}

echo "=== Creating project: $PROJECT_NAME ==="

cd ~/projects
uv init "$PROJECT_NAME"
cd "$PROJECT_NAME"

case $TEMPLATE in
    minimal)
        uv add numpy pandas scikit-learn matplotlib jupyter
        ;;
    standard)
        uv add numpy pandas scipy scikit-learn matplotlib seaborn jupyter jupyterlab ydata-profiling
        ;;
    full)
        uv add numpy pandas scipy scikit-learn matplotlib seaborn jupyter jupyterlab
        uv add ydata-profiling sweetviz
        uv add xgboost lightgbm catboost
        uv add mlflow optuna
        uv add fastapi uvicorn streamlit
        uv add shap lime
        uv add sqlalchemy psycopg2-binary redis
        uv add beautifulsoup4 httpx lxml
        ;;
    gpu)
        uv add numpy pandas scipy scikit-learn matplotlib seaborn jupyter jupyterlab
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        uv pip install tensorflow
        ;;
    *)
        echo "Unknown template: $TEMPLATE"
        echo "Available: minimal, standard, full, gpu"
        exit 1
        ;;
esac

# Create directory structure
mkdir -p data/{raw,processed,external}
mkdir -p notebooks
mkdir -p src/{data,features,models,visualization}
mkdir -p reports/figures
mkdir -p tests

# Create .gitignore
cat > .gitignore << "EOF"
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
.env

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
data/
*.csv
*.parquet
*.pkl
*.h5

# Models
models/
*.pt
*.pth
*.onnx
*.joblib

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/

# Logs
*.log
logs/
EOF

echo "=== Project created: ~/projects/$PROJECT_NAME ==="
echo "To start:"
echo "  cd ~/projects/$PROJECT_NAME"
echo "  uv run jupyter lab --ip=0.0.0.0"