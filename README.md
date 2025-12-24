# Razer Data Science Setup

Configuración completa de una workstation Ubuntu para Data Science y MLOps.

## Especificaciones del Sistema

| Componente | Especificación |
|------------|----------------|
| **OS** | Ubuntu 24.04.3 LTS |
| **CPU** | Intel Core i7-10875H (8 cores, 16 threads) |
| **GPU** | NVIDIA GeForce RTX 2070 Super Max-Q (8GB VRAM) |
| **RAM** | 16 GB |
| **Storage** | 1.8 TB SSD |
| **CUDA** | 13.0 (Driver 580.95.05) |

## Stack Instalado

### Core Data Science
- Python 3.12+ (via uv)
- NumPy, Pandas, SciPy, Scikit-learn
- Matplotlib, Seaborn
- JupyterLab 4.5.1

### Machine Learning
| Librería | Versión | Uso |
|----------|---------|-----|
| scikit-learn | 1.8.0 | ML clásico |
| XGBoost | 3.1.2 | Gradient Boosting |
| LightGBM | 4.6.0 | Gradient Boosting |
| CatBoost | 1.2.8 | Gradient Boosting |

### Deep Learning (GPU)
| Librería | Versión | Backend |
|----------|---------|---------|
| PyTorch | 2.6.0+cu124 | CUDA 12.4 |
| TensorFlow | 2.20.0 | CUDA |
| Keras | 3.13.0 | TensorFlow |
| Transformers | 4.57.3 | PyTorch/TF |

### MLOps
| Librería | Versión | Uso |
|----------|---------|-----|
| MLflow | 3.8.0 | Experiment tracking |
| Optuna | 4.6.0 | Hyperparameter tuning |
| Wandb | 0.23.1 | Experiment tracking |
| Great Expectations | 1.10.0 | Data validation |
| Evidently | 0.7.18 | ML monitoring |

### Model Serving
| Librería | Versión |
|----------|---------|
| FastAPI | 0.127.0 |
| Streamlit | 1.52.2 |
| Gradio | 6.2.0 |

### Explainability
- SHAP 0.50.0
- LIME 0.2.0.1

### Data Profiling
- ydata-profiling 4.18.0
- Sweetviz 2.3.1

### NLP & Computer Vision
- spaCy 3.8.11 + en_core_web_sm
- OpenCV 4.12.0
- Torchvision 0.21.0

### Web Scraping
- BeautifulSoup4, Selenium, Playwright
- HTTPX, lxml

### Bases de Datos
- PostgreSQL 16
- Redis 7
- SQLAlchemy 2.0

### Herramientas de Desarrollo
- VS Code
- Git + GitHub CLI
- Docker + Docker Compose
- R 4.3.3 + RStudio

### Seguridad
- UFW Firewall (SSH + Tailscale)
- Fail2ban (protección SSH)
- SSH hardening (solo llaves públicas)
- ClamAV antivirus

---

## Crear Nuevo Proyecto

### Método Rápido (con script)

```bash
# Proyecto estándar
./scripts/new-project.sh mi-proyecto standard

# Proyecto mínimo
./scripts/new-project.sh mi-proyecto minimal

# Proyecto completo
./scripts/new-project.sh mi-proyecto full

# Proyecto con GPU
./scripts/new-project.sh mi-proyecto gpu
```

### Método Manual

```bash
# 1. Crear proyecto
cd ~/projects
uv init mi-nuevo-proyecto
cd mi-nuevo-proyecto

# 2. Agregar dependencias base
uv add numpy pandas scikit-learn matplotlib seaborn jupyter jupyterlab

# 3. Agregar dependencias específicas según necesidad
uv add xgboost lightgbm          # Gradient boosting
uv add mlflow optuna              # MLOps
uv add fastapi uvicorn streamlit  # Serving
uv add ydata-profiling sweetviz   # Profiling

# 4. Para GPU (PyTorch)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Para GPU (TensorFlow)
uv pip install tensorflow

# 6. Iniciar Jupyter
uv run jupyter lab --ip=0.0.0.0
```

### Estructura de Proyecto Recomendada

```
mi-proyecto/
├── data/
│   ├── raw/              # Datos originales (nunca modificar)
│   ├── processed/        # Datos procesados
│   └── external/         # Datos de fuentes externas
├── notebooks/            # Jupyter notebooks
├── src/
│   ├── data/            # Scripts de carga/procesamiento
│   ├── features/        # Feature engineering
│   ├── models/          # Entrenamiento y predicción
│   └── visualization/   # Gráficos y reportes
├── reports/
│   └── figures/         # Gráficos exportados
├── tests/               # Tests unitarios
├── pyproject.toml       # Dependencias (uv)
└── README.md
```

---

## Comandos Útiles

### Jupyter Lab
```bash
# Desde proyecto específico
cd ~/projects/datascience
uv run jupyter lab --ip=0.0.0.0

# O usar alias
jlab
```

### GPU Check
```bash
# NVIDIA status
nvidia-smi

# PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"

# TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Docker
```bash
# Limpiar todo
docker system prune -af

# Ver contenedores
docker ps -a

# Ver imágenes
docker images
```

### Bases de Datos
```bash
# PostgreSQL
sudo -u postgres psql

# Redis
redis-cli ping
```

### Seguridad
```bash
# Estado firewall
sudo ufw status verbose

# IPs baneadas por Fail2ban
sudo fail2ban-client status sshd

# Escanear con ClamAV
clamscan -r /path/to/scan
```

---

## Conexión Remota

### SSH (via Tailscale)
```bash
# Desde Mac
ssh razer

# O con IP directa
ssh gonzalezulises@100.104.40.57
```

### VS Code Remote
1. Instalar extensión "Remote - SSH"
2. Conectar a `razer` o `gonzalezulises@100.104.40.57`

### Jupyter Lab Remoto
```bash
# En Razer
cd ~/projects/datascience && uv run jupyter lab --ip=0.0.0.0 --port=8888

# Desde Mac, abrir en navegador:
# http://100.104.40.57:8888
```

---

## Mejores Prácticas

### 1. Aislamiento de Proyectos
- Usar `uv` para cada proyecto independiente
- Nunca instalar paquetes globalmente
- Cada proyecto tiene su propio `pyproject.toml`

### 2. Control de Versiones
```bash
# Siempre ignorar
data/
models/
*.pkl
*.h5
.venv/
mlruns/
```

### 3. Reproducibilidad
```bash
# Exportar dependencias exactas
uv pip freeze > requirements.txt

# Usar pyproject.toml con versiones fijas
```

### 4. GPU Memory
```python
# PyTorch - liberar memoria
torch.cuda.empty_cache()

# TensorFlow - limitar memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 5. Experiment Tracking
```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

---

## Troubleshooting

### uv command not found
```bash
source ~/.bashrc
# o
export PATH="$HOME/.local/bin:$PATH"
```

### CUDA out of memory
```python
# Reducir batch size
# O liberar memoria
import torch
torch.cuda.empty_cache()
```

### Jupyter kernel no encuentra paquetes
```bash
# Asegurarse de usar uv run
uv run jupyter lab

# No usar jupyter lab directamente
```

### SSH connection refused
```bash
# Verificar servicio
sudo systemctl status sshd

# Reiniciar si es necesario
sudo systemctl restart sshd
```

---

## Actualizaciones

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Actualizar uv
uv self update

# Actualizar paquetes de proyecto
cd ~/projects/mi-proyecto
uv sync --upgrade
```

---

## Licencia

MIT License - Úsalo como quieras.
