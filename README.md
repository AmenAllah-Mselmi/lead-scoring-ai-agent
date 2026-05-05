# Lead Scoring AI Agent 🚀

A high-performance microservice that uses machine learning to predict lead conversion probability. Designed for seamless integration into CRM ecosystems, this agent helps sales teams focus on the most promising prospects.

## 🌟 Key Features
- **Predictive Scoring**: Real-time lead scoring using a Random Forest model.
- **FastAPI Core**: High-performance, asynchronous API endpoints.
- **Containerized**: Fully Dockerized for consistent environments.
- **Cloud Ready**: Includes Kubernetes manifests and Helm charts for production-grade orchestration.
- **Automated Health Checks**: Built-in liveness and readiness probes for K8s.

## 🛠 Tech Stack
- **Language**: Python 3.11
- **Framework**: FastAPI + Uvicorn
- **ML Library**: scikit-learn, pandas, numpy
- **DevOps**: Docker, Kubernetes, Helm

## 🚀 Getting Started

### Prerequisites
- Docker installed
- Kubernetes cluster (Docker Desktop, Minikube, or EKS)
- Helm (optional)

### Running with Docker
```bash
# Build and run locally
docker build -t ai-agent .
docker run -p 8000:8000 ai-agent
```

### Deploying to Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/manifests/

# Or use Helm
helm install ai-agent ./helm/ai-agent-chart
```

## 📡 API Endpoints
- `GET /health`: Returns the health status of the agent.
- `POST /predict`: Submit lead data to receive a conversion score (0.0 to 1.0).

---
*Developed as part of the Lead Scoring AI Agent project.*
