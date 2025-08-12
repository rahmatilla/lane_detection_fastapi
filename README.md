# Lane Detection API

This is a FastAPI-based lane detection service. The application is containerized using Docker and can be easily run locally.

---

## 🚀 Getting Started

### ✅ Prerequisites

- Docker
- Docker Compose
- Git

---

### 📦 Installation

1. **Clone the repository:**

```bash
git clone https://gitlab.com/vision7555031/lane_detection.git
cd lane_detection
```

2. **Create a **``** file:**

Create a `.env` file in the project root directory and define the following environment variable:

```env
API_KEY=your_api_token_here
```

3. **Build and run the container:**

```bash
sudo docker-compose build
sudo docker-compose up
```

The API will be available at:\
📍 `http://localhost:8002`


