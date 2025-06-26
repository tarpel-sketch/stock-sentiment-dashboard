# Deployment Guide

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-sentiment-dashboard.git
cd stock-sentiment-dashboard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements_github.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy

## Heroku Deployment

1. Install Heroku CLI
2. Create a `Procfile` in the root directory:
```
web: sh setup.sh && streamlit run app.py
```

3. Create a `setup.sh` file:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

4. Deploy to Heroku:
```bash
git add .
git commit -m "Deploy to Heroku"
heroku create your-app-name
git push heroku main
```

## Docker Deployment

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_github.txt .
RUN pip install -r requirements_github.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t stock-sentiment-dashboard .
docker run -p 8501:8501 stock-sentiment-dashboard
```

## Environment Variables

For production deployments, consider setting:
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

## Performance Optimization

- Use caching for data processing with `@st.cache_data`
- Implement session state management for better user experience
- Consider using multiprocessing for large datasets
- Enable gzip compression for faster loading