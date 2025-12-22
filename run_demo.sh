#!/bin/bash
# DRUGVISTA Demo Launcher (Linux/Mac)
# ====================================

echo ""
echo "===================================="
echo " DRUGVISTA - AI Pharma Intelligence"
echo "===================================="
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[WARNING] OPENAI_API_KEY not set!"
    echo ""
    echo "Please set it first:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "Or create a .env file from .env.example"
    echo ""
    exit 1
fi

echo "[OK] OpenAI API key detected"
echo ""

# Start backend in background
echo "Starting backend server..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Start frontend
echo "Starting frontend..."
cd frontend
streamlit run app.py &
FRONTEND_PID=$!
cd ..

echo ""
echo "===================================="
echo " DRUGVISTA is running!"
echo "===================================="
echo ""
echo " Backend: http://localhost:8000"
echo " Frontend: http://localhost:8501"
echo " API Docs: http://localhost:8000/docs"
echo ""
echo " Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait

