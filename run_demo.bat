@echo off
REM DRUGVISTA Demo Launcher (Windows)
REM ==================================

echo.
echo  ====================================
echo   DRUGVISTA - AI Pharma Intelligence
echo  ====================================
echo.

REM Check for API key
if "%OPENAI_API_KEY%"=="" (
    echo [WARNING] OPENAI_API_KEY not set!
    echo.
    echo Please set it first:
    echo   set OPENAI_API_KEY=your-key-here
    echo.
    echo Or create a .env file from .env.example
    echo.
    pause
    exit /b 1
)

echo [OK] OpenAI API key detected
echo.

REM Start backend in background
echo Starting backend server...
start "DRUGVISTA Backend" cmd /k "cd backend && python main.py"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

REM Start frontend
echo Starting frontend...
start "DRUGVISTA Frontend" cmd /k "cd frontend && streamlit run app.py"

echo.
echo ====================================
echo  DRUGVISTA is starting!
echo ====================================
echo.
echo  Backend: http://localhost:8000
echo  Frontend: http://localhost:8501
echo  API Docs: http://localhost:8000/docs
echo.
echo  Press any key to exit this launcher...
echo  (Backend and frontend will keep running)
echo.
pause > nul

