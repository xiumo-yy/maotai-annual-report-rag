@echo off
cd /d "%~dp0"

REM ============ 配置区 ============
REM 换电脑时只改下面这一行，指向装了 streamlit 的 python.exe
set "PY=E:\Users\lenovo\miniconda3\envs\langchain1.2\python.exe"
if "%PORT%"=="" set "PORT=8501"
REM ================================

if not exist "%PY%" (
    echo [错误] 找不到 Python：%PY%
    echo 请用记事本打开 run_ui.bat，把 PY 那一行改成你自己的 python.exe 路径。
    echo.
    pause
    exit /b 1
)

REM 服务已经在跑就直接开页面，重复双击不会报"端口被占用"
netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo 服务已在运行，正在打开页面：http://localhost:%PORT%
    start "" "http://localhost:%PORT%"
    exit /b 0
)

echo ============================================================
echo   贵州茅台 2024 年报 · 智能问答
echo.
echo   地址：http://localhost:%PORT%
echo   浏览器会在服务就绪后自动打开
echo.
echo   启动后请【最小化】本窗口，不要关闭。
echo   关闭本窗口就是停止服务，演示全部结束后再关。
echo ============================================================
echo.

"%PY%" -m streamlit run app.py --server.port=%PORT% --browser.gatherUsageStats=false

echo.
echo 服务已停止。
pause
