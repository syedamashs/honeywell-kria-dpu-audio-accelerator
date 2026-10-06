@echo off
REM docker_build.bat - Helper script for building the KV260 Docker image on Windows

REM Ensure Docker is in PATH
set PATH=%PATH%;C:\Program Files\Docker\Docker\resources\bin;C:\Program Files\Docker\Docker\resources

echo ====================================================================
echo  AMD Kria KV260 Docker Image Builder
echo ====================================================================

REM Set default image tag
set DOCKER_USER=syedamash07
set IMAGE_NAME=kria-kv260-kws
set TAG=latest
set FULL_IMAGE=%DOCKER_USER%/%IMAGE_NAME%:%TAG%

echo.
echo [1/3] Building Docker image (%FULL_IMAGE%)...
docker build -t %IMAGE_NAME%:%TAG% -t %FULL_IMAGE% .

if %ERRORLEVEL% NEQ 0 (
    echo [!] Docker build failed! Ensure Docker Desktop is installed and running.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Build completed successfully!
echo.
echo [3/3] Available next steps:
echo   A. Run locally to test:
echo      docker run -p 8080:8080 %FULL_IMAGE%
echo.
echo   B. Export as offline .tar for the KV260 board:
echo      docker save -o kv260_kws_image.tar %FULL_IMAGE%
echo.
echo   C. Push to Docker Hub:
echo      docker push %FULL_IMAGE%
echo ====================================================================
