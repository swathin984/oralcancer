@echo off
setlocal

set API_KEY=gUnVOYDxT4Dn61uIqIkd

set ROBOFLOW_API_KEY=%API_KEY%
echo environment variable set: ROBOFLOW_API_KEY=%API_KEY%

echo Verifying environment variable...
echo ROBOFLOW_API_KEY=%ROBOFLOW_API_KEY%

python app.py

pause