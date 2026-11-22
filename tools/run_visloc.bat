@echo off
rem Double-click to run the real-imagery benchmark. Takes about an hour.
rem When it finishes, send back: AviNavi_data\uav_visloc_sample\visloc_out\visloc_results.json
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\run_visloc_benchmark.py --data "%USERPROFILE%\Desktop\AviNavi_data\uav_visloc_sample\UAV_VisLoc_example" %*
echo.
echo Finished. Send the file visloc_results.json from the visloc_out folder.
pause
