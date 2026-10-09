@echo off
REM One-click Windows runner for FLO CLTV project.
python cltv_senior_refactored.py --data flo_data_20k.csv --output outputs_cltv_senior --bootstrap-iterations 1000
if errorlevel 1 exit /b %errorlevel%
python cltv_senior_refactored.py --output outputs_cltv_senior --excel-only
if errorlevel 1 exit /b %errorlevel%
echo CLTV pipeline and Excel exports completed successfully.
