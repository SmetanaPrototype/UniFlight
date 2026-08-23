echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" DESKTOP-Q5AGLJJ 61155 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 22308) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 756) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 16664) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 3324) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 13544) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 20864) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 21056) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 21688)
del "E:\UniFlight\ansys\cleanup-fluent-DESKTOP-Q5AGLJJ-21056.bat"
