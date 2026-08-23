echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" DESKTOP-Q5AGLJJ 55524 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 15468) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 15268) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 15948) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 21640) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 13520) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 16984) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 20212) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 19976)
del "E:\UniFlight\ansys\cleanup-fluent-DESKTOP-Q5AGLJJ-20212.bat"
