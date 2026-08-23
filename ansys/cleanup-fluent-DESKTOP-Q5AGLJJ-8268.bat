echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" DESKTOP-Q5AGLJJ 54954 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 15972) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 16972) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 20252) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 9904) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 7000) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 6236) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 8268) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 16564)
del "E:\UniFlight\ansys\cleanup-fluent-DESKTOP-Q5AGLJJ-8268.bat"
