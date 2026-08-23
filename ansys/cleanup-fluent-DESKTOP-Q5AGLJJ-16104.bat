echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" DESKTOP-Q5AGLJJ 56657 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 18992) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 3924) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 23904) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 3952) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 16104) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 22824)
del "E:\UniFlight\ansys\cleanup-fluent-DESKTOP-Q5AGLJJ-16104.bat"
