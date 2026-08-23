echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v242\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\tell.exe" DESKTOP-Q5AGLJJ 51527 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v242\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 5552) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 12232) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 7072) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 15064) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 17852) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 21452) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 16380) 
if /i "%LOCALHOST%"=="DESKTOP-Q5AGLJJ" (%KILL_CMD% 10740)
del "E:\UniFlight\ansys\cleanup-fluent-DESKTOP-Q5AGLJJ-16380.bat"
