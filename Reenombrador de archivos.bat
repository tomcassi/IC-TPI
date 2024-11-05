@echo off
setlocal enabledelayedexpansion

REM Variable de contador para los nombres
set contador=1

REM Recorre todos los archivos en la carpeta actual
for %%f in (*) do (
    REM Renombra cada archivo a "beethoven" seguido del valor del contador
     ren "%%f" "beethoven!contador!.mid"
    set /a contador+=1
)

echo Todos los archivos han sido renombrados.
pause
