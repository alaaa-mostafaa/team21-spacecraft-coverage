@echo off
rem Windows launcher - same commands as ./sim :
rem   sim                        viewer on the example design
rem   sim scenario.json          viewer on any scenario
rem   sim eval scenario.json     cost and constraints only
rem   sim optimise scenario.json --algorithm sa --evals 3000 --seed 1 --n-max 7 --save best.json
rem   sim validate [--quick]     validation of the current formulation -^> results\validation\
rem   sim experiments [--quick]  Milestone 3 experiments (legacy) -^> results\
rem   sim test                   run the tests
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% (
  echo Setting up .venv ^(first run only^) ...
  py -3 -m venv .venv || python -m venv .venv
  %PY% -m pip install -q -r requirements.txt
)
set "CMD=%~1"
set "ARGS="
shift
:collect
if "%~1"=="" goto dispatch
set ARGS=%ARGS% %1
shift
goto collect
:dispatch
if "%CMD%"=="" ( %PY% -m viz scenarios\orbital_example.json & goto :eof )
if "%CMD%"=="eval" ( %PY% -m spacecov %ARGS% & goto :eof )
if "%CMD%"=="optimise" ( %PY% -m spacecov %ARGS% --optimise & goto :eof )
if "%CMD%"=="validate" ( %PY% experiments\validate.py %ARGS% & goto :eof )
if "%CMD%"=="experiments" ( %PY% -m spacecov.legacy.m3.experiments.run_experiments %ARGS% & goto :eof )
if "%CMD%"=="test" ( %PY% -m pytest -q %ARGS% & goto :eof )
%PY% -m viz %CMD% %ARGS%
