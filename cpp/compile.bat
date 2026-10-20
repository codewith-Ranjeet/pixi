@echo off
echo Compiling Pixi C++ AVX2/FMA Inference Engine...
g++ -O3 -mavx2 -mf16c -mfma -std=c++20 pixi_infer.cpp -o pixi_infer.exe
if %ERRORLEVEL% EQU 0 (
    echo Compilation successful: pixi_infer.exe generated!
) else (
    echo Compilation failed!
)
