@echo off
set EXE=crsample_Win32_Release.exe
if not exist %EXE% set EXE=crsample.exe
set LOG=benchmark.log

set SCENENAME_A=FairyForest
set SCENESPEC_A=--mesh=scenes/fairyforest/fairyforest.obj --camera="cIxMx/sK/Ty/EFu3z/5m9mWx/YPA5z/8///m007toC10AnAHx///Uy200" --camera="KI/Qz/zlsUy/TTy6z13BdCZy/LRxzy/8///m007toC10AnAHx///Uy200" --camera="mF5Gz1SuO1z/ZMooz11Q0bGz/CCNxx18///m007toC10AnAHx///Uy200" --camera="vH7Jy19GSHx/YN45x//P2Wpx1MkhWy18///m007toC10AnAHx///Uy200" --camera="ViGsx/KxTFz/Ypn8/05TJTmx1ljevx18///m007toC10AnAHx///Uy200"

set SCENENAME_B=Buddha
set SCENESPEC_B=--mesh=scenes/buddha/buddha.obj --camera="HPI2v1F8J8y/KWjMy/5obv7w15:Rax18jVXcw/sZPF10aFuns/cH0ay/0" --camera="5/1dv/wd3Hy/cxYIx/55Mifx1cH3Mx/8jVXcw/sZPF10aFuns/cH0ay/0" --camera="WXz2w/ZSfVy/Bw07x/45H26z1cBWfx18jVXcw/sZPF10aFuns/cH0ay/0" --camera="BSTSx/VdErx/ou1Yx/5fIYAz1B7WEz/8jVXcw/sZPF10aFuns/cH0ay/0" --camera="MKYVx1JieYx/TeR6y/57Tsfy/wljmy/8jVXcw/sZPF10aFuns/cH0ay/0"

%EXE% bench-single %SCENESPEC_A% --image=%SCENENAME_A%.png
%EXE% bench-single %SCENESPEC_B% --image=%SCENENAME_B%.png

echo. >%LOG%
echo Table 1: Resolutions >>%LOG%
echo -------------------- >>%LOG%
echo. >>%LOG%

echo %SCENENAME_A% / CudaRaster: >>%LOG%    & %EXE% bench-resolution --renderer=cuda %SCENESPEC_A% --log=%LOG%
echo %SCENENAME_A% / OpenGL: >>%LOG%        & %EXE% bench-resolution --renderer=gl %SCENESPEC_A% --log=%LOG%
echo %SCENENAME_B% / CudaRaster: >>%LOG%    & %EXE% bench-resolution --renderer=cuda %SCENESPEC_B% --log=%LOG%
echo %SCENENAME_B% / OpenGL: >>%LOG%        & %EXE% bench-resolution --renderer=gl %SCENESPEC_B% --log=%LOG%

echo. >>%LOG%
echo Table 2: Rendering modes >>%LOG%
echo ------------------------ >>%LOG%
echo. >>%LOG%

echo %SCENENAME_A% / CudaRaster: >>%LOG%    & %EXE% bench-rendermode --renderer=cuda %SCENESPEC_A% --log=%LOG%
echo %SCENENAME_A% / OpenGL: >>%LOG%        & %EXE% bench-rendermode --renderer=gl %SCENESPEC_A% --log=%LOG%
echo %SCENENAME_B% / CudaRaster: >>%LOG%    & %EXE% bench-rendermode --renderer=cuda %SCENESPEC_B% --log=%LOG%
echo %SCENENAME_B% / OpenGL: >>%LOG%        & %EXE% bench-rendermode --renderer=gl %SCENESPEC_B% --log=%LOG%

echo. >>%LOG%
echo Table 3: Profiling >>%LOG%
echo ------------------ >>%LOG%
echo. >>%LOG%

echo %SCENENAME_A%: >>%LOG%                 & %EXE% bench-profile %SCENESPEC_A% --log=%LOG%
echo %SCENENAME_B%: >>%LOG%                 & %EXE% bench-profile %SCENESPEC_B% --log=%LOG%

echo. >>%LOG%
echo Figure 5: Synthetic test >>%LOG%
echo ------------------------ >>%LOG%
echo. >>%LOG%

echo CudaRaster: >>%LOG%                    & %EXE% bench-synthetic --renderer=cuda --depth=0 --lerp=0 --log=%LOG%
echo OpenGL: >>%LOG%                        & %EXE% bench-synthetic --renderer=gl --depth=0 --lerp=0 --log=%LOG%

echo. >>%LOG%
echo Figure 6: Batching >>%LOG%
echo ------------------ >>%LOG%
echo. >>%LOG%

echo %SCENENAME_A% / CudaRaster: >>%LOG%    & %EXE% bench-batch --renderer=cuda %SCENESPEC_A% --log=%LOG%
echo %SCENENAME_A% / OpenGL: >>%LOG%        & %EXE% bench-batch --renderer=gl %SCENESPEC_A% --log=%LOG%
echo %SCENENAME_B% / CudaRaster: >>%LOG%    & %EXE% bench-batch --renderer=cuda %SCENESPEC_B% --log=%LOG%
echo %SCENENAME_B% / OpenGL: >>%LOG%        & %EXE% bench-batch --renderer=gl %SCENESPEC_B% --log=%LOG%

type %LOG%
