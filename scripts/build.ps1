param([ValidateSet('Host','Android')][string]$Target = 'Android', [int]$Jobs = 2)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
& "$PSScriptRoot/bootstrap.ps1"
$cmake = "$env:ANDROID_HOME/cmake/3.22.1/bin/cmake.exe"
$ninja = "$env:ANDROID_HOME/cmake/3.22.1/bin/ninja.exe"
$build = "$root/build/$($Target.ToLowerInvariant())"
$arguments = @('-S', $root, '-B', $build, '-DCMAKE_BUILD_TYPE=Release')
if ($Target -eq 'Android') {
    $arguments += @('-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja",
        "-DCMAKE_TOOLCHAIN_FILE=$env:ANDROID_HOME/ndk/28.2.13676358/build/cmake/android.toolchain.cmake",
        '-DANDROID_ABI=arm64-v8a', '-DANDROID_PLATFORM=android-28', '-DANDROID_STL=c++_shared')
} else {
    $arguments += @('-G', 'Visual Studio 17 2022', '-A', 'x64')
}
& $cmake @arguments
if ($LASTEXITCODE) { throw 'CMake configure failed' }
$targets = @('eqt-bench', 'eqt-check', 'eqt-simd-test', 'eqt-kernel-bench')
if ($Target -eq 'Android') { $targets += 'eqt-io-bench' }
& $cmake --build $build --config Release --parallel $Jobs --target @targets
if ($LASTEXITCODE) { throw 'Native build failed' }
