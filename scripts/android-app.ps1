param(
    [ValidateSet('Debug', 'Release')][string]$Variant = 'Debug',
    [switch]$Install,
    [string]$Serial
)
# Builds the Equity APK. Release builds are minified and signed with a local keystore that is created on first
# use and never committed (android/keystore/, android/keystore.properties). Back it up: an app signed with a lost
# key cannot be updated in place.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$android = Join-Path $root 'android'
& "$PSScriptRoot/bootstrap.ps1"

# GPU mode: GGML's Vulkan shader generator is compiled for this PC with Visual Studio's C++ tools.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (!$env:VCToolsInstallDir -and (Test-Path $vswhere)) {
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vs) {
        cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] }
        }
    }
}
# A running Gradle daemon keeps the environment it started with, so GPU builds use a fresh one.
$gradleFlags = if ($env:VCToolsInstallDir) { @('--no-daemon', '-Pequity.vulkan=true') } else {
    Write-Warning 'Visual Studio C++ tools not found: building without the GPU mode.'
    @()
}

if ($Variant -eq 'Release') {
    $properties = Join-Path $android 'keystore.properties'
    if (!(Test-Path $properties)) {
        $keytool = if ($env:JAVA_HOME) { Join-Path $env:JAVA_HOME 'bin/keytool.exe' } else { 'keytool' }
        $directory = Join-Path $android 'keystore'
        New-Item -ItemType Directory -Force $directory | Out-Null
        $password = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
        & $keytool -genkeypair -v -keystore (Join-Path $directory 'equity-release.jks') -alias equity -keyalg RSA `
            -keysize 4096 -validity 10000 -storepass $password -keypass $password -dname 'CN=Equity, O=Equity'
        if ($LASTEXITCODE) { throw 'keytool failed' }
        @("storeFile=keystore/equity-release.jks", "storePassword=$password", "keyAlias=equity", "keyPassword=$password") |
            Set-Content -Encoding ascii $properties
        Write-Output "Created a local release keystore in $directory (back it up; it is not in Git)."
    }
}

& "$android/gradlew.bat" -p $android @gradleFlags "assemble$Variant"
if ($LASTEXITCODE) { throw "Gradle assemble$Variant failed" }
$apk = Join-Path $android "app/build/outputs/apk/$($Variant.ToLowerInvariant())/app-$($Variant.ToLowerInvariant()).apk"
Write-Output "APK: $apk"

if ($Install) {
    $device = if ($Serial) { @('-s', $Serial) } else { @() }
    & adb @device install -r $apk
    if ($LASTEXITCODE) { throw 'adb install failed (a debug and a release build cannot replace each other: uninstall first)' }
}
