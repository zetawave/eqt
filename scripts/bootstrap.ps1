$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$lock = Get-Content "$root/dependencies.json" -Raw | ConvertFrom-Json
$dest = "$root/third_party/llama.cpp"
if (!(Test-Path "$dest/.git")) {
    git init $dest
    if ($LASTEXITCODE) { throw 'git init failed' }
    git -C $dest remote add origin $lock.llama_cpp.url
    git -C $dest fetch --depth 1 origin $lock.llama_cpp.revision
    if ($LASTEXITCODE) { throw 'Dependency fetch failed' }
    git -C $dest checkout --detach FETCH_HEAD
}
$actual = git -C $dest rev-parse HEAD
if ($actual -ne $lock.llama_cpp.revision) { throw 'Dependency revision mismatch; preserve local work and resolve manually.' }
if (git -C $dest status --porcelain) { throw 'Dependency is modified; record or revert changes explicitly.' }
Write-Output "llama.cpp $actual"
