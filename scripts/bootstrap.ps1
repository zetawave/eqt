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
$patches = @($lock.llama_cpp.patches | ForEach-Object {
    $path = Join-Path $root $_.file
    if ((Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $_.sha256) { throw "Patch hash mismatch: $($_.file)" }
    $path
})
if (!(git -C $dest status --porcelain)) {
    foreach ($patch in $patches) {
        git -C $dest apply $patch
        if ($LASTEXITCODE) { throw "Cannot apply $patch" }
    }
}
# The working tree must equal the pinned revision plus exactly the recorded patches. git diff orders files by
# path, so both sides are compared per file; each file may be changed by one patch only.
function Split-FileDiffs([string]$text) {
    $sections = @{}
    foreach ($section in ($text.Replace("`r`n", "`n") -split '(?m)^(?=diff --git )')) {
        if (!$section.Trim()) { continue }
        $header = ($section -split "`n")[0]
        if ($sections.ContainsKey($header)) { throw "More than one recorded patch changes $header" }
        $sections[$header] = $section.Trim()
    }
    $sections
}
$expected = Split-FileDiffs (($patches | ForEach-Object { (Get-Content $_ -Raw).TrimEnd() + "`n" }) -join '')
$actual_diff = Split-FileDiffs ((git -C $dest diff) -join "`n")
$mismatch = @($expected.Keys + $actual_diff.Keys | Sort-Object -Unique | Where-Object { $expected[$_] -ne $actual_diff[$_] })
if ($mismatch) { throw "Dependency differs from revision + recorded patches ($($mismatch -join '; ')); record or revert changes explicitly." }
if (git -C $dest ls-files --others --exclude-standard) { throw 'Dependency has untracked files.' }
Write-Output "llama.cpp $actual + $($patches.Count) patch(es)"

# Header-only dependencies of the GPU mode (GGML's Vulkan backend), at pinned revisions.
foreach ($name in 'vulkan_headers', 'spirv_headers') {
    $pin = $lock.$name
    $path = Join-Path $root $pin.path
    if (!(Test-Path "$path/.git")) {
        git init $path
        if ($LASTEXITCODE) { throw 'git init failed' }
        git -C $path remote add origin $pin.url
        git -C $path fetch --depth 1 origin $pin.revision
        if ($LASTEXITCODE) { throw "Dependency fetch failed: $name" }
        git -C $path checkout --detach FETCH_HEAD
    }
    if ((git -C $path rev-parse HEAD) -ne $pin.revision) { throw "Dependency revision mismatch: $name" }
    if (git -C $path status --porcelain) { throw "Dependency has local changes: $name" }
    Write-Output "$name $($pin.revision)"
}
