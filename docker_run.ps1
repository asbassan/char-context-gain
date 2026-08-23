# docker_run.ps1 — build the KenLM experiment image and run it.
#
# SQLite cache, KenLM model files, corpus TXTs, and output CSVs all write
# to the local chapter1 directory via the volume mount — nothing lives inside
# the container.
#
# Usage:
#   .\docker_run.ps1               # build (if needed) then run
#   .\docker_run.ps1 -Build        # force-rebuild the image
#   .\docker_run.ps1 -Shell        # interactive shell for debugging

param(
    [switch]$Build,
    [switch]$Shell
)

$IMAGE  = "kenlm-experiment"
$MNTDIR = $PSScriptRoot   # the directory containing this script

# Build if image missing or -Build requested
$exists = docker image inspect $IMAGE 2>$null
if ($Build -or -not $exists) {
    Write-Host "Building $IMAGE ..." -ForegroundColor Cyan
    docker build -t $IMAGE "$MNTDIR"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Shell) {
    # Drop into bash for debugging
    docker run --rm -it `
        -v "${MNTDIR}:/workspace" `
        --entrypoint bash `
        $IMAGE
} else {
    # Run the experiment; SQLite written to host via volume
    docker run --rm `
        -v "${MNTDIR}:/workspace" `
        $IMAGE `
        run_experiment_v3.py `
        --backend kenlm `
        --db /workspace/experiment_cache_kenlm.db
}
