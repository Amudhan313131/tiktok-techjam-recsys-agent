[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RexArgs
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

docker compose version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose v2 is required."
}

if (-not $env:REX_SOURCE_DIR) { $env:REX_SOURCE_DIR = $Root }
if (-not $env:REX_DATA_DIR) { $env:REX_DATA_DIR = Join-Path $Root "data/KuaiRand-Pure/data" }
if (-not $env:REX_RUNS_DIR) { $env:REX_RUNS_DIR = Join-Path $Root "runs" }
if (-not $env:REX_WORKER_IMAGE) { $env:REX_WORKER_IMAGE = "rex:local" }
if (-not $env:REX_SOURCE_COMMIT) {
    $env:REX_SOURCE_COMMIT = (git rev-parse HEAD).Trim()
}
if (-not $env:REX_PYPROJECT_SHA256) {
    $env:REX_PYPROJECT_SHA256 = (Get-FileHash -Algorithm SHA256 "pyproject.toml").Hash.ToLowerInvariant()
}
if (-not $env:REX_STARTER_MANIFEST_SHA256) {
    $env:REX_STARTER_MANIFEST_SHA256 = (Get-FileHash -Algorithm SHA256 "configs/frozen/starter_manifest.json").Hash.ToLowerInvariant()
}

$DockerArchitecture = (docker info --format '{{.Architecture}}').Trim()
switch -Regex ($DockerArchitecture) {
    '^(x86_64|amd64)$' { $LockArchitecture = "amd64"; break }
    '^(aarch64|arm64)$' { $LockArchitecture = "arm64"; break }
    default { throw "Unsupported Docker architecture: $DockerArchitecture" }
}
if (-not $env:REX_DEPENDENCY_LOCK_SHA256) {
    $LockPath = "requirements-lock-linux-$LockArchitecture.txt"
    $env:REX_DEPENDENCY_LOCK_SHA256 = (Get-FileHash -Algorithm SHA256 $LockPath).Hash.ToLowerInvariant()
}
if (-not $env:REX_IMAGE_PLATFORM) { $env:REX_IMAGE_PLATFORM = "linux/$LockArchitecture" }
if (-not $env:REX_DOCKER_GID) { $env:REX_DOCKER_GID = "0" }
if (-not $env:REX_UID) { $env:REX_UID = "10001" }
if (-not $env:REX_GID) { $env:REX_GID = "10001" }

New-Item -ItemType Directory -Force $env:REX_RUNS_DIR | Out-Null

$ComposeCommands = @("build", "config", "down", "images", "logs", "ls", "pull", "push", "stop")
if (($RexArgs.Count -gt 0) -and ($RexArgs[0] -eq "build")) {
    $DirtyStatus = (& git status --porcelain --untracked-files=normal) -join "`n"
    if ($DirtyStatus) {
        throw "Refusing a production image build from a dirty Git checkout."
    }
}
if (($RexArgs.Count -eq 0) -or -not ($ComposeCommands -contains $RexArgs[0])) {
    if (-not $env:REX_EXPECTED_IMAGE_DIGEST) {
        $ImageInspectOutput = & docker image inspect --format '{{.Id}}' $env:REX_WORKER_IMAGE 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Worker image '$($env:REX_WORKER_IMAGE)' is not present; run scripts/rex.ps1 build first."
        }
        $ImageDigest = ([string]$ImageInspectOutput).Trim()
        if (-not $ImageDigest) { throw "Docker returned an empty worker-image identity." }
        $env:REX_EXPECTED_IMAGE_DIGEST = $ImageDigest
        $env:REX_WORKER_IMAGE = $ImageDigest
    } elseif ($env:REX_WORKER_IMAGE -notmatch '^(sha256:[0-9a-f]{64}|.+@sha256:[0-9a-f]{64})$') {
        $LocalImageOutput = & docker image inspect --format '{{.Id}}' $env:REX_WORKER_IMAGE 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "REX_WORKER_IMAGE must be an immutable image ID or name@sha256 reference."
        }
        $LocalImageId = ([string]$LocalImageOutput).Trim()
        if ($LocalImageId -ne $env:REX_EXPECTED_IMAGE_DIGEST) {
            throw "REX_WORKER_IMAGE does not match REX_EXPECTED_IMAGE_DIGEST."
        }
        $env:REX_WORKER_IMAGE = $LocalImageId
    }
    if ($env:REX_EXPECTED_IMAGE_DIGEST -notmatch '^sha256:[0-9a-f]{64}$') {
        throw "REX_EXPECTED_IMAGE_DIGEST must be sha256 followed by 64 lowercase hex characters."
    }
    $ImageCommit = (& docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' $env:REX_WORKER_IMAGE).Trim()
    $ImageLock = (& docker image inspect --format '{{index .Config.Labels "org.rex.dependency-lock-sha256"}}' $env:REX_WORKER_IMAGE).Trim()
    $ImagePyproject = (& docker image inspect --format '{{index .Config.Labels "org.rex.pyproject-sha256"}}' $env:REX_WORKER_IMAGE).Trim()
    $ImageStarter = (& docker image inspect --format '{{index .Config.Labels "org.rex.starter-kit-sha256"}}' $env:REX_WORKER_IMAGE).Trim()
    if (($ImageCommit -ne $env:REX_SOURCE_COMMIT) -or
        ($ImageLock -ne $env:REX_DEPENDENCY_LOCK_SHA256) -or
        ($ImagePyproject -ne $env:REX_PYPROJECT_SHA256) -or
        ($ImageStarter -ne $env:REX_STARTER_MANIFEST_SHA256)) {
        throw "Worker image provenance does not match the clean source and locked environment."
    }
}
if ($RexArgs.Count -eq 0) {
    & docker compose run --rm rex doctor --config configs/run/production.yaml --tree --llm fixed
} elseif ($ComposeCommands -contains $RexArgs[0]) {
    & docker compose @RexArgs
} else {
    & docker compose run --rm rex @RexArgs
}
exit $LASTEXITCODE
