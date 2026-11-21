<#
  通过 GitHub REST API 推送本地提交（不需要 github.com 直连，只需要 api.github.com）。

  为什么需要它：国内网络下 github.com:443 常被阻断，但 api.github.com 通常可访问，
  因此可以绕开 git 传输协议，直接用 Git Data API 上传对象并更新分支指针。

  用法（在本文件所在目录的上一级，即 P.A 目录下执行）：
      powershell -ExecutionPolicy Bypass -File tools\push_to_github.ps1

  可选参数：
      -RemoteName origin        远端名（默认 origin）
      -Branch     main          分支名（默认读取 origin 的跟踪分支）
      -DryRun                   只报告将要推送的提交，不写远端

  令牌来源：P.A\令牌.txt（首行整串），或环境变量 GITHUB_TOKEN。
  令牌需要 classic PAT 的 repo 权限，或 fine-grained PAT 的 Contents: Read and write。
#>

[CmdletBinding()]
param(
    [string]$RemoteName = 'origin',
    [string]$Branch = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoDir
try {
    # ---------- 令牌 ----------
    $tokenFile = Join-Path $repoDir '令牌.txt'
    if (Test-Path $tokenFile) {
        $token = (Get-Content $tokenFile -Raw -Encoding UTF8).Trim()
    }
    elseif ($env:GITHUB_TOKEN) {
        $token = $env:GITHUB_TOKEN.Trim()
    }
    else {
        throw "找不到令牌：请在 P.A\令牌.txt 写入 GitHub 令牌，或设置 GITHUB_TOKEN 环境变量。"
    }
    if (-not $token) { throw "令牌为空。" }

    # ---------- 仓库地址 ----------
    $remoteUrl = (git remote get-url $RemoteName)
    if ($remoteUrl -notmatch 'github\.com[:/](?<owner>[^/]+)/(?<repo>[^/.]+?)(\.git)?$') {
        throw "无法从远端地址解析 owner/repo：$remoteUrl"
    }
    $owner = $Matches['owner']
    $repoName = $Matches['repo']

    if (-not $Branch) {
        $Branch = (git rev-parse --abbrev-ref "$RemoteName/HEAD" 2>$null)
        if (-not $Branch -or $Branch -match 'HEAD') {
            $Branch = (git rev-parse --abbrev-ref HEAD)
        }
    }
    $Branch = $Branch -replace "^$RemoteName/", ''

    $apiBase = "https://api.github.com/repos/$owner/$repoName"
    $headers = @{
        Authorization = "Bearer $token"
        'User-Agent'  = 'photo-advisor-push'
        Accept        = 'application/vnd.github+json'
    }

    Write-Host "仓库    : $owner/$repoName"
    Write-Host "分支    : $Branch"

    function Invoke-Api {
        param([string]$Method, [string]$Url, $Body)
        $p = @{ Uri = $Url; Method = $Method; Headers = $headers; TimeoutSec = 180 }
        if ($null -ne $Body) {
            $p.Body = [System.Text.Encoding]::UTF8.GetBytes(($Body | ConvertTo-Json -Depth 12 -Compress))
            $p.ContentType = 'application/json; charset=utf-8'
        }
        Invoke-RestMethod @p
    }

    # 以二进制取 git 对象内容，避开 PowerShell 文本管线的编码/换行处理
    function Get-GitBytes {
        param([string[]]$GitArgs)
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = 'git'
        $psi.Arguments = ($GitArgs -join ' ')
        $psi.WorkingDirectory = $repoDir
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $proc = [System.Diagnostics.Process]::Start($psi)
        $ms = New-Object System.IO.MemoryStream
        $proc.StandardOutput.BaseStream.CopyTo($ms)
        $stderr = $proc.StandardError.ReadToEnd()
        $proc.WaitForExit()
        if ($proc.ExitCode -ne 0) { throw ("git {0} 失败：{1}" -f ($GitArgs -join ' '), $stderr) }
        return $ms.ToArray()
    }

    function Get-GitText {
        param([string[]]$GitArgs)
        return [System.Text.Encoding]::UTF8.GetString((Get-GitBytes $GitArgs))
    }

    # ---------- 远端当前状态 ----------
    $remoteHead = $null
    try {
        $ref = Invoke-Api 'GET' "$apiBase/git/ref/heads/$Branch" $null
        $remoteHead = $ref.object.sha
        Write-Host "远端 HEAD: $remoteHead"
    }
    catch {
        Write-Host "远端分支 $Branch 尚不存在，将创建。" -ForegroundColor Yellow
    }

    $localHead = (git rev-parse HEAD).Trim()
    Write-Host "本地 HEAD: $localHead"

    if ($remoteHead -eq $localHead) {
        Write-Host "本地与远端一致，无需推送。" -ForegroundColor Green
        return
    }

    # 远端提交必须在本地对象库里，否则无法计算增量
    if ($remoteHead) {
        git cat-file -e "$remoteHead^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "本地缺少远端提交 $remoteHead 的对象。请先联网执行 git fetch $RemoteName，或在能直连 github.com 的环境下同步。"
        }
        git merge-base --is-ancestor $remoteHead $localHead
        if ($LASTEXITCODE -ne 0) {
            throw "远端 $Branch 不是本地 HEAD 的祖先（历史已分叉），脚本不会强推。请先手工处理分叉。"
        }
        $toPush = @(git rev-list --reverse "$remoteHead..HEAD")
    }
    else {
        throw "远端分支为空时不支持用本脚本初始化，请先用常规 git push 建立首个提交。"
    }

    Write-Host ("待推送提交: " + $toPush.Count + " 个")
    foreach ($c in $toPush) {
        Write-Host ("  " + (git log -1 --format='%h %s' $c))
    }
    if ($DryRun) {
        Write-Host "DryRun：未写入远端。" -ForegroundColor Yellow
        return
    }

    # ---------- 逐个重放提交 ----------
    $prevRemoteSha = $remoteHead
    $prevTree = (Invoke-Api 'GET' "$apiBase/git/commits/$remoteHead" $null).tree.sha
    Write-Host "起始 tree: $prevTree"

    $index = 0
    foreach ($commitSha in $toPush) {
        $index++
        $parentLocal = (git rev-parse "$commitSha^").Trim()

        # 变更清单：以本地提交为准，内容一律取 HEAD 侧规范化 blob
        $diffLines = @(git diff --name-status $parentLocal $commitSha)
        $upload = @()
        $delete = @()
        foreach ($line in $diffLines) {
            $parts = $line -split "`t"
            $status = $parts[0]
            if ($status -like 'R*') { $delete += $parts[1]; $upload += $parts[2] }
            elseif ($status -eq 'D') { $delete += $parts[1] }
            else { $upload += $parts[1] }
        }

        Write-Host ("[{0}/{1}] {2}  ({3} 改动)" -f $index, $toPush.Count, (git log -1 --format='%h %s' $commitSha), ($upload.Count + $delete.Count))

        $entries = New-Object System.Collections.ArrayList
        foreach ($rel in $upload) {
            $blobSha = (git rev-parse "${commitSha}:$rel").Trim()
            $bytes = Get-GitBytes @('cat-file', 'blob', $blobSha)
            $blob = Invoke-Api 'POST' "$apiBase/git/blobs" @{
                content  = [Convert]::ToBase64String($bytes)
                encoding = 'base64'
            }
            if ($blob.sha -ne $blobSha) {
                throw "$rel 上传后的 blob 与本地不一致：$($blob.sha) != $blobSha"
            }
            $mode = '100644'
            $lsOut = (git ls-tree $commitSha -- $rel)
            if ($lsOut -and ($lsOut -split '\s+')[0] -eq '100755') { $mode = '100755' }
            $null = $entries.Add(@{ path = $rel; mode = $mode; type = 'blob'; sha = $blobSha })
        }
        foreach ($rel in $delete) {
            $null = $entries.Add(@{ path = $rel; mode = '100644'; type = 'blob'; sha = $null })
        }

        $tree = Invoke-Api 'POST' "$apiBase/git/trees" @{ base_tree = $prevTree; tree = @($entries) }
        $localTree = (git rev-parse "${commitSha}^{tree}").Trim()
        if ($tree.sha -ne $localTree) {
            throw "生成的 tree 与本地不一致：$($tree.sha) != $localTree"
        }

        # 提交信息取原始对象字节，保留首尾格式
        $rawCommit = Get-GitText @('cat-file', 'commit', $commitSha)
        $splitAt = $rawCommit.IndexOf("`n`n")
        $message = $rawCommit.Substring($splitAt + 2)
        $authorName = (git log -1 --format='%an' $commitSha).Trim()
        $authorEmail = (git log -1 --format='%ae' $commitSha).Trim()
        $authorDate = (git log -1 --format='%aI' $commitSha).Trim()
        $committerDate = (git log -1 --format='%cI' $commitSha).Trim()

        $newCommit = Invoke-Api 'POST' "$apiBase/git/commits" @{
            message   = $message
            tree      = $tree.sha
            parents   = @($prevRemoteSha)
            author    = @{ name = $authorName; email = $authorEmail; date = $authorDate }
            committer = @{ name = $authorName; email = $authorEmail; date = $committerDate }
        }

        if ($newCommit.sha -ne $commitSha) {
            Write-Host ("  注意：远端生成的提交 sha 与本地不同（{0} vs {1}）" -f $newCommit.sha.Substring(0, 7), $commitSha.Substring(0, 7)) -ForegroundColor Yellow
            Write-Host "  内容一致（tree 已校验），仅元数据格式不同；推送后可用 git fetch 对齐本地。" -ForegroundColor Yellow
        }
        else {
            Write-Host "  -> $($newCommit.sha.Substring(0, 7)) 与本地一致" -ForegroundColor Green
        }

        $prevRemoteSha = $newCommit.sha
        $prevTree = $tree.sha
    }

    # ---------- 更新分支指针 ----------
    $body = @{ sha = $prevRemoteSha; force = $false }
    if ($null -eq $remoteHead) { $body['force'] = $true }
    Invoke-Api 'PATCH' "$apiBase/git/refs/heads/$Branch" $body | Out-Null

    $after = Invoke-Api 'GET' "$apiBase/git/ref/heads/$Branch" $null
    Write-Host ("远端 $Branch 现为 $($after.object.sha)") -ForegroundColor Green

    if ($after.object.sha -eq $localHead) {
        git update-ref "refs/remotes/$RemoteName/$Branch" $after.object.sha
        Write-Host "本地跟踪引用已同步，git status 应显示干净。" -ForegroundColor Green
    }
    else {
        Write-Host "提示：远端 sha 与本地不同，执行 git fetch $RemoteName 后按需 git reset --hard $RemoteName/$Branch 对齐。" -ForegroundColor Yellow
    }
}
finally {
    Pop-Location
}
