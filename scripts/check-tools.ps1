param(
    [string]$BaseUrl = "http://127.0.0.1:8080/v1",
    [ValidateSet("coder", "agent")]
    [string]$Model = "coder"
)

$tool = @{
    type = "function"
    function = @{
        name = "read_file"
        description = "Read one file by path."
        parameters = @{
            type = "object"
            properties = @{ path = @{ type = "string" } }
            required = @("path")
        }
    }
}

function Invoke-Check([hashtable]$Body) {
    $json = $Body | ConvertTo-Json -Depth 12 -Compress
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/chat/completions" -ContentType "application/json" -Body $json
}

$request = @{
    model = $Model
    messages = @(@{ role = "user"; content = "Use read_file on README.md. Return only the tool request." })
    tools = @($tool)
    stream = $false
}
$reply = Invoke-Check $request
if ($reply.choices[0].finish_reason -ne "tool_calls") { throw "expected tool_calls, got $($reply.choices[0].finish_reason)" }
$call = $reply.choices[0].message.tool_calls[0]
if ($call.function.name -ne "read_file") { throw "unexpected tool: $($call.function.name)" }
$null = $call.function.arguments | ConvertFrom-Json

$none = $request.Clone(); $none.tool_choice = "none"; $none.messages = @(@{ role = "user"; content = "Say hello without tools." })
$reply = Invoke-Check $none
if ($reply.choices[0].finish_reason -eq "tool_calls" -or $reply.choices[0].message.tool_calls) { throw "tool_choice none produced a call" }

Write-Host "Tool checks passed for $Model at $BaseUrl"
