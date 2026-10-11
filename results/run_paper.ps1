Set-Location E:\Telugu_Hate_Speech_Detector
while (-not (Select-String -Path results\best_checks.log -Pattern '=== DONE' -Quiet -Encoding Unicode)) { Start-Sleep 20 }
$env:PYTHONIOENCODING='utf-8'
& .venv\Scripts\python.exe src/sweep.py --plan paper *> results\study\paper.log
