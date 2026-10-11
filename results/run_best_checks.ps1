Set-Location E:\Telugu_Hate_Speech_Detector
$env:PYTHONIOENCODING='utf-8'
$py = '.venv\Scripts\python.exe'
function Step($name, $args_) { "=== $name $(Get-Date -Format HH:mm)" | Out-File -Append results\best_checks.log; & $py -I @args_ *>> results\best_checks.log }
Step calibration @('src/calibration.py','--model_dir','models/best','--out','results/calibration.json')
Step policy @('src/set_policy.py','models/best')
Step export @('src/export_best.py','--run','xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024','--out','results/final_best')
Step robustness @('src/robustness.py','--model_dir','models/best','--n','200','--out','results/robustness_results.csv')
Step stress @('src/stress_test.py','--model_dir','models/best','--n','300','--out','results/stress_test')
Step errors @('src/analysis.py','--action','error_analysis','--model_dir','models/best')
"=== DONE $(Get-Date -Format HH:mm)" | Out-File -Append results\best_checks.log
