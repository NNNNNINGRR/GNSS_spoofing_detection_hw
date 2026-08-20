$dir = 'results\full'
$recs = @()
Get-ChildItem "$dir\*_metrics.csv" | ForEach-Object {
    if ($_.BaseName -match '^(v[12])_(.+?)_(ds\d+)_metrics$') {
        $r = Import-Csv $_.FullName | Select-Object -First 1
        $add = $r.add_s
        $addv = if ($add -eq 'inf') { $null } else { [double]$add }
        $recs += [pscustomobject]@{
            ver = $matches[1]; model = $matches[2]; scen = $matches[3]
            roc = [double]$r.roc_auc; pr = [double]$r.pr_auc
            tpr1 = [double]$r.'tpr@fpr1'; tpr5 = [double]$r.'tpr@fpr5'
            f1 = [double]$r.f1; mcc = [double]$r.mcc
            hit = [int]$r.hit; add = $addv
        }
    }
}
"Macro average (8 scenarios)"
$recs | Group-Object ver, model | ForEach-Object {
    $g = $_.Group
    $adds = @($g | ForEach-Object { $_.add } | Where-Object { $null -ne $_ })
    [pscustomobject]@{
        Version = $_.Name.Split(',')[0]; Model = $_.Name.Split(',')[1]
        ROC_AUC = [math]::Round(($g | Measure-Object roc -Average).Average, 4)
        PR_AUC = [math]::Round(($g | Measure-Object pr -Average).Average, 4)
        TPR_FPR1 = [math]::Round(($g | Measure-Object tpr1 -Average).Average, 4)
        TPR_FPR5 = [math]::Round(($g | Measure-Object tpr5 -Average).Average, 4)
        F1 = [math]::Round(($g | Measure-Object f1 -Average).Average, 4)
        MCC = [math]::Round(($g | Measure-Object mcc -Average).Average, 4)
        HitRate = [math]::Round(($g | Measure-Object hit -Average).Average, 3)
        ADD_mean_s = if ($adds.Count) { [math]::Round(($adds | Measure-Object -Average).Average, 1) } else { 'inf' }
    }
} | Sort-Object Version, Model | Format-Table -AutoSize
"Key scenarios ds3/ds7:"
$recs | Where-Object { $_.scen -in @('ds3','ds7') } | Sort-Object scen, ver, model | Format-Table ver, model, scen, roc, tpr1, f1, add, hit -AutoSize
