Get-ChildItem "c:\Users\LENOVO\Downloads\F1\dataset\*.csv" | ForEach-Object {
    $c = (Get-Content $_.FullName).Count - 1
    Write-Host "$($_.Name): $c rows"
}
