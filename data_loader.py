# debug_loader.py
import os
import glob
from datetime import datetime
from data_loader import (
    get_sp500_tickers_from_wikipedia,
    get_sp500_historical_changes,
    get_all_available_tickers_with_historical_validation
)

print("=" * 60)
print("DIAGNÓSTICO DE DATA_LOADER")
print("=" * 60)

# 1. Verificar archivos CSV disponibles
csv_files = glob.glob("data/*.csv")
print(f"\n1. ARCHIVOS CSV DISPONIBLES: {len(csv_files)}")
if csv_files:
    # Mostrar algunos ejemplos
    examples = [os.path.basename(f).replace('.csv', '') for f in csv_files[:10]]
    print(f"   Ejemplos: {examples}")

# 2. Verificar tickers actuales del S&P 500
print("\n2. TICKERS ACTUALES DEL S&P 500:")
current = get_sp500_tickers_from_wikipedia()
if current and 'tickers' in current:
    print(f"   Total: {len(current['tickers'])}")
    print(f"   Primeros 10: {current['tickers'][:10]}")

# 3. Verificar cambios históricos
print("\n3. CAMBIOS HISTÓRICOS:")
changes = get_sp500_historical_changes()
if not changes.empty:
    print(f"   Total cambios: {len(changes)}")
    print(f"   Removidos: {len(changes[changes['Action'] == 'Removed'])}")
    print(f"   Añadidos: {len(changes[changes['Action'] == 'Added'])}")
    
    # Ver algunos removidos recientes
    removed = changes[changes['Action'] == 'Removed'].head(5)
    print("\n   Últimos 5 removidos:")
    for _, row in removed.iterrows():
        print(f"     {row['Ticker']} - {row['Date']}")

# 4. Probar validación histórica
print("\n4. VALIDACIÓN HISTÓRICA (2020-01-01 a 2024-12-31):")
test_start = datetime(2020, 1, 1)
test_end = datetime(2024, 12, 31)

result, error = get_all_available_tickers_with_historical_validation(
    "SP500", test_start, test_end
)

if result:
    print(f"   Tickers finales: {len(result['tickers'])}")
    if len(result['tickers']) < 100:
        print(f"   ⚠️ MUY POCOS TICKERS!")
        print(f"   Lista completa: {result['tickers']}")
    else:
        print(f"   Primeros 20: {result['tickers'][:20]}")
    
    if 'removed_problematic' in result:
        print(f"   Problemáticos removidos: {result['removed_problematic']}")

# 5. Verificar intersección
print("\n5. ANÁLISIS DE INTERSECCIÓN:")
if current and result:
    current_set = set(current['tickers'])
    result_set = set(result['tickers'])
    
    # Tickers en current pero no en result
    missing = current_set - result_set
    print(f"   En actuales pero no en resultado: {len(missing)}")
    if missing:
        print(f"   Ejemplos faltantes: {list(missing)[:10]}")
    
    # Ver si los archivos CSV existen para los faltantes
    missing_with_csv = []
    for ticker in list(missing)[:20]:
        if os.path.exists(f"data/{ticker}.csv"):
            missing_with_csv.append(ticker)
    
    if missing_with_csv:
        print(f"\n   ⚠️ Tickers con CSV pero excluidos: {missing_with_csv}")

print("\n" + "=" * 60)
