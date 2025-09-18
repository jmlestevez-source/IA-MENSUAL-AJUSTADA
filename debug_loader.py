debug_loader.py# debug_loader.py
import os
import sys
import glob
from datetime import datetime
import pandas as pd

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

# 2. Verificar archivos de cambios históricos
print("\n2. ARCHIVOS DE CAMBIOS HISTÓRICOS:")
for file in ["sp500_changes.csv", "ndx_changes.csv", "data/sp500_changes.csv", "data/ndx_changes.csv"]:
    if os.path.exists(file):
        df = pd.read_csv(file)
        print(f"   ✅ {file}: {len(df)} registros")
    else:
        print(f"   ❌ {file}: No encontrado")

# 3. Ahora importar data_loader
print("\n3. IMPORTANDO data_loader.py...")
try:
    import data_loader
    print("   ✅ data_loader importado correctamente")
    
    # 4. Verificar tickers actuales del S&P 500
    print("\n4. TICKERS ACTUALES DEL S&P 500:")
    current = data_loader.get_sp500_tickers_from_wikipedia()
    if current and 'tickers' in current:
        print(f"   Total: {len(current['tickers'])}")
        print(f"   Primeros 10: {current['tickers'][:10]}")
    
    # 5. Verificar cambios históricos
    print("\n5. CAMBIOS HISTÓRICOS:")
    changes = data_loader.get_sp500_historical_changes()
    if not changes.empty:
        print(f"   Total cambios: {len(changes)}")
        print(f"   Removidos: {len(changes[changes['Action'] == 'Removed'])}")
        print(f"   Añadidos: {len(changes[changes['Action'] == 'Added'])}")
        
        # Ver algunos removidos recientes
        removed = changes[changes['Action'] == 'Removed'].head(5)
        print("\n   Últimos 5 removidos:")
        for _, row in removed.iterrows():
            print(f"     {row['Ticker']} - {row['Date']}")
    
    # 6. Probar validación histórica
    print("\n6. VALIDACIÓN HISTÓRICA (2020-01-01 a 2024-12-31):")
    test_start = datetime(2020, 1, 1)
    test_end = datetime(2024, 12, 31)
    
    result, error = data_loader.get_all_available_tickers_with_historical_validation(
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
    
    # 7. Verificar intersección
    print("\n7. ANÁLISIS DE INTERSECCIÓN:")
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

except ImportError as e:
    print(f"   ❌ Error importando data_loader: {e}")
except Exception as e:
    print(f"   ❌ Error general: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
