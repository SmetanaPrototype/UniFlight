def write_arrays_to_csv(filename, **arrays):
   """Запись массивов в CSV файл"""
   if not arrays:
       raise ValueError("Array is required.")

   os.makedirs(os.path.dirname(filename), exist_ok=True)

   headers = list(arrays.keys())
   max_length = min(len(arr) for arr in arrays.values())

   with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
       writer = csv.writer(csvfile)
       writer.writerow(headers)
       for i in range(max_length):
           row = [arrays[name][i] for name in headers]
           writer.writerow(row)
   print(f"Data was moved to '{filename}'.")


def read_array_from_csv(filename, arrayname):
    try:
        df = pd.read_csv(filename)
        if arrayname in df.columns:
            column_c = df[arrayname].tolist()
            return column_c
        else:
            print(f"Столбец '{arrayname}' не найден в файле {filename}")
            return None
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None