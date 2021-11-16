import csv_handler

csv_handler.write_value(0,0,1)
csv_handler.write_value(0,1,2)
csv_handler.write_value(0,2,3)
csv_handler.write_value(1,0,4)

csv_handler.save_csv("test_csv")

csv_handler.load_csv("test_csv")
matrix = []
matrix.append(csv_handler.read_value(0,0))
matrix.append(csv_handler.read_value(0,1))
matrix.append(csv_handler.read_value(0,2))
print(matrix)