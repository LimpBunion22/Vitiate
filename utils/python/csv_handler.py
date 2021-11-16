
import csv



aux_matrix = None

def load_csv(csv_name):
    global aux_matrix
    aux_matrix = []
    with open(csv_name+".csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        x = 1
        for row in spamreader:
            if x>len(aux_matrix):
                aux_matrix.append(list())
            aux_matrix[x-1] = row
            x = x+1
    return

def save_csv(csv_name):
    global aux_matrix
    with open(csv_name+".csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in range(len(aux_matrix)):
            spamwriter.writerow(aux_matrix[row])
    return

def read_value(x,y):
    global aux_matrix
    return float(aux_matrix[x][y])

def write_value(x,y,value):
    global aux_matrix
    if not(isinstance(aux_matrix,list)):
        aux_matrix = []
    while x+1>len(aux_matrix):
        aux_matrix.append(list())
    if y+1>len(aux_matrix[x]):
        while y+1>len(aux_matrix[x]):
            aux_matrix[x].append(None)
    aux_matrix[x][y] = value
    return

