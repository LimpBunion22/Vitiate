
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def log(message, gravitas = ""):

    if gravitas == "INFO":
        print(OKBLUE + gravitas + ": " + message + ENDC)
    elif gravitas == "WARNING":
        print(WARNING + gravitas + ": " + message + ENDC)
    elif gravitas == "ERROR":
        print(FAIL + gravitas + ": " + message + ENDC)
    elif gravitas == "FATAL":
        print(BOLD + FAIL + gravitas + ": " + message + ENDC)
    else:
        print(gravitas + ": " + message)

    return