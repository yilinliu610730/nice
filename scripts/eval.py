import sys

sys.path.append(".")

from nice.abo import ABODataset

def main():

    dataset = ABODataset("data")
    for item in dataset:
        print(item)

if __name__ == '__main__':
    main()