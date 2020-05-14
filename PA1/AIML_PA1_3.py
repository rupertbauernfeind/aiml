def main():
    try:
        arr = stringToAsciiArr(input())
        bubbleSort(arr)

    except EOFError:
        return


def stringToAsciiArr(string):
    return [ord(c) for c in string]


def asciiArrToString(asciiArr):
    return ''.join(chr(i) for i in asciiArr)


def bubbleSort(arr):
    if len(arr) == 1:
        print(asciiArrToString(arr))

    while True:
        swapped = False
        for i in range(1, len(arr)):
            if arr[i - 1] > arr[i]:
                buff = arr[i]
                arr[i] = arr[i - 1]
                arr[i - 1] = buff
                swapped = True
        if not swapped:
            break
        else:
            print(asciiArrToString(arr))


if __name__ == "__main__":
    main()
