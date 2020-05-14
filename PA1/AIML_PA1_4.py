def main():
    try:
        arr = stringToAsciiArr(input())
        insertionSort(arr)
    except EOFError:
        return


def stringToAsciiArr(string):
    return [ord(c) for c in string]


def asciiArrToString(asciiArr):
    return ''.join(chr(i) for i in asciiArr)


def insertionSort(arr):
    if len(arr) == 1:
        print(asciiArrToString(arr))

    for i in range(1, len(arr)):
        toSort = arr[i]
        k = i - 1
        while arr[k] > toSort and k >= 0:
            arr[k + 1] = arr[k]
            k = k - 1
        arr[k + 1] = toSort
        print(asciiArrToString(arr))


if __name__ == "__main__":
    main()
