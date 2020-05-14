def main():
    try:
        arr = stringToAsciiArr(input())
        mergeSort(arr, 0, len(arr) - 1)
    except EOFError:
        return


def stringToAsciiArr(string):
    return [ord(c) for c in string]


def asciiArrToString(asciiArr):
    return ''.join(chr(i) for i in asciiArr)


def mergeSort(arr, indexLeft, indexRight):
    if indexLeft < indexRight:
        indexMiddle = int((indexLeft + indexRight) / 2)
        # Divide and provide a sorted list:
        mergeSort(arr, indexLeft, indexMiddle)
        mergeSort(arr, indexMiddle + 1, indexRight)
        # Merge the sorted lists
        merge(arr, indexLeft, indexMiddle, indexRight)


def merge(arr, indexLeft, indexMiddle, indexRight):
    # Get the sizes of the sorted sub-arrays
    leftSize = indexMiddle - indexLeft + 1
    rightSize = indexRight - indexMiddle

    # Buffer the sub-arrays for sorting
    lArrBuff = arr[indexLeft: indexMiddle + 1]
    rArrBuff = arr[indexMiddle + 1:indexRight + 1]

    left = 0
    right = 0

    for k in range(indexLeft, indexRight + 1):
        # if one arrBuff is solved:
        if left >= len(lArrBuff):
            arr[k] = rArrBuff[right]
            right = right + 1
        elif right >= len(rArrBuff):
            arr[k] = lArrBuff[left]
            left = left + 1
        elif lArrBuff[left] <= rArrBuff[right]:
            arr[k] = lArrBuff[left]
            left = left + 1
        else:
            arr[k] = rArrBuff[right]
            right = right + 1
    print(asciiArrToString(arr[indexLeft:indexRight + 1]))


if __name__ == "__main__":
    main()
