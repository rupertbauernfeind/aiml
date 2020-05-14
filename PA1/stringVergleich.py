if __name__ == "__main__":
    file1 = open("../PA2/resultFile.txt", "r")
    file2 = open("../PA2/outputFile.txt", "r")

    st1 = ""
    st2 = ""

    for s in file1:
        st1 += s

    for s in file2:
        st2 += s

    if st1 == st2:
        print("gleich")
    else:
        print("ungleich")