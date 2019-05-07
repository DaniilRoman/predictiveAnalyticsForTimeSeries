inputArray = ['s','a','d', ' ', 'i', 's', ' ', 'l', 'i', 'f', 'e']

def revertWords(inputArray):
    revertChars(inputArray)
    revertCharsByWord(inputArray)

def revertChars(inputArray):
    lengthArr = len(inputArray) - 1
    for i in range(int(lengthArr/2)):
        tmp = inputArray[i]
        inputArray[i] = inputArray[lengthArr-i]
        inputArray[lengthArr-i] = tmp

def revertCharsByWord(inputArray):

    pass

def revertChars(inputArray, startIndex, endIndex):
    pass



revertWords(inputArray)