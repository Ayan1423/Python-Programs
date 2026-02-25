def count(text):
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

x = input("Enter a string: ")
print("Number of vowels:", count(x))