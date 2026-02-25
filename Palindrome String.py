def is_palindrome(text):
    return text == text[::-1]

word = input("Enter a string: ")

if is_palindrome(word):
    print("Palindrome")
else:
    print("Not a Palindrome")