numbers = input("Enter a list of numbers separated by spaces: ").split()
numbers = [int(num) for num in numbers]

total = sum(numbers)
average = total / len(numbers)

print("Sum:", total)
print("Average:", average)