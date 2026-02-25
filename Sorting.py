numbers = [55, 21, 67, 37, 97]

for i in range(len(numbers)):
  for j in range(i + 1, len(numbers)):
    if numbers[i] > numbers[j]:
      numbers[i], numbers[j] = numbers[j], numbers[i]

print("Sorted listr:", numbers)

print(sorted(numbers,reverse=True))