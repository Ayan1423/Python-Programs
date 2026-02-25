students = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78,
    "David": 60,
    "Emma": 55
}

print("Student Grades above 75:")
for student, grade in students.items():
  if grade > 75:
    print(f"{student}: {grade}")