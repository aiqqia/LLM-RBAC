import random
import faker

# Initialize the Faker generator
fake = faker.Faker()

# Function to generate random employee data
def generate_employee_data(num_employees=100):
    employees = []
    for _ in range(num_employees):
        employee = {
            "employee_id": fake.uuid4(),
            "name": fake.name(),
            "address": fake.address().replace('\n', ', '),  # Replace newlines in address with commas
            "phone_number": fake.phone_number(),
            "salary": round(random.uniform(30000, 120000), 2)  # Random salary between 30k and 120k
        }
        employees.append(employee)
    return employees

# Generate 100 employees' information
employee_data = generate_employee_data()

# Save the employee data to a text file
with open("data/textfiles/database.txt", "w") as file:
    for employee in employee_data:
        file.write(f"Employee ID: {employee['employee_id']}\n")
        file.write(f"Name: {employee['name']}\n")
        file.write(f"Address: {employee['address']}\n")
        file.write(f"Phone Number: {employee['phone_number']}\n")
        file.write(f"Salary: ${employee['salary']}\n")
        file.write("\n" + "-"*40 + "\n\n")

print("Employee data has been generated and saved to data.txt.")
