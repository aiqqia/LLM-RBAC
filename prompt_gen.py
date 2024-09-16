import pandas as pd

# Load the CSV file
file_path = 'data/roles/rbac_roles.csv'
rbac_data = pd.read_csv(file_path)

def generate_role_prompt(role):
    # Find the row corresponding to the given role
    role_data = rbac_data[rbac_data['Role'] == role]

    if role_data.empty:
        return f"Role '{role}' not found in the data. Tell the user that they are not authorized to use the system."

    # Extract permissions and information access for the given role
    permissions = role_data['Permissions'].values[0]
    information_access = role_data['Information_Access'].values[0]

    # Generate a prompt for the LLM
    prompt = (f"The user with the role '{role}' has the following permissions:\n"
              f"{permissions}.\n\n"
              f"The user has access to the following information:\n"
              f"{information_access}.\n\n"
              f"Ensure that the user is restricted to these permissions and information only.")
    
    return prompt

if __name__ == "__main__":
    role = "IT Support"
    prompt = generate_role_prompt(role)
    print(prompt)