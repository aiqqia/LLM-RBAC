import csv

# Define the roles and their access details as a list of dictionaries
roles_data = [
    {
        "Role": "Administrator",
        "Permissions": "Full access: Read, Write, Execute, Manage Users",
        "Information_Access": "All Information: Access to all systems, configuration data, user accounts, logs, financial records, HR data, and authority to modify security policies, roles, and access rules."
    },
    {
        "Role": "IT Support",
        "Permissions": "Read, Execute, Troubleshoot",
        "Information_Access": "System Logs, User Accounts: Diagnostic data, system performance reports, user technical support data, but no access to confidential HR or financial information."
    },
    {
        "Role": "Security Officer",
        "Permissions": "Read, Monitor, Execute, Configure Security Policies",
        "Information_Access": "Access Logs, Security Policies, Encryption Keys: Focus on security events, audit logs, and role compliance monitoring."
    },
    {
        "Role": "Compliance Officer",
        "Permissions": "Read, Monitor",
        "Information_Access": "Audit Logs, Access Logs, Compliance Reports: Access to audit reports for regulatory compliance, and user behavior monitoring logs."
    },
    {
        "Role": "HR Manager",
        "Permissions": "Read, Write",
        "Information_Access": "Employee Data, Payroll Information, Benefits Data: Manage employee personal records, performance reviews, but no access to system or financial logs."
    },
    {
        "Role": "Finance Manager",
        "Permissions": "Read, Write, Approve Payments",
        "Information_Access": "Financial Records, Payroll Data, Budget Reports: Access to company accounts, budgets, payroll, but no access to employee personal data or system logs."
    },
    {
        "Role": "Project Manager",
        "Permissions": "Read, Write, Approve",
        "Information_Access": "Project Plans, Budget Forecasts, Employee Performance (limited): Oversee project-related data, budget forecasts, timelines, with limited access to team-specific HR data."
    },
    {
        "Role": "Developer",
        "Permissions": "Read, Write Code, Access Development Tools",
        "Information_Access": "Source Code, Technical Documentation, Development Data: Access to code repositories, project documentation, product blueprints, but no access to financial or employee personal data."
    },
    {
        "Role": "Salesperson",
        "Permissions": "Read, Update CRM",
        "Information_Access": "Customer Data, Sales Reports, Product Catalog: Access to customer details, sales records, CRM tools, but no access to HR, financial, or technical data."
    },
    {
        "Role": "Marketing Team",
        "Permissions": "Read, Write Campaign Data, Analyze Metrics",
        "Information_Access": "Customer Segments, Campaign Data, Website Analytics: Access to customer targeting, marketing data, campaign effectiveness, but no access to financial or HR data."
    },
    {
        "Role": "Legal Department",
        "Permissions": "Read, Review, Audit",
        "Information_Access": "Contracts, Compliance Reports, Legal Documentation: Access to legal contracts, compliance records, but no access to operational, HR, or system logs unless related to compliance cases."
    },
    {
        "Role": "Customer Service",
        "Permissions": "Read, Update Customer Data, Log Tickets",
        "Information_Access": "Customer Data, Support Tickets: Access to customer account details, support interactions, but no access to sales records or financial systems."
    },
    {
        "Role": "Auditor",
        "Permissions": "Read-Only",
        "Information_Access": "Access Logs, Financial Reports, System Logs: Read-only access to logs and reports for auditing purposes, no modification rights."
    },
    {
        "Role": "Intern",
        "Permissions": "Read, Limited Write (Department-Specific)",
        "Information_Access": "Limited Department Data: Temporary or restricted access based on department, limited access to project data, customer data for support, etc."
    },
    {
        "Role": "Client",
        "Permissions": "Read-Only (Self-Service Portals)",
        "Information_Access": "Own Account Information, Orders: Can access only their personal data, purchase history, and account settings, but no access to broader company systems or data."
    }
]

# Write data to CSV
csv_file = "data/roles/rbac_roles.csv"

# Open file and write
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Role", "Permissions", "Information_Access"])
    
    # Write header
    writer.writeheader()
    
    # Write rows from roles_data
    for role in roles_data:
        writer.writerow(role)

print(f"RBAC roles and access data successfully written to {csv_file}")
