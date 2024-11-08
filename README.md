# Role Based Access Control in Large Language Models

This is the code repository for implementing and exploring different ways to achieve role based access control in Large Language Models. I have tried out different methods like RAG and fine tuning. In RAG, a layer is being built between the user and the LLM which acts as system for role based access control.

## Table of Contents
- [Installation](#installation)
- [SetUp](#setup)
- [Usage](#usage)

### Installation & Requirements

The following commands need to be run in order to run the project:
```
!pip install faker
!pip install pandas
!pip install -U langchain-openai
!pip install python-magic
!pip install unstructured
!pip install chromadb
!pip install langchain-community
!pip install python-dotenv
!pip install matplotlib
!pip install seaborn
```

### Setup
Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/aiqqia/LLM-RBAC.git
   cd LLM-RBAC
   ```

### Usage

Running these files to generate relevant files for testing:

1. Run this to generate synthetic database for employee records:
```
python3 data_gen.py
```
2. Run this to create a CSV file for all roles, and its access information:
```
python3 role_gen.py
```
3. Run this to vectorize the entire database into chunks and store it in the Chroma database:
```
python3 generate_database.py
```
4. Finally, we can query the interface and provide the role and ask it any question:
```
python3 query_data "My Role" "My Question"
```

Example: 
```
python3 query_data "HR Manager" "What is Cheryl Mack's salary?"
```

