# llm-pdf-retrieval
This small project leverages Large Language Models (LLMs) to automatically extract structured data from a set of scholarly articles in PDF format. It uses Mistral, lightweight Retrieval-Augmented Generation (RAG) and LangChain to process the input documents and identify key details.

## Configuration

- Edit `config.py` to add your own `INPUT_PATH` and `OUTPUT PATH`.
- Edit `config.py` to add your own Mistral model under `MODEL_NAME`.
- Toggle `PARSER_USAGE` in  `config.py` to True if you would like to use a specific parser.

## Installation

To install and run the project locally, follow the steps below:


1. A Mistral API key is needed ([Get API Key](https://mistral.ai/)).

2. Install Python. Version 3.12 was used for this development.

3. Clone the repository from terminal (git must be installed): 

    ```bash
    git clone https://github.com/carobs9/llm-pdf-retrieval.git
    ```
    - [Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

4. Navigate to the project directory: 

    ```bash
    cd [YOUR PROJECT DIRECTORY]
    ```

5. Create a virtual environment:

    ```bash
    python3.12 -m venv <env_name>
    ```

    - [Creating Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

6. Activate the virtual environment: 

* Mac:

    ```bash
    source venv/bin/activate
    ```

* Windows:

    ```bash
    ./env_name>/Scripts/activate
    ```

* Linux:

    ```bash
    ./<env_name>/bin/activate
    ```
    - [Activating a virtual environment](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments)

6. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
    - [Installing Packages](https://packaging.python.org/tutorials/installing-packages/)
