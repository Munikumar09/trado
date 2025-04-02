# market_data_api

create conda enviroment with the following command:  
`conda create --name option-chain python=3.12.3`

If poetry not available install poetry:  
`curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2 -y`  
Refer the (Digital Ocean)[https://www.digitalocean.com/community/tutorials/how-to-install-poetry-to-manage-python-dependencies-on-ubuntu-22-04]

Install required pacakages from poetry with the following command:  
`poetry install`

Add the following in `.bashrc` file
```bash
    export SMARTAPI_CREDENTIALS="path/to/smart_api_credentials.json"
    gdrive_credentials_path="path/to/gdrive_credentials_path.json"
    export GDRIVE_CREDENTIALS_DATA=$(jq -c '.' "$gdrive_credentials_path")
```

Run the `main.py` with the `uvicorn` command:  
`uvicorn main:app --reload`
