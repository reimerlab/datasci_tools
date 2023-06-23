'''

Purpose of module: a module that has:
1) Base class that a custom class can inherit from:
- will automatically run checks on the types (or convert data to write type) of data arguments are
- has pre-written functions like outputing dictionaries, jsons

2) Different ways to put constraints on data

Overall Purpose:
to more securily store data with checks

--- Base settings: 
link: https://pavledjuric.medium.com/how-to-configure-the-settings-of-your-python-app-with-pydantic-d8191113dcb8

What problem is it solving: 
1) how to store API keys and sensitive information in a class
without necessarily hard coding it
2) apply type hinting to check arguments and hard coded values

How does it do that? It will first try to extract these values from environment variables, and only if they are not found in the environment, it will read the hardcoded values

What happens: if something is defined as an environment variable, it will use that (
so don't have to hardcode it)

'''

def example():
    from datetime import datetime
    from typing import List, Optional
    from python_tools.pydantic import BaseModel
    
    """
    Example: will automatically convert id into the write data type
    """
    
    class User(BaseModel):
        id: int
        username : str
        password : str
        confirm_password : str
        alias = 'anonymous'
        timestamp: Optional[datetime] = None
        friends: List[int] = []
    
    
    data = {'id': '1234', 'username': 'wai foong', 'password': 'Password123', 'confirm_password': 'Password123', 'timestamp': '2020-08-03 10:30', 'friends': [1, '2', b'3']}
    user = User(**data)
    
    
    # ---- Ex: BaseSettings ---
    # -- run this in bash --
    export API_KEY=from_env_vars
    export DB_URL=from_env_vars_also
    
    """
    the value of API_KEY would actually be 'from_env_vars'
    because it is overwritten with the .env file
    """
    
    from python_tools.pydantic import BaseSettings, PostgresDsn
    class Settings(BaseSettings):
        API_KEY: str = 'some_secret'
        DB_URL: PostgresDsn = 'postgres://username:passw
            
            
    # --- Ex: Using env variable to define class variables
    class NeptuneSettings(BaseSettings):
        """
        Reads the variables from the environment.
        Errors will be raised if the required variables are not set.
        """
        api_key: str = Field(default=..., env="NEPTUNE")
        OWNER: str = "johschmidt42"  # set your name here, e.g. johndoe22
        PROJECT: str = "Heads"  # set your project name here, e.g. Heads
        EXPERIMENT: str = "heads"  # set your experiment name here, e.g. heads
        class Config:
            # this tells pydantic to read the variables from the .env file
            env_file = ".env"
