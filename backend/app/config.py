from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

'''
This file intializes the project settings, so that the other backend files don't have to implement them again

This includes: 
- database url
- deepseek api key
- openai api key
- llm provider
- models
'''

class Settings (BaseSettings):
    app_name: str = "Diabetes Paper Helper"
    database_url: str = 'sqlite:///./project_databse.db'
    openai_api_key:str | None  = None
    deepseek_api_key:str | None = None
    llm_provider:str  = 'nothing_yet'
    model_config = SettingsConfigDict(env_file = '.env', extra = 'ignore')
    #extra = ignore avoids the code crashes when .env doesn't have some mentioned variables

@lru_cache #least recently cached: only run first time, load it from then
def get_settings():
    return Settings()
