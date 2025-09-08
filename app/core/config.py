from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB
    MONGO_DB: str
    MONGO_URI: str

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_USERNAME: str

    # HuggingFace
    BASE_MODEL: str
    HUGGINGFACE_TOKEN: str

    # AWS
    BUCKET_NAME: str
    AWS_REGION: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_ACCESS_KEY_ID: str

    # Production
    PRODUCTION: bool

    class Config:
        env_file = ".env"


settings = Settings()
