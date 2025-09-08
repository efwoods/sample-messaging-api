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

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # HuggingFace
    BASE_MODEL: str
    HUGGINGFACE_TOKEN: str

    # Stripe
    STRIPE_SECRET_KEY: str

    # AWS
    BUCKET_NAME: str
    AWS_ACCESS_KEY_ID: str
    AWS_REGION: str
    AWS_SECRET_ACCESS_KEY: str

    # Production
    PRODUCTION: bool

    class Config:
        env_file = ".env"


settings = Settings()
