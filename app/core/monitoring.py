from prometheus_client import Counter, Gauge, REGISTRY


def get_or_create_metric(name, description, metric_type="counter", labelnames="None"):
    """Utility function to create or retrieve a Prometheus metric."""
    registry = REGISTRY._names_to_collectors
    labelnames = labelnames or []
    if name in registry:
        return registry[name]
    if metric_type == "counter":
        return Counter(name, description)
    elif metric_type == "gauge":
        if labelnames == "None":
            return Gauge(name, description)
        else:
            return Gauge(name, description, labelnames=labelnames)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")


# Health and WebSocket Metrics
health_requests = get_or_create_metric(
    "health_requests_total", "Total number of health check requests received by the API"
)
websocket_url_requests = get_or_create_metric(
    "websocket_url_requests_total",
    "Total number of WebSocket URL requests made to the API",
)
active_websockets = get_or_create_metric(
    "active_websockets", "Number of currently active WebSocket connections", "gauge"
)
transcriptions_processed = get_or_create_metric(
    "transcriptions_processed_total",
    "Total number of audio transcriptions processed by the system",
)
ngrok_connections = get_or_create_metric(
    "ngrok_connections_total",
    "Total number of successful ngrok connections established",
)
ngrok_errors = get_or_create_metric(
    "ngrok_errors_total", "Total number of errors encountered during ngrok connections"
)
websocket_errors = get_or_create_metric(
    "websocket_errors_total", "Total number of errors in WebSocket operations"
)

# Database and GitHub Metrics
db_connection_status = get_or_create_metric(
    "db_connection_status",
    "Current database connection status (1=connected, 0=disconnected)",
    "gauge",
    labelnames=["database"],
)
github_gist_update_errors = get_or_create_metric(
    "github_gist_update_errors_total",
    "Total number of errors during GitHub Gist updates",
)
github_gist_url_updated = get_or_create_metric(
    "github_gist_url_updated",
    "Status of GitHub Gist URL updates with ngrok URL (1=success, 0=failure)",
    "gauge",
    labelnames=["database"],
)

# S3 Metrics
s3_connections = get_or_create_metric(
    "s3_connections_total", "Total number of successful S3 connection attempts"
)
s3_connection_errors = get_or_create_metric(
    "s3_connection_errors_total", "Total number of errors during S3 connection attempts"
)
s3_operations_success = get_or_create_metric(
    "s3_operations_success_total",
    "Total number of successful S3 operations (e.g., uploads, deletes)",
)
s3_operations_errors = get_or_create_metric(
    "s3_operations_errors_total",
    "Total number of errors during S3 operations (e.g., uploads, deletes)",
)

# User Management Metrics
user_signup_requests = get_or_create_metric(
    "user_signup_requests_total", "Total number of user signup requests received"
)
user_signup_errors = get_or_create_metric(
    "user_signup_errors_total", "Total number of errors during user signup attempts"
)
user_signup_success = get_or_create_metric(
    "user_signup_success_total", "Total number of successful user signups"
)
user_update_requests = get_or_create_metric(
    "user_update_requests_total", "Total number of requests to update user information"
)
user_update_errors = get_or_create_metric(
    "user_update_errors_total", "Total number of errors during user information updates"
)
user_update_success = get_or_create_metric(
    "user_update_success_total", "Total number of successful user information updates"
)

# Model and Embedding Metrics
model_loads = get_or_create_metric(
    "model_loads_total",
    "Total number of times the embedder model was successfully loaded",
)
model_load_errors = get_or_create_metric(
    "model_load_errors_total", "Total number of errors during embedder model loading"
)

# Document Management Metrics
documents_retrieved_total = get_or_create_metric(
    "documents_retrieved_total", "Total number of documents retrieved from the database"
)
documents_updated_total = get_or_create_metric(
    "documents_updated_total",
    "Total number of documents successfully updated in the database",
)
documents_created_total = get_or_create_metric(
    "documents_created_total",
    "Total number of documents successfully created in the database",
)
documents_deleted_total = get_or_create_metric(
    "documents_deleted_total",
    "Total number of documents successfully deleted from the database",
)

# Text Generation and Redis Metrics
text_generation_errors = get_or_create_metric(
    "text_generation_errors_total",
    "Total number of errors during text generation by the avatar model",
)
redis_operations_total = get_or_create_metric(
    "redis_operations_total",
    "Total number of successful Redis operations (e.g., get, set)",
)
redis_errors = get_or_create_metric(
    "redis_errors_total", "Total number of errors during Redis operations"
)

# API Metrics
api_errors = get_or_create_metric(
    "api_errors_total", "Total number of errors encountered in API endpoints"
)

avatars_selected_total = get_or_create_metric(
    "avatars_selected_total", "Total Number of Selected Avatars"
)

avatars_created_total = get_or_create_metric(
    "avatars_created_total", "Total Number of Created Avatars"
)
avatars_deleted_total = get_or_create_metric(
    "avatars_deleted_total", "Total Number of Deleted Avatars"
)

avatars_updated_total = get_or_create_metric(
    "avatars_updated_total", "Total Number of Updated Avatars"
)


class Metrics:
    def __init__(self):
        self.health_requests = health_requests
        self.websocket_url_requests = websocket_url_requests
        self.active_websockets = active_websockets
        self.transcriptions_processed = transcriptions_processed
        self.ngrok_connections = ngrok_connections
        self.ngrok_errors = ngrok_errors
        self.websocket_errors = websocket_errors
        self.db_connection_status = db_connection_status
        self.github_gist_update_errors = github_gist_update_errors
        self.github_gist_url_updated = github_gist_url_updated
        self.s3_connections = s3_connections
        self.s3_connection_errors = s3_connection_errors
        self.s3_operations_success = s3_operations_success
        self.s3_operations_errors = s3_operations_errors
        self.user_signup_requests = user_signup_requests
        self.user_signup_errors = user_signup_errors
        self.user_signup_success = user_signup_success
        self.user_update_requests = user_update_requests
        self.user_update_errors = user_update_errors
        self.user_update_success = user_update_success
        self.model_loads = model_loads
        self.model_load_errors = model_load_errors
        self.documents_retrieved_total = documents_retrieved_total
        self.documents_updated_total = documents_updated_total
        self.documents_created_total = documents_created_total
        self.documents_deleted_total = documents_deleted_total
        self.text_generation_errors = text_generation_errors
        self.redis_operations_total = redis_operations_total
        self.redis_errors = redis_errors
        self.api_errors = api_errors
        self.avatars_selected_total = avatars_selected_total
        self.avatars_created_total = avatars_created_total
        self.avatars_deleted_total = avatars_deleted_total 
        self.avatars_updated_total = avatars_updated_total


metrics = Metrics()
