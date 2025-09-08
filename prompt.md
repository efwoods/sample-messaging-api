Please update the ModelManager and main.py to perform this behavior.

I need to be able to send a query to the query endpoint. The adapter name is the name of the avatar_id. The adapter is stored in s3. I need to lazy load the adapter. I need to cache recently used adapters in the container locally. This api needs to have the /query endpoint allow for the avatar_id to be sent as an optional parameter. If the current adapter is not the adapter for the avatar_id, then it will check the cache. If the adapter is cached then it will attach the adapter and complete the query. if not cached, it will check the s3 endpoint. If available in s3 it will pull and attach the adapter as well as cache the adapter in recent use. If the adapter is not available in s3, the base model will be queried and a response will be appended indicating that the adapter was not used. There needs to be an optional boolean to enable context via a vectorstore. When true, an external query to a chroma_db vectorstore is made and the response is appended to the model response. (the model will use the context)

For example, the context is first sent to the vectorstore via http, then the result is added to the prompt, tokenized, and the model is then queried. This endpoint will use an http client for chroma_db that will need to be added at a later time. 

Here is an example of the use:

def generate_with_context(user_input: str, top_k: int = 10, max_new_tokens: int = 50) -> str:
    # Embed query and retrieve from vector DB
    query_vec = embedder.encode(user_input).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    print(f"results: {results}")

    docs = results.get("documents", [[]])[0]
    if not docs:
        context = "No relevant context found."
    else:
        # Optionally truncate context length for model input token limits
        # Here we join and limit length (e.g., first 1000 chars)
        context = "\n".join(docs)
        context = context[:1000]

    # Improved prompt with explicit instruction and clear delimiters
    prompt = f"""You are the person in the contextual statements. Use the context to answer the question briefly and only once.

Context:
{context}

Q: {user_input}
A:"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)

    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Set True to enable sampling
            # temperature=0.7,  # Uncomment if do_sample=True
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract text after last "A:" in case prompt or output has multiple
    answer = decoded.split("A:")[-1].strip()
    return answer
generate_with_context(user_input=input())


----

The query endpoint needs to save the user query in mongo_db, push the query into the redis cache, perform inference as described above to produce a response, store the avatar response in mongo_db for the avatar_conversation and push the new response to the redis stream. It needs to provide a separate endpoint for Server Side Events that will update the frontend with every message in the redis cache.

Here is a trivial example:
#### Backend (FastAPI with Redis Pub/Sub)
```python
# server.py
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import redis
import json
import asyncio

app = FastAPI()
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Endpoint to set cache and publish update
@app.post("/set-cache/{key}")
async def set_cache(key: str, value: str):
    redis_client.set(key, value)
    redis_client.publish("cache_updates", json.dumps({"key": key, "value": value}))
    return {"status": "set"}

# SSE endpoint to stream Redis Pub/Sub updates
@app.get("/cache-stream")
async def cache_stream():
    async def event_generator():
        pubsub = redis_client.pubsub()
        pubsub.subscribe("cache_updates")
        try:
            while True:
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.01)  # Yield control to event loop
        finally:
            pubsub.unsubscribe()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

