The sample messaging api endpoint only performs the following:
Loads the avatar (loads the base model and adapter) (first request is slow; load the avatar on first inference (first query))
switch adapters (collect new adapters upon switch if they are not available)
return inference to a query
---
# The chroma db needs to be it's own container. This allows for the instance to be updated and queried for low cost (free). Adds latency
update the (replace) the chroma_db
The creation of adapters, management of the chroma_db, and update & training of the adapters happens on a separte


I need an image to manage the chroma_db & accept & return inference
I need an image to manage the adapter & adapter related material


