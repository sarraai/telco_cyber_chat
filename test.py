import os, sys
from langgraph_sdk import get_sync_client

RESOURCE_URL = os.getenv("RESOURCE_URL")
API_KEY = os.getenv("LANGSMITH_API_KEY")

def mask(s): 
    return s[:4] + "…" + s[-4:] if s and len(s) > 8 else str(bool(s))

print("Python:", sys.version.split()[0])
print("RESOURCE_URL:", RESOURCE_URL)
print("LANGSMITH_API_KEY:", mask(API_KEY))

assert RESOURCE_URL, "Set RESOURCE_URL"
assert API_KEY, "Set LANGSMITH_API_KEY"

client = get_sync_client(url=RESOURCE_URL, api_key=API_KEY)
print(client.assistants.search())
