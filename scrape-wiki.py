import os
import json
import requests
import psutil
from tqdm import tqdm

# Step 1: Use Apify API
api_token = 'apify_api_7h0HPgNVrUJiCcIfJb94aeFog9WVcQ2C4IXG'  # Replace with your Apify API token
actor_id = 'apify/website-content-crawler'  # Replace with your actor id

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Token {api_token}',
}

# Trigger the actor
input_data = {
    'startUrls': [
        {'url': 'https://coppermind.net/wiki'},  # Replace with the URL to scrape
    ],
    # Add any other necessary fields for your actor here.
}

# Trigger the actor
response = requests.post(
    f'https://api.apify.com/v2/actors/{actor_id}/runs',
    headers=headers,
    json=input_data,
)

run_id = response.json().get('id')

# Wait for the actor to finish
while True:
    response = requests.get(
        f'https://api.apify.com/v2/actor-runs/{run_id}',
        headers=headers,
    )

    status = response.json().get('status')

    if status == 'SUCCEEDED':
        break
    elif status == 'FAILED':
        raise RuntimeError('The actor run has failed.')

# Retrieve the results
response = requests.get(
    f'https://api.apify.com/v2/actor-runs/{run_id}/dataset/items',
    headers=headers,
)

data = response.json()

# Step 2: Save the scraped content
chunk_size = 100  # Define the size of each file chunk
chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

for i, chunk in enumerate(tqdm(chunks, desc='Saving data chunks')):
    with open(f'chunk_{i}.json', 'w') as f:
        json.dump(chunk, f)

    # Step 3: Display the file system usage
    usage = psutil.disk_usage('/')
    print(f'File system usage: {usage.percent}%')
