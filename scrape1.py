import requests
from bs4 import BeautifulSoup
import json
import time


def parse_content(content):
    # Parse the HTML
    soup = BeautifulSoup(content, 'html.parser')

    # Find all divs that contains the Q&A content
    entry_contents = soup.find_all('div', class_='entry-content')

    # Initialize the Q&A list
    qa_list = []

    # Loop through each entry_content
    for entry_content in entry_contents:
        # Find all h4 (speakers) and p (messages) elements
        speakers = entry_content.find_all('h4', class_='entry-speaker')
        messages = entry_content.find_all('p')

        # Initialize question and answer variables
        question = None
        answer = None

        # Loop through all speakers and their corresponding messages
        for speaker, message in zip(speakers, messages):
            # If speaker is Brandon Sanderson, then the message is an answer
            if speaker.text.strip() == "Brandon Sanderson":
                answer = message.text.strip()
            else:  # Otherwise, the message is a question
                question = message.text.strip()

        # Append the Q&A pair to the list
        if question and answer:
            qa_list.append({"instruction": question, "input": "","output": answer})

    return qa_list


def fetch_and_parse(url):
    # Send HTTP request to the URL
    response = requests.get(url)

    # If the request was successful, parse the content
    if response.status_code == 200:
        return parse_content(response.content)
        #return []
    else:
        return []

def fetch_and_parse_event_links(url):
    # Send HTTP request to the URL
    response = requests.get(url)

    # If the request was successful, parse the content
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        event_links = ["https://wob.coppermind.net" + link.get('href') for link in soup.select('li.event a')]
        return event_links
    else:
        return []


# Initialize the list to hold all Q&A pairs
all_qa_pairs = []

# Define the base URL
base_url = "https://wob.coppermind.net"

for i in range(1,11):
    url = base_url + "/events/?page="+str(i)+"&"
    print(f"Processing archive page {url}")
    # Fetch event links from the current URL
    event_links = fetch_and_parse_event_links(url)

    for event_link in event_links:
        #print(f"Processing event page {event_link}")
        # Fetch and parse Q&A pairs from the event URL
        qa_pairs = fetch_and_parse(event_link)

        # Append Q&A pairs to the list of all Q&A pairs
        all_qa_pairs.extend(qa_pairs)

    # Parse the HTML
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    time.sleep(10)

# Convert the list of all Q&A pairs to JSON
all_qa_pairs_json = json.dumps(all_qa_pairs)
print(all_qa_pairs_json)
